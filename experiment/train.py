from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import Compose
import wandb

from . import console
from .model import Model


# --- MODELS ---


@dataclass
class ModelParams:
    # probably need to change these for your hardware
    batch_size: int = 32

    # hyperparams, total guess for now
    epochs: int = 10
    learning_rate: float = 1e4


@dataclass
class TrainParams:
    run_name: str
    run_group: str
    checkpoint_dir: Path
    device: str
    amp: bool
    nccl_bind: str
    resume: bool = True


# --- COMMON ---


def setup_gloo(rank: int, world_size: int) -> dist.HashStore:
    store = dist.HashStore()

    # gloo for cpu
    dist.init_process_group(
        backend='gloo', store=store, world_size=world_size, rank=rank
    )

    return store


def setup_nccl(rank: int, world_size: int, tcp_bind: str):
    """
    Setup NCCL process group
    """
    # tcp_bind = 'tcp://10.1.1.20:23456'
    dist.init_process_group(
        backend='nccl', init_method=tcp_bind, rank=rank, world_size=world_size
    )


def cleanup():
    # Clean up the process group
    dist.destroy_process_group()


def train(
    rank: int, world_size: int, model_params: ModelParams, train_params: TrainParams
):
    # dist setup
    setup_nccl(rank, world_size, train_params.nccl_bind)

    if rank == 0:
        wandb_project = os.getenv('WANDB_PROJECT')
        if wandb_project is None:
            console.print('[yellow]wandb project is None, this might be a problem!')
        console.print(f'wandb project: {wandb_project}')

        wandb.init(
            project=wandb_project,
            group=train_params.run_group,
            config={**asdict(model_params), **asdict(train_params)},
        )
    # pick the device
    device = torch.device(f'cuda:{rank}')

    # REPLACE ME
    dataset = Dataset()

    # sampler and loaders
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=model_params.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # amp maybe?
    scaler = None
    if train_params.amp:
        scaler = GradScaler()

    # create the model
    model = Model()
    model.to(device)

    # create the loss
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optimizer = Adam(model.parameters(), lr=model_params.learning_rate)

    # Load from checkpoint if resume is True
    if train_params.resume:
        checkpoint_path = (
            train_params.checkpoint_dir
            / f'checkpoint_{train_params.run_name}_latest.pth'
        )
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            console.print(f'Resuming training from epoch {start_epoch}')
        else:
            console.print('[red]Checkpoint not found, starting from scratch')
            start_epoch = 0
    else:
        start_epoch = 0

    # now we begin!
    for epoch in range(start_epoch, model_params.epochs):
        console.print(
            f'beginning epoch {epoch}/{model_params.epochs} (rank {rank}/{world_size})'
        )
        model.train()

        # make sure each process gets different data
        sampler.set_epoch(epoch)

        total_loss = 0
        for batch_idx, batch in enumerate(loader):
            samples, labels = batch.float().to(device)

            optimizer.zero_grad()

            if train_params.amp:
                # AMP pass
                with autocast('cuda'):
                    predictions = model(
                        samples,
                        rank=rank,
                        world_size=world_size,
                    )

                    loss = loss_fn(labels, predictions)

                assert (
                    scaler is not None
                )  # not necessary but keeps my editor from yelling at me

                # AMP backward pass
                scaler.scale(loss).backward()

                # AMP optimizer step
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
            else:
                # no AMP training
                predictions = model(
                    samples,
                    rank=rank,
                    world_size=world_size,
                )

                loss = loss_fn(samples, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if rank == 0 and batch_idx % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{model_params.epochs}], Step [{batch_idx + 1}/{len(loader)}], Loss: {loss.item():.4f}'
                )
                wandb.log(
                    {
                        'epoch': epoch + 1,
                        'batch_idx': batch_idx + 1,
                        'train_loss': loss.item(),
                    }
                )

        # Save the model checkpoint after each epoch (only rank 0 saves to avoid duplication)
        if rank == 0:
            model_path = (
                train_params.checkpoint_dir
                / f'checkpoint_{train_params.run_name}_epoch_{epoch}.pth'
            )
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                model_path,
            )

            # Save the latest checkpoint
            latest_model_path = (
                train_params.checkpoint_dir
                / f'checkpoint_{train_params.run_name}_latest.pth'
            )
            shutil.copy(model_path, latest_model_path)
