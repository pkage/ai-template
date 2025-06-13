from pathlib import Path

import asyncclick as click
from dotenv import load_dotenv
from rich.table import Table
from torch import cuda
import torch.distributed as dist
import torch.multiprocessing as mp

from . import console, pprint
from .train import ModelParams, TrainParams, cleanup, train


# pre-work
load_dotenv()


@click.group()
def cli():
    pass


@cli.command('hardware', help='show hardware status')
def cli_hardware():
    def available(flag):
        if flag:
            return '[green]available[/]'
        return '[red]unavailable[/]'

    table = Table('feature', 'status', show_edge=False)

    table.add_row('cuda', available(cuda.is_available()))
    table.add_row('cuda/devices', f'{cuda.device_count()}')
    table.add_row('dist', available(dist.is_available()))
    table.add_row('dist/nccl', available(dist.is_nccl_available()))
    table.add_row('dist/gloo', available(dist.is_gloo_available()))

    console.print(table)


@cli.command('train', help='train hook')
@click.option(
    '--ckpts',
    type=click.Path(dir_okay=True, file_okay=False),
    required=True,
    help='path to write checkpoints to',
)
@click.option('--run-name', type=str, required=True, help='run name')
@click.option('--run-group', type=str, help='run group (for wandb)')
@click.option(
    '-l', '--learning-rate', type=float, required=True, help='learning rate (Adam)'
)
@click.option('-e', '--epochs', type=int, required=True, help='epoch count')
@click.option('-b', '--batch-size', type=int, required=True, help='batch size')
@click.option(
    '-d',
    '--device',
    type=click.Choice(['cpu', 'cuda', 'mps']),
    required=True,
    help='device to run on',
)
@click.option('--no-resume', is_flag=True, help='always start from scratch')
@click.option(
    '--nccl-bind',
    type=str,
    default='tcp://localhost:33445',
    help='distributed synchronization store',
)
@click.option('--amp', type=bool, is_flag=True, help='enable automatic mixed precision')
@click.option(
    '--half-resolution',
    type=bool,
    is_flag=True,
    help='train at half resolution to conserve vram',
)
def cli_train(
    ckpts,
    run_name,
    run_group,
    learning_rate,
    epochs,
    batch_size,
    device,
    no_resume,
    nccl_bind,
    amp,
    half_resolution,
):
    ckpts = Path(ckpts)
    ckpts.mkdir(exist_ok=True, parents=True)

    salibi_params = ModelParams(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
    )
    train_params = TrainParams(
        checkpoint_dir=ckpts,
        run_name=run_name,
        run_group=run_group,
        device=device,
        amp=amp,
        nccl_bind=nccl_bind,
        resume=not no_resume,
    )

    console.print('parsed parameters:')
    pprint(salibi_params)

    world_size = cuda.device_count()
    console.print(f'using world size of {world_size}')

    mp.spawn(
        train, (world_size, salibi_params, train_params), nprocs=world_size, join=True
    )

    cleanup()
