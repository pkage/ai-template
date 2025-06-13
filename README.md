# pat's ai template

This repository serves as a basic scaffold for building AI research projects. This includes:

- Main stack
    - `torch`
    - `einops`
- Support tools
    - `rich`
    - `asyncclick`
    - `wandb`
- Dev tools:
    - `ruff`
    - `ty`
    - `ds-run`
    - `uv`

## getting started

You may wish to install the `ds_run` git hooks (may require [this PR to be merged]()):

```bash
$ ds --sync-git-hooks
```

and configure your HPC solution via Git:

```bash
$ ssh user@host "cd path/to/repo; git init; git config receive.denyCurrentBranch updateInstead"
$ git remote add deploy user@host:path/to/repo
```

