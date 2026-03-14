# RL Dino Agent

A separate reinforcement learning project for training an agent to play [`rl-dino`](D:\rl-dino) from browser visuals.

## Approach

This repo starts with `DQN` rather than PPO because the action space is tiny:

- `0`: idle
- `1`: jump

The agent still learns from rendered frames, not handcrafted state vectors. We use the game's browser API only for:

- stepping the simulation
- receiving reward and done signals
- resetting episodes

That keeps the setup close to a real "see screen, press control" loop while staying practical enough to iterate on quickly.

## What This Repo Includes

- a browser-backed Gymnasium environment
- visual frame capture from the Dino canvas
- frame preprocessing and stacking
- config-driven experiments via YAML
- verbose Stable Baselines3 DQN training
- automatic CSV metrics and reward/loss plots
- run folders for checkpoints and training artifacts

## Layout

```text
configs/
  default.yaml
scripts/
  train.py
src/rl_dino_agent/
  config.py
  envs/browser_dino_env.py
  training/callbacks.py
  training/factory.py
  utils/plotting.py
  utils/server.py
```

## Setup

1. Create a virtual environment:

   ```powershell
   C:\Users\panka\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the package:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -e .
   python -m playwright install chromium
   ```

3. Make sure the game repo exists at `D:\rl-dino`, or update `configs/default.yaml`.

Or do the full setup in one shot:

```powershell
.\scripts\setup.ps1
```

## Train

```powershell
python .\scripts\train.py --config .\configs\default.yaml
```

Resume from the saved partial model:

```powershell
python .\scripts\train.py --config .\configs\default.yaml --resume-from .\artifacts\best_partial_model_140k_steps.zip
```

Artifacts are written under `runs/<timestamp>-<run_name>/`.

## Watch Training

- the browser stays visible by default, so you can see the agent act
- TensorBoard logs are written into each run folder
- CSV metrics are written to `metrics.csv`
- plots are refreshed into `training_curves.png`

Start TensorBoard with:

```powershell
.\scripts\start_tensorboard.ps1
```

## Record Demos

You can record your own play sessions for later imitation learning or fine-tuning:

```powershell
python .\scripts\record_demos.py --episodes 5
```

This writes compressed trajectories under `demos/episode_###/` with frames, actions, rewards, and metadata.

## Notes

- The training pipeline is intentionally config-first so we can plug in Optuna later without restructuring the repo.
- The default setup keeps the browser visible so you can watch the agent play while metrics are being logged.
