# RL Dino Agent

This repository trains a reinforcement learning agent to play the Chrome Dino-style game from raw browser frames.

In simple words:

- the agent looks at the game screen
- it decides whether to `do nothing` or `jump`
- it learns by trial and error
- we improve it with tuning and longer training runs

The project is built on top of the game repo:

- https://github.com/lucifer-Hell/rl-dino

For local training, that repo still needs to be cloned somewhere on your machine, and the config files must point to that local folder.

## Experiment Overview

This repository is an experiment in teaching a game-playing agent from vision alone.

The big idea is:

- show the agent the game screen
- let it choose between `idle` and `jump`
- reward it for surviving and clearing obstacles
- improve the training setup over multiple runs

We did not try to solve the game with hardcoded rules. Instead, we treated it like a small research project:

1. create a browser-based RL environment
2. train a visual DQN baseline
3. shape the reward so learning is faster and more stable
4. tune the hyperparameters with Optuna
5. continue training from the best checkpoints
6. evaluate how well the tuned settings transfer to longer runs

What makes this experiment interesting is that the agent learns from image frames, not from custom game logic such as "if obstacle distance < X, then jump".

## At A Glance

- task: play a Dino runner game from browser visuals
- algorithm: `DQN`
- actions: `idle` or `jump`
- input: stacked grayscale frames
- backbone: custom CNN feature extractor
- tuning method: Optuna
- training style: config-driven experiments with checkpoint resume

## What This Repo Does

This is a visual RL project, not a rules-based bot.

That means:

- we do **not** hand-code "jump when obstacle is X pixels away"
- we let the model learn patterns from image frames
- the game is played through a browser-backed Gym environment

The browser API is only used for:

- resetting the game
- stepping the game forward
- reading reward, score, and done signals

## Quick Demo

If you want to see the current trained model play in a browser, run:

```powershell
python .\scripts\play_model.py --config .\configs\optuna_phase2_best_resume_30k_steps.yaml --model .\runs\20260315-165352-dqn-optuna-phase2-best-resume-30k-steps\best_score_model.zip --episodes 3 --visible --deterministic
```

## Demo Video

A recorded demo is included in this repo:

[![RL Dino demo](assets/demo.gif)](assets/demo.webm)

Direct video file:

[Watch the demo video](assets/demo.webm)

This demo was recorded from the current trained model and gives users a quick way to see what the project has built without setting up the environment first.

## Strategy We Used

We chose `DQN` because the action space is very small:

- `0`: idle
- `1`: jump

That makes DQN a natural first choice.

Our training strategy was:

1. build a visual DQN baseline
2. add reward shaping so the agent learns survival and timing faster
3. tune the important hyperparameters with Optuna
4. take the best Optuna result and run longer GPU training
5. resume from the best checkpoints instead of restarting every time

## Reward Setup

The main reward mode we used is `clip_survival`.

In beginner-friendly terms:

- if the Dino dies, reward becomes `-1.0`
- if the Dino survives, reward is clipped to a small positive value
- then we add extra bonuses and penalties to guide learning

Extra reward terms used in the tuned runs:

- `jump_penalty`
- `obstacle_clear_bonus`
- `unsafe_descent_penalty`

This helps teach the model:

- jumping too much is bad
- clearing obstacles is good
- unsafe landings should be discouraged

## DQN Architecture We Used

We use Stable Baselines3 `DQN` with `CnnPolicy`.

But we do **not** use the default image backbone. We use a custom CNN called `MediumDinoCNN` from [extractors.py](/d:/rl-dino-agent/src/rl_dino_agent/training/extractors.py).

The CNN looks like this:

- `Conv2d(4, 32, kernel_size=5, stride=2, padding=1)`
- `Conv2d(32, 64, kernel_size=3, stride=2, padding=1)`
- `Conv2d(64, 96, kernel_size=3, stride=2, padding=1)`
- `Conv2d(96, 128, kernel_size=3, stride=1, padding=1)`
- `Flatten`
- `Linear(..., 256)`
- `ReLU`

Then the DQN head uses:

- `q_net_arch: [256, 128]`

Input format:

- grayscale frames
- image size `84 x 84`
- frame stack of `4`

## About Optuna

Optuna is a hyperparameter tuning library.

Instead of manually trying values one by one, Optuna runs multiple experiments and searches for better settings automatically.

In this repo, Optuna was used to tune:

- learning rate
- learning rate end value
- gamma
- batch size
- buffer size
- target update interval
- exploration fraction
- jump penalty
- obstacle clear bonus
- unsafe descent penalty
- unsafe descent distance
- bottom crop amount

The tuner script is:

- [tune_optuna.py](/d:/rl-dino-agent/scripts/tune_optuna.py)

Best Optuna study summary:

- file: [rl-dino-gpu-phase2_best.json](/d:/rl-dino-agent/runs/optuna/rl-dino-gpu-phase2_best.json)
- best trial run: [20260315-134235-optuna-trial-007](/d:/rl-dino-agent/runs/20260315-134235-optuna-trial-007)
- best short-run score during tuning: `501`

Best parameters found in that study:

- learning rate: `0.00016047603182752895`
- learning rate end: `0.000011629939087813681`
- gamma: `0.9891830368631348`
- batch size: `32`
- buffer size: `50000`
- target update interval: `500`
- exploration fraction: `0.12741026988907145`
- jump penalty: `0.025309667634950364`
- obstacle clear bonus: `0.331522607507297`
- unsafe descent penalty: `0.02405156938736382`
- unsafe descent distance: `47.93211318146528`
- crop bottom: `32`

## Fine-Tuning Effort So Far

By "amount spent in fine-tuning", this README means local training effort, not money.

Main training runs:

- baseline visible tuned run: `49,727` logged steps, best score `368`
- first long GPU run: `80,180` logged steps, best score `327`
- resumed run with browser demos: `32,519` logged steps, best score `438`
- resumed run without demos: `124,414` logged steps, best score `443`

Total main training effort:

- about `286,840` logged environment steps

Optuna tuning effort:

- `8` main GPU trials in the phase-2 study
- about `171,690` logged steps across those trials
- plus `1` smoke-test trial used to verify the tuning loop

Combined observed effort so far:

- about `458,530` logged environment steps

Training was done locally, including GPU-backed runs after switching PyTorch to CUDA.

## Best Result So Far

Best score seen during Optuna tuning:

- `501` in [20260315-134235-optuna-trial-007](/d:/rl-dino-agent/runs/20260315-134235-optuna-trial-007)

Best score from longer continuation training:

- `443` in [20260315-165352-dqn-optuna-phase2-best-resume-30k-steps](/d:/rl-dino-agent/runs/20260315-165352-dqn-optuna-phase2-best-resume-30k-steps)

Current project status:

- the agent clearly learns
- tuning improved the short-run behavior
- long-run training is still somewhat unstable

So the project is promising, but not fully solved yet.

## Repo Layout

```text
configs/
  default.yaml
  optuna_*.yaml
scripts/
  train.py
  tune_optuna.py
  play_model.py
src/rl_dino_agent/
  config.py
  envs/browser_dino_env.py
  training/extractors.py
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

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
python -m playwright install chromium
```

3. Clone the game repo locally:

```powershell
git clone https://github.com/lucifer-Hell/rl-dino D:\rl-dino
```

If you clone it somewhere else, update the `repo_path` value in the YAML config you want to use.

Or run everything in one step:

```powershell
.\scripts\setup.ps1
```

## How To Train

Baseline training:

```powershell
python .\scripts\train.py --config .\configs\default.yaml
```

Run Optuna tuning:

```powershell
python .\scripts\tune_optuna.py --config .\configs\optuna_tune_gpu.yaml --trials 8 --timesteps 40000 --study-name rl-dino-gpu-phase2 --storage sqlite:///runs/optuna/rl_dino_gpu_phase2.db --headless
```

Resume from a good checkpoint:

```powershell
python .\scripts\train.py --config .\configs\optuna_phase2_best_resume_30k_steps.yaml --resume-from .\runs\20260315-165352-dqn-optuna-phase2-best-resume-30k-steps\best_score_model.zip
```

## Notes

- training is config-driven, so experiments are easier to reproduce
- the repo supports GPU training when CUDA PyTorch is installed
- plateau stopping was added to stop wasting time on clearly stagnant runs
- live evaluation in a separate browser is still limited by the current Playwright sync setup

## Glossary

Here are simple definitions for the main terms used in this repo.

`Reinforcement Learning (RL)`

- A type of machine learning where an agent learns by trying actions, seeing results, and improving over time.

`Agent`

- The model that makes decisions in the game.

`Environment`

- The game world the agent interacts with. In this repo, the environment is the browser-backed Dino game.

`Observation`

- What the agent sees at each step. Here, that is a stack of processed game frames.

`Action`

- A decision the agent can take. In this project the actions are only `idle` and `jump`.

`Reward`

- A number telling the agent whether what it just did was good or bad.

`Episode`

- One full run of the game, from reset until the Dino dies or the maximum episode step limit is reached.

`DQN`

- Deep Q-Network. A reinforcement learning algorithm that learns how good each action is in a given situation.

`Q-value`

- The model's estimate of how useful an action is from the current state.

`CNN`

- Convolutional Neural Network. A neural network type that works well on images.

`Frame Stack`

- Several recent frames combined together so the model can understand motion, not just a single still image.

`Hyperparameters`

- Settings chosen by us before training, such as learning rate, batch size, or gamma.

`Optuna`

- A tool that automatically searches for better hyperparameters by running many experiments.

`Checkpoint`

- A saved copy of the model during training that can be loaded later to continue training or run a demo.

`Fine-tuning`

- Continuing training from a previously trained model instead of starting from random weights again.

`Replay Buffer`

- A memory of past experiences used by DQN to learn from older game situations.

`Exploration`

- When the agent tries different actions instead of always following its current best guess.

`Plateau Stopping`

- Automatically stopping a run when the recent training scores stop improving for long enough.
