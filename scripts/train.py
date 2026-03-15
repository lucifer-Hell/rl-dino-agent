from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table
from stable_baselines3 import DQN
from stable_baselines3.common.utils import LinearSchedule

from rl_dino_agent.config import AppConfig
from rl_dino_agent.training.callbacks import build_callback_list
from rl_dino_agent.training.factory import (
    build_dqn_model,
    build_eval_vector_env,
    build_vector_env,
)
from rl_dino_agent.utils.plotting import initialize_run_dir, persist_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RL Dino visual agent.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the training config YAML.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint zip to resume training from.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Optional total timesteps for this training invocation.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless browser mode for training stability.",
    )
    parser.add_argument(
        "--resume-final-epsilon",
        action="store_true",
        help="When resuming, keep epsilon fixed at the configured final value instead of re-annealing.",
    )
    return parser.parse_args()


def print_summary(console: Console, config: AppConfig, run_dir: Path) -> None:
    table = Table(title="RL Dino Agent Training")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Algorithm", "DQN")
    table.add_row("Game repo", str(config.game.repo_path))
    table.add_row("Frames", f"{config.game.env.frame_width}x{config.game.env.frame_height} x {config.game.env.frame_stack}")
    table.add_row("Headless", str(config.game.browser.headless))
    table.add_row("Timesteps", str(config.training.total_timesteps))
    table.add_row("Run dir", str(run_dir))
    console.print(table)


def freeze_resume_exploration(model: DQN, final_epsilon: float) -> None:
    model.exploration_initial_eps = final_epsilon
    model.exploration_final_eps = final_epsilon
    model.exploration_fraction = 0.0
    model.exploration_schedule = LinearSchedule(
        start=final_epsilon,
        end=final_epsilon,
        end_fraction=0.0,
    )
    model.exploration_rate = final_epsilon


def main() -> None:
    args = parse_args()
    console = Console()
    config = AppConfig.load(args.config)
    if args.timesteps is not None:
        config.training.total_timesteps = args.timesteps
    if args.headless:
        config.game.browser.headless = True
    run_dir = initialize_run_dir(config)
    persist_run_metadata(config, args.config.resolve(), run_dir)
    print_summary(console, config, run_dir)

    env = build_vector_env(config)
    eval_env = None
    try:
        eval_env = build_eval_vector_env(config)
    except Exception as exc:
        console.print(
            "[bold yellow]Evaluation disabled:[/bold yellow] "
            f"{exc}"
        )
    tensorboard_log = str(run_dir / config.training.tensorboard_log_subdir)
    if args.resume_from is not None:
        console.print(f"[bold yellow]Resuming from[/bold yellow] {args.resume_from}")
        model = DQN.load(
            str(args.resume_from.resolve()),
            env=env,
            device=config.training.device,
            tensorboard_log=tensorboard_log,
        )
        model.verbose = config.training.verbose
        if args.resume_final_epsilon:
            freeze_resume_exploration(model, config.training.exploration_final_eps)
            console.print(
                f"[bold yellow]Resume exploration locked[/bold yellow] at epsilon={config.training.exploration_final_eps}"
            )
    else:
        model = build_dqn_model(
            config,
            env=env,
            tensorboard_log=tensorboard_log,
        )
    callbacks = build_callback_list(
        run_dir=run_dir,
        config_path=args.config.resolve(),
        save_checkpoint_every_steps=config.training.save_checkpoint_every_steps,
        keep_last_checkpoints=config.training.keep_last_checkpoints,
        save_replay_buffer_checkpoints=config.training.save_replay_buffer_checkpoints,
        plot_every_episodes=config.training.plot_every_episodes,
        verbose=config.training.verbose,
        eval_env=eval_env,
        eval_freq_steps=config.evaluation.eval_freq_steps,
        eval_episodes=config.evaluation.episodes,
        eval_deterministic=config.evaluation.deterministic,
        early_stop_patience_episodes=config.training.early_stop_patience_episodes,
        early_stop_min_episodes=config.training.early_stop_min_episodes,
        early_stop_min_timesteps=config.training.early_stop_min_timesteps,
        early_stop_window_episodes=config.training.early_stop_window_episodes,
        early_stop_metric=config.training.early_stop_metric,
        early_stop_min_delta=config.training.early_stop_min_delta,
        demo_every_steps=config.training.demo_every_steps,
        demo_episodes=config.training.demo_episodes,
        demo_deterministic=config.training.demo_deterministic,
        demo_headless=config.training.demo_headless,
        demo_sleep_after_episode=config.training.demo_sleep_after_episode,
    )

    try:
        model.learn(
            total_timesteps=config.training.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name=config.run.name,
            reset_num_timesteps=args.resume_from is None,
        )
        model.save(str(run_dir / "final_model"))
        console.print(f"[bold green]Training complete.[/bold green] Model saved to {run_dir / 'final_model.zip'}")
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
