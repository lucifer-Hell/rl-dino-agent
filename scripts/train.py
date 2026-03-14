from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rl_dino_agent.config import AppConfig
from rl_dino_agent.training.callbacks import build_callback_list
from rl_dino_agent.training.factory import build_dqn_model, build_vector_env
from rl_dino_agent.utils.plotting import initialize_run_dir, persist_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RL Dino visual agent.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the training config YAML.",
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


def main() -> None:
    args = parse_args()
    console = Console()
    config = AppConfig.load(args.config)
    run_dir = initialize_run_dir(config)
    persist_run_metadata(config, args.config.resolve(), run_dir)
    print_summary(console, config, run_dir)

    env = build_vector_env(config)
    model = build_dqn_model(
        config,
        env=env,
        tensorboard_log=str(run_dir / config.training.tensorboard_log_subdir),
    )
    callbacks = build_callback_list(
        run_dir=run_dir,
        save_checkpoint_every_steps=config.training.save_checkpoint_every_steps,
        plot_every_episodes=config.training.plot_every_episodes,
        verbose=config.training.verbose,
    )

    try:
        model.learn(
            total_timesteps=config.training.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name=config.run.name,
        )
        model.save(str(run_dir / "final_model"))
        console.print(f"[bold green]Training complete.[/bold green] Model saved to {run_dir / 'final_model.zip'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
