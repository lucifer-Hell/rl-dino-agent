from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import optuna
from rich.console import Console
from rich.table import Table

from rl_dino_agent.config import AppConfig
from rl_dino_agent.training.callbacks import build_callback_list
from rl_dino_agent.training.factory import build_dqn_model, build_vector_env
from rl_dino_agent.utils.plotting import initialize_run_dir, persist_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune RL Dino DQN hyperparameters with Optuna.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=12000)
    parser.add_argument("--study-name", type=str, default="rl-dino-dqn-optuna")
    parser.add_argument("--storage", type=str, default=None, help="Optional Optuna storage URL.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Optional base model to warm-start each trial.")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def suggest_config(trial: optuna.Trial, config: AppConfig) -> AppConfig:
    config.training.learning_rate = trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True)
    config.training.learning_rate_end = trial.suggest_float("learning_rate_end", 1e-5, 8e-5, log=True)
    config.training.gamma = trial.suggest_float("gamma", 0.97, 0.995)
    config.training.batch_size = trial.suggest_categorical("batch_size", [32, 64])
    config.training.buffer_size = trial.suggest_categorical("buffer_size", [50000, 75000, 100000])
    config.training.target_update_interval = trial.suggest_categorical("target_update_interval", [500, 1000, 2000, 4000])
    config.training.exploration_fraction = trial.suggest_float("exploration_fraction", 0.05, 0.2)
    config.game.env.jump_penalty = trial.suggest_float("jump_penalty", 0.0, 0.03)
    config.game.env.obstacle_clear_bonus = trial.suggest_float("obstacle_clear_bonus", 0.1, 0.6)
    config.game.env.unsafe_descent_penalty = trial.suggest_float("unsafe_descent_penalty", 0.02, 0.2)
    config.game.env.unsafe_descent_distance = trial.suggest_float("unsafe_descent_distance", 24.0, 52.0)
    config.game.env.crop_bottom = trial.suggest_categorical("crop_bottom", [16, 24, 32, 40])
    return config


def summarize_trial(console: Console, trial_number: int, run_dir: Path, summary: dict[str, float]) -> None:
    table = Table(title=f"Optuna Trial {trial_number}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Run dir", str(run_dir))
    for key, value in summary.items():
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    console.print(table)


def load_metrics(metrics_path: Path) -> dict[str, float]:
    rows = metrics_path.read_text(encoding="utf-8").strip().splitlines()[1:]
    if not rows:
        return {"episodes": 0.0, "best_score": 0.0, "mean_last5_score": 0.0, "mean_last5_reward": 0.0}

    parsed = []
    for row in rows:
        episode, reward, length, score = row.split(",")
        parsed.append(
            {
                "episode": int(episode),
                "reward": float(reward),
                "length": int(length),
                "score": int(score),
            }
        )

    tail = parsed[-5:]
    return {
        "episodes": float(len(parsed)),
        "best_score": float(max(item["score"] for item in parsed)),
        "mean_last5_score": float(mean(item["score"] for item in tail)),
        "mean_last5_reward": float(mean(item["reward"] for item in tail)),
    }


def objective_factory(args: argparse.Namespace, console: Console):
    def objective(trial: optuna.Trial) -> float:
        config = AppConfig.load(args.config)
        config.run.name = f"optuna-trial-{trial.number:03d}"
        config.training.total_timesteps = args.timesteps
        config.training.verbose = 0
        config.training.plot_every_episodes = 5
        config.training.save_checkpoint_every_steps = max(args.timesteps, 10000)
        config.training.keep_last_checkpoints = 1
        config.evaluation.episodes = 1
        if args.headless:
            config.game.browser.headless = True

        config = suggest_config(trial, config)
        run_dir = initialize_run_dir(config)
        persist_run_metadata(config, args.config.resolve(), run_dir)

        env = build_vector_env(config)
        model = build_dqn_model(config, env, str(run_dir / config.training.tensorboard_log_subdir))
        callbacks = build_callback_list(
            run_dir=run_dir,
            save_checkpoint_every_steps=config.training.save_checkpoint_every_steps,
            keep_last_checkpoints=config.training.keep_last_checkpoints,
            save_replay_buffer_checkpoints=config.training.save_replay_buffer_checkpoints,
            plot_every_episodes=config.training.plot_every_episodes,
            verbose=0,
        )

        try:
            model.learn(
                total_timesteps=config.training.total_timesteps,
                callback=callbacks,
                progress_bar=False,
                tb_log_name=config.run.name,
            )
            summary = load_metrics(run_dir / "metrics.csv")
            score = summary["mean_last5_score"] + 0.25 * summary["best_score"]
            trial.set_user_attr("run_dir", str(run_dir))
            trial.set_user_attr("summary", summary)
            (run_dir / "trial_summary.json").write_text(
                json.dumps(
                    {
                        "objective": score,
                        "summary": summary,
                        "params": trial.params,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            summarize_trial(console, trial.number, run_dir, summary)
            return score
        finally:
            env.close()

    return objective


def main() -> None:
    args = parse_args()
    console = Console()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
    )
    console.print(
        f"Starting Optuna study '{args.study_name}' with {args.trials} trials at {args.timesteps} timesteps per trial."
    )
    study.optimize(objective_factory(args, console), n_trials=args.trials)

    best = {
        "value": study.best_value,
        "params": study.best_params,
        "run_dir": study.best_trial.user_attrs.get("run_dir"),
        "summary": study.best_trial.user_attrs.get("summary"),
    }

    output_dir = Path("runs") / "optuna"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.study_name}_best.json").write_text(
        json.dumps(best, indent=2),
        encoding="utf-8",
    )

    table = Table(title="Optuna Best Trial")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Objective", f"{study.best_value:.4f}")
    table.add_row("Run dir", str(best["run_dir"]))
    for key, value in study.best_params.items():
        table.add_row(key, str(value))
    console.print(table)


if __name__ == "__main__":
    main()
