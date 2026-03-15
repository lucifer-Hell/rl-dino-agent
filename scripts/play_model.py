from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from stable_baselines3 import DQN

from rl_dino_agent.config import AppConfig
from rl_dino_agent.training.factory import build_vector_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained RL Dino model play.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--sleep-after-episode", type=float, default=1.0)
    parser.add_argument("--record-video-dir", type=Path, default=None)
    return parser.parse_args()


def print_summary(console: Console, model_path: Path, episodes: int, headless: bool) -> None:
    table = Table(title="Model Playback")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Model", str(model_path))
    table.add_row("Episodes", str(episodes))
    table.add_row("Headless", str(headless))
    console.print(table)


def main() -> None:
    args = parse_args()
    console = Console()
    config = AppConfig.load(args.config)
    if args.visible:
        config.game.browser.headless = False
    elif args.headless:
        config.game.browser.headless = True
    if args.record_video_dir is not None:
        args.record_video_dir.mkdir(parents=True, exist_ok=True)
        config.game.browser.record_video_dir = str(args.record_video_dir.resolve())

    env = build_vector_env(config)
    model = DQN.load(str(args.model.resolve()), env=env, device=config.training.device)
    print_summary(console, args.model.resolve(), args.episodes, config.game.browser.headless)

    episode_scores: list[int] = []
    episode_rewards: list[float] = []

    try:
        for episode_idx in range(1, args.episodes + 1):
            observation = env.reset()
            done = False
            total_reward = 0.0
            last_score = 0

            while not done:
                action, _ = model.predict(observation, deterministic=args.deterministic)
                observation, rewards, dones, infos = env.step(action)
                total_reward += float(rewards[0])
                done = bool(dones[0])
                last_score = int(infos[0].get("score", last_score))

            episode_scores.append(last_score)
            episode_rewards.append(total_reward)
            console.print(
                f"episode={episode_idx} score={last_score} total_reward={total_reward:.3f}"
            )
            time.sleep(max(0.0, args.sleep_after_episode))

        console.print(
            f"[bold green]Done.[/bold green] mean_score={np.mean(episode_scores):.2f} "
            f"mean_reward={np.mean(episode_rewards):.3f}"
        )
    finally:
        env.close()
        if args.record_video_dir is not None:
            videos = sorted(
                args.record_video_dir.glob("*.webm"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if videos:
                console.print(f"video={videos[0].resolve()}")


if __name__ == "__main__":
    main()
