from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList


class TrainingArtifactsCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        plot_every_episodes: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.run_dir = run_dir
        self.plot_every_episodes = max(1, plot_every_episodes)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.rewards: list[float] = []
        self.lengths: list[int] = []

    def _on_training_start(self) -> None:
        self.metrics_path.write_text("episode,reward,length\n", encoding="utf-8")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if not episode:
                continue
            self.rewards.append(float(episode["r"]))
            self.lengths.append(int(episode["l"]))
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{len(self.rewards)},{float(episode['r'])},{int(episode['l'])}\n"
                )
            if len(self.rewards) % self.plot_every_episodes == 0:
                self._write_plots()
        return True

    def _on_training_end(self) -> None:
        self._write_plots()

    def _write_plots(self) -> None:
        if not self.rewards:
            return
        episodes = np.arange(1, len(self.rewards) + 1)
        rewards = np.array(self.rewards, dtype=np.float32)
        lengths = np.array(self.lengths, dtype=np.float32)
        reward_ma = _moving_average(rewards, window=10)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)
        axes[0].plot(episodes, rewards, label="episode reward", color="#1f77b4", alpha=0.45)
        axes[0].plot(episodes[: len(reward_ma)], reward_ma, label="reward MA(10)", color="#d62728")
        axes[0].set_title("Training Reward")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(episodes, lengths, label="episode length", color="#2ca02c")
        axes[1].set_title("Episode Length")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Steps")
        axes[1].grid(alpha=0.3)

        fig.savefig(self.run_dir / "training_curves.png")
        plt.close(fig)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


def build_callback_list(
    run_dir: Path,
    save_checkpoint_every_steps: int,
    plot_every_episodes: int,
    verbose: int,
) -> CallbackList:
    checkpoint_callback = CheckpointCallback(
        save_freq=save_checkpoint_every_steps,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="dqn_model",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )
    artifacts_callback = TrainingArtifactsCallback(
        run_dir=run_dir,
        plot_every_episodes=plot_every_episodes,
        verbose=verbose,
    )
    return CallbackList([checkpoint_callback, artifacts_callback])

