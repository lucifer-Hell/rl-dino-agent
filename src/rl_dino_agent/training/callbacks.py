from __future__ import annotations

from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


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
        self.best_model_path = self.run_dir / "best_model.zip"
        self.best_score_model_path = self.run_dir / "best_score_model.zip"
        self.rewards: list[float] = []
        self.lengths: list[int] = []
        self.scores: list[int] = []
        self.best_reward = float("-inf")
        self.best_score = -1
        self._metrics_handle = None

    def _on_training_start(self) -> None:
        self._metrics_handle = self.metrics_path.open("w", encoding="utf-8", newline="\n")
        self._metrics_handle.write("episode,reward,length,score\n")
        self._metrics_handle.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if not episode:
                continue
            episode_reward = float(episode["r"])
            episode_length = int(episode["l"])
            episode_score = int(info.get("score", 0))
            self.rewards.append(episode_reward)
            self.lengths.append(episode_length)
            self.scores.append(episode_score)
            if self._metrics_handle is not None:
                self._metrics_handle.write(
                    f"{len(self.rewards)},{episode_reward},{episode_length},{episode_score}\n"
                )
                self._metrics_handle.flush()
            if episode_reward >= self.best_reward:
                self.best_reward = episode_reward
                self.model.save(str(self.best_model_path.with_suffix("")))
            if episode_score >= self.best_score:
                self.best_score = episode_score
                self.model.save(str(self.best_score_model_path.with_suffix("")))
            if len(self.rewards) % self.plot_every_episodes == 0:
                self._write_plots()
        return True

    def _on_training_end(self) -> None:
        self._write_plots()
        if self._metrics_handle is not None:
            self._metrics_handle.close()
            self._metrics_handle = None

    def _write_plots(self) -> None:
        if not self.rewards:
            return
        episodes = np.arange(1, len(self.rewards) + 1)
        rewards = np.array(self.rewards, dtype=np.float32)
        lengths = np.array(self.lengths, dtype=np.float32)
        scores = np.array(self.scores, dtype=np.float32)
        reward_ma = _moving_average(rewards, window=10)
        score_ma = _moving_average(scores, window=10)

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), tight_layout=True)
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

        axes[2].plot(episodes, scores, label="episode score", color="#9467bd", alpha=0.45)
        axes[2].plot(episodes[: len(score_ma)], score_ma, label="score MA(10)", color="#8c564b")
        axes[2].set_title("Episode Score")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Score")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        fig.savefig(self.run_dir / "training_curves.png")
        plt.close(fig)


class RollingCheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: Path,
        name_prefix: str,
        keep_last: int,
        save_replay_buffer: bool,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.save_freq = max(1, save_freq)
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.keep_last = max(1, keep_last)
        self.save_replay_buffer = save_replay_buffer
        self.saved_steps: deque[int] = deque()

    def _on_training_start(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq != 0:
            return True
        step = self.num_timesteps
        model_stem = self.save_path / f"{self.name_prefix}_{step}_steps"
        self.model.save(str(model_stem))
        if self.save_replay_buffer and hasattr(self.model, "save_replay_buffer"):
            self.model.save_replay_buffer(
                str(self.save_path / f"{self.name_prefix}_replay_buffer_{step}_steps.pkl")
            )
        self.saved_steps.append(step)
        while len(self.saved_steps) > self.keep_last:
            stale_step = self.saved_steps.popleft()
            stale_model = self.save_path / f"{self.name_prefix}_{stale_step}_steps.zip"
            if stale_model.exists():
                stale_model.unlink()
            stale_buffer = self.save_path / f"{self.name_prefix}_replay_buffer_{stale_step}_steps.pkl"
            if stale_buffer.exists():
                stale_buffer.unlink()
        return True


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


def build_callback_list(
    run_dir: Path,
    save_checkpoint_every_steps: int,
    keep_last_checkpoints: int,
    save_replay_buffer_checkpoints: bool,
    plot_every_episodes: int,
    verbose: int,
) -> CallbackList:
    checkpoint_callback = RollingCheckpointCallback(
        save_freq=save_checkpoint_every_steps,
        save_path=run_dir / "checkpoints",
        name_prefix="dqn_model",
        keep_last=keep_last_checkpoints,
        save_replay_buffer=save_replay_buffer_checkpoints,
        verbose=verbose,
    )
    artifacts_callback = TrainingArtifactsCallback(
        run_dir=run_dir,
        plot_every_episodes=plot_every_episodes,
        verbose=verbose,
    )
    return CallbackList([checkpoint_callback, artifacts_callback])
