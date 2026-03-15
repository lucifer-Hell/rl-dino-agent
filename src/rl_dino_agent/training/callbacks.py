from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback


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
        self.best_reward_model_path = self.run_dir / "best_reward_model.zip"
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
                self.model.save(str(self.best_reward_model_path.with_suffix("")))
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


class StopOnTrainingPlateauCallback(BaseCallback):
    def __init__(
        self,
        patience_episodes: int,
        min_episodes: int,
        min_timesteps: int,
        metric: str,
        window_episodes: int,
        min_delta: float,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.patience_episodes = max(0, patience_episodes)
        self.min_episodes = max(0, min_episodes)
        self.min_timesteps = max(0, min_timesteps)
        self.metric = metric
        self.window_episodes = max(1, window_episodes)
        self.min_delta = float(min_delta)
        self.metric_history: list[float] = []
        self.best_window_value = float("-inf")
        self.best_episode_index = 0

    def _extract_metric(self, info: dict) -> float | None:
        episode = info.get("episode")
        if not episode:
            return None
        if self.metric == "reward":
            return float(episode["r"])
        return float(info.get("score", 0))

    def _on_step(self) -> bool:
        if self.patience_episodes <= 0:
            return True

        infos = self.locals.get("infos", [])
        should_continue = True
        for info in infos:
            metric_value = self._extract_metric(info)
            if metric_value is None:
                continue

            self.metric_history.append(metric_value)
            episode_idx = len(self.metric_history)
            window_values = self.metric_history[-self.window_episodes :]
            window_mean = float(np.mean(window_values))

            if window_mean > self.best_window_value + self.min_delta:
                self.best_window_value = window_mean
                self.best_episode_index = episode_idx

            enough_episodes = episode_idx >= self.min_episodes
            enough_timesteps = self.num_timesteps >= self.min_timesteps
            stale_episodes = episode_idx - self.best_episode_index
            if enough_episodes and enough_timesteps and stale_episodes >= self.patience_episodes:
                if self.verbose > 0:
                    print(
                        "Early stopping: training plateau detected "
                        f"(metric={self.metric}, best_window={self.best_window_value:.3f}, "
                        f"episodes_since_improvement={stale_episodes})."
                    )
                should_continue = False
                break
        return should_continue


class PeriodicDemoCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        config_path: Path,
        every_steps: int,
        episodes: int,
        deterministic: bool,
        headless: bool,
        sleep_after_episode: float,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.run_dir = run_dir
        self.config_path = config_path
        self.every_steps = max(0, every_steps)
        self.episodes = max(1, episodes)
        self.deterministic = deterministic
        self.headless = headless
        self.sleep_after_episode = max(0.0, sleep_after_episode)
        self.demo_dir = self.run_dir / "demo"
        self.demo_model_stem = self.demo_dir / "latest_demo_model"

    def _on_training_start(self) -> None:
        if self.every_steps > 0:
            self.demo_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.every_steps <= 0 or self.num_timesteps % self.every_steps != 0:
            return True

        self.model.save(str(self.demo_model_stem))
        command = [
            str(Path(sys.executable)),
            str(Path.cwd() / "scripts" / "play_model.py"),
            "--config",
            str(self.config_path),
            "--model",
            str(self.demo_model_stem.with_suffix(".zip")),
            "--episodes",
            str(self.episodes),
            "--sleep-after-episode",
            str(self.sleep_after_episode),
        ]
        if self.deterministic:
            command.append("--deterministic")
        if self.headless:
            command.append("--headless")
        else:
            command.append("--visible")

        if self.verbose > 0:
            print(
                f"Launching demo playback at step={self.num_timesteps} "
                f"using {self.demo_model_stem.with_suffix('.zip')}"
            )
        subprocess.run(command, check=True, cwd=Path.cwd())
        return True


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


def build_callback_list(
    run_dir: Path,
    config_path: Path,
    save_checkpoint_every_steps: int,
    keep_last_checkpoints: int,
    save_replay_buffer_checkpoints: bool,
    plot_every_episodes: int,
    verbose: int,
    eval_env=None,
    eval_freq_steps: int = 0,
    eval_episodes: int = 5,
    eval_deterministic: bool = True,
    early_stop_patience_episodes: int = 0,
    early_stop_min_episodes: int = 0,
    early_stop_min_timesteps: int = 0,
    early_stop_window_episodes: int = 10,
    early_stop_metric: str = "score",
    early_stop_min_delta: float = 0.0,
    demo_every_steps: int = 0,
    demo_episodes: int = 1,
    demo_deterministic: bool = True,
    demo_headless: bool = False,
    demo_sleep_after_episode: float = 0.5,
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
    callbacks: list[BaseCallback] = [checkpoint_callback, artifacts_callback]
    if demo_every_steps > 0:
        callbacks.append(
            PeriodicDemoCallback(
                run_dir=run_dir,
                config_path=config_path,
                every_steps=demo_every_steps,
                episodes=demo_episodes,
                deterministic=demo_deterministic,
                headless=demo_headless,
                sleep_after_episode=demo_sleep_after_episode,
                verbose=verbose,
            )
        )
    if early_stop_patience_episodes > 0:
        callbacks.append(
            StopOnTrainingPlateauCallback(
                patience_episodes=early_stop_patience_episodes,
                min_episodes=early_stop_min_episodes,
                min_timesteps=early_stop_min_timesteps,
                metric=early_stop_metric,
                window_episodes=early_stop_window_episodes,
                min_delta=early_stop_min_delta,
                verbose=verbose,
            )
        )
    if eval_env is not None and eval_freq_steps > 0:
        eval_dir = run_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            EvalCallback(
                eval_env=eval_env,
                best_model_save_path=str(eval_dir),
                log_path=str(eval_dir),
                eval_freq=max(1, eval_freq_steps),
                n_eval_episodes=max(1, eval_episodes),
                deterministic=eval_deterministic,
                render=False,
                verbose=verbose,
            )
        )
    return CallbackList(callbacks)
