from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RunConfig:
    name: str
    seed: int
    output_root: Path
    save_replay_frames: bool


@dataclass(slots=True)
class ServeConfig:
    host: str
    port: int
    auto_start: bool


@dataclass(slots=True)
class BrowserConfig:
    headless: bool
    slow_mo_ms: int
    viewport_width: int
    viewport_height: int


@dataclass(slots=True)
class EnvConfig:
    canvas_selector: str
    frame_width: int
    frame_height: int
    frame_stack: int
    grayscale: bool
    crop_top: int
    crop_bottom: int
    crop_left: int
    crop_right: int
    step_ms: float
    max_episode_steps: int
    reward_mode: str
    reward_scale: float
    jump_penalty: float
    obstacle_clear_bonus: float
    unsafe_descent_penalty: float
    unsafe_descent_distance: float


@dataclass(slots=True)
class GameConfig:
    repo_path: Path
    index_file: str
    serve: ServeConfig
    browser: BrowserConfig
    env: EnvConfig


@dataclass(slots=True)
class TrainingConfig:
    total_timesteps: int
    learning_rate: float
    learning_rate_schedule: str
    learning_rate_end: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    gamma: float
    train_freq: int
    gradient_steps: int
    target_update_interval: int
    exploration_fraction: float
    exploration_initial_eps: float
    exploration_final_eps: float
    stats_window_size: int
    tensorboard_log_subdir: str
    save_checkpoint_every_steps: int
    keep_last_checkpoints: int
    save_replay_buffer_checkpoints: bool
    plot_every_episodes: int
    model_policy: str
    device: str
    verbose: int


@dataclass(slots=True)
class EvaluationConfig:
    deterministic: bool
    episodes: int
    eval_freq_steps: int
    headless: bool


@dataclass(slots=True)
class AppConfig:
    run: RunConfig
    game: GameConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return cls(
            run=RunConfig(
                name=raw["run"]["name"],
                seed=int(raw["run"]["seed"]),
                output_root=Path(raw["run"]["output_root"]),
                save_replay_frames=bool(raw["run"]["save_replay_frames"]),
            ),
            game=GameConfig(
                repo_path=Path(raw["game"]["repo_path"]),
                index_file=raw["game"]["index_file"],
                serve=ServeConfig(**raw["game"]["serve"]),
                browser=BrowserConfig(**raw["game"]["browser"]),
                env=EnvConfig(**raw["game"]["env"]),
            ),
            training=TrainingConfig(**raw["training"]),
            evaluation=EvaluationConfig(**raw["evaluation"]),
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run"]["output_root"] = str(self.run.output_root)
        payload["game"]["repo_path"] = str(self.game.repo_path)
        return payload
