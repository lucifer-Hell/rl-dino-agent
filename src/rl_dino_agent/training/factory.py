from __future__ import annotations

from typing import Any

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from rl_dino_agent.config import AppConfig
from rl_dino_agent.envs import BrowserDinoEnv
from rl_dino_agent.training.extractors import MediumDinoCNN


def build_vector_env(config: AppConfig):
    def _make_env():
        env = BrowserDinoEnv(config)
        return Monitor(env)

    vec_env = DummyVecEnv([_make_env])
    return VecTransposeImage(vec_env)


def build_learning_rate(training) -> float | callable:
    if training.learning_rate_schedule == "constant":
        return training.learning_rate

    start = float(training.learning_rate)
    end = float(training.learning_rate_end)

    def linear_schedule(progress_remaining: float) -> float:
        return end + (start - end) * max(progress_remaining, 0.0)

    return linear_schedule


def build_policy_kwargs(training) -> dict[str, Any]:
    policy_kwargs: dict[str, Any] = {
        "net_arch": list(training.q_net_arch),
    }
    if training.feature_extractor == "medium_dino_cnn":
        policy_kwargs["features_extractor_class"] = MediumDinoCNN
        policy_kwargs["features_extractor_kwargs"] = {
            "features_dim": training.features_dim,
        }
    return policy_kwargs


def build_dqn_model(config: AppConfig, env, tensorboard_log: str) -> DQN:
    training = config.training
    return DQN(
        policy=training.model_policy,
        env=env,
        learning_rate=build_learning_rate(training),
        buffer_size=training.buffer_size,
        learning_starts=training.learning_starts,
        batch_size=training.batch_size,
        gamma=training.gamma,
        train_freq=training.train_freq,
        gradient_steps=training.gradient_steps,
        target_update_interval=training.target_update_interval,
        exploration_fraction=training.exploration_fraction,
        exploration_initial_eps=training.exploration_initial_eps,
        exploration_final_eps=training.exploration_final_eps,
        stats_window_size=training.stats_window_size,
        tensorboard_log=tensorboard_log,
        policy_kwargs=build_policy_kwargs(training),
        device=training.device,
        verbose=training.verbose,
        seed=config.run.seed,
    )
