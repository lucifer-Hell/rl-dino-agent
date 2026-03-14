from __future__ import annotations

from collections import deque
from typing import Any
import base64
import time

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from playwright.sync_api import Browser, Page, Playwright, sync_playwright

from rl_dino_agent.config import AppConfig
from rl_dino_agent.utils.server import LocalStaticServer


class BrowserDinoEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.server: LocalStaticServer | None = None
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.page: Page | None = None
        self.current_step = 0
        self.episode_index = 0
        self.episode_reward = 0.0
        self.previous_game_observation: dict[str, Any] | None = None
        self.frame_buffer: deque[np.ndarray] = deque(
            maxlen=self.config.game.env.frame_stack
        )

        channels = self.config.game.env.frame_stack if self.config.game.env.grayscale else 3 * self.config.game.env.frame_stack
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.config.game.env.frame_height,
                self.config.game.env.frame_width,
                channels,
            ),
            dtype=np.uint8,
        )

        self._launch()

    def _launch(self) -> None:
        if self.config.game.serve.auto_start:
            self.server = LocalStaticServer(
                root=self.config.game.repo_path,
                host=self.config.game.serve.host,
                port=self.config.game.serve.port,
            )
            self.server.start()

        self.playwright = sync_playwright().start()
        browser_type = self.playwright.chromium
        self.browser = browser_type.launch(
            headless=self.config.game.browser.headless,
            slow_mo=self.config.game.browser.slow_mo_ms,
        )
        context = self.browser.new_context(
            viewport={
                "width": self.config.game.browser.viewport_width,
                "height": self.config.game.browser.viewport_height,
            }
        )
        self.page = context.new_page()
        url = (
            f"http://{self.config.game.serve.host}:{self.config.game.serve.port}/"
            f"{self.config.game.index_file}"
        )
        self.page.goto(url, wait_until="domcontentloaded")
        self.page.wait_for_selector(self.config.game.env.canvas_selector)
        self.page.wait_for_function("() => Boolean(window.RLDino)")
        time.sleep(0.5)

    def _page(self) -> Page:
        if self.page is None:
            raise RuntimeError("Browser page is not available.")
        return self.page

    def _capture_frame(self) -> np.ndarray:
        script = f"""
() => {{
  const canvas = document.querySelector('{self.config.game.env.canvas_selector}');
  return canvas.toDataURL('image/png').split(',')[1];
}}
"""
        encoded = self._page().evaluate(script)
        image_bytes = base64.b64decode(encoded)
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = self._crop_frame(image)
        resized = cv2.resize(
            image,
            (self.config.game.env.frame_width, self.config.game.env.frame_height),
            interpolation=cv2.INTER_AREA,
        )
        if self.config.game.env.grayscale:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            return gray
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def _crop_frame(self, image: np.ndarray) -> np.ndarray:
        env_cfg = self.config.game.env
        top = max(0, env_cfg.crop_top)
        bottom = image.shape[0] - max(0, env_cfg.crop_bottom)
        left = max(0, env_cfg.crop_left)
        right = image.shape[1] - max(0, env_cfg.crop_right)
        if top >= bottom or left >= right:
            return image
        return image[top:bottom, left:right]

    def _stacked_observation(self) -> np.ndarray:
        if len(self.frame_buffer) != self.frame_buffer.maxlen:
            raise RuntimeError("Frame buffer is not fully initialized.")
        return np.dstack(list(self.frame_buffer)).astype(np.uint8)

    def _reset_game(self) -> dict[str, Any]:
        return self._page().evaluate("() => window.RLDino.reset({ autoplay: true })")

    def _step_game(self, action: int) -> dict[str, Any]:
        step_ms = self.config.game.env.step_ms
        return self._page().evaluate(
            "(payload) => window.RLDino.step(payload.action, payload.deltaMs)",
            {"action": int(action), "deltaMs": step_ms},
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_index += 1
        self.episode_reward = 0.0
        self.previous_game_observation = None
        self.frame_buffer.clear()
        observation = self._reset_game()
        self.previous_game_observation = observation
        frame = self._capture_frame()
        for _ in range(self.config.game.env.frame_stack):
            self.frame_buffer.append(frame.copy())
        info = {
            "episode_index": self.episode_index,
            "game_observation": observation,
        }
        return self._stacked_observation(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        result = self._step_game(action)
        frame = self._capture_frame()
        self.frame_buffer.append(frame)
        self.current_step += 1

        raw_reward = float(result["reward"])
        terminated = bool(result["done"])
        truncated = self.current_step >= self.config.game.env.max_episode_steps
        reward = self._transform_reward(
            raw_reward=raw_reward,
            action=int(action),
            terminated=terminated,
            observation=result["observation"],
        )
        self.episode_reward += reward
        info = {
            "score": int(result["score"]),
            "game_observation": result["observation"],
            "raw_reward": raw_reward,
        }
        if terminated or truncated:
            info["episode"] = {"r": float(self.episode_reward), "l": self.current_step}
        self.previous_game_observation = result["observation"]

        return self._stacked_observation(), reward, terminated, truncated, info

    def _transform_reward(
        self,
        raw_reward: float,
        action: int,
        terminated: bool,
        observation: dict[str, Any],
    ) -> float:
        env_cfg = self.config.game.env
        reward = raw_reward
        if env_cfg.reward_mode == "clip_survival":
            reward = -1.0 if terminated else min(raw_reward, 0.1)
        elif env_cfg.reward_mode == "clip_all":
            reward = float(np.clip(raw_reward, -1.0, 1.0))
        elif env_cfg.reward_mode == "survival_bonus":
            reward = -1.0 if terminated else 0.05

        if action == 1 and not terminated:
            reward -= env_cfg.jump_penalty

        reward += self._obstacle_clear_bonus(observation)
        reward -= self._unsafe_descent_penalty(observation, terminated)

        return reward * env_cfg.reward_scale

    def _obstacle_clear_bonus(self, observation: dict[str, Any]) -> float:
        env_cfg = self.config.game.env
        previous = self.previous_game_observation
        if previous is None:
            return 0.0

        prev_obstacle = previous.get("nextObstacle")
        curr_obstacle = observation.get("nextObstacle")
        if not prev_obstacle:
            return 0.0

        prev_distance = float(prev_obstacle.get("distance", 1e9))
        curr_distance = float(curr_obstacle.get("distance", 1e9)) if curr_obstacle else 1e9

        if prev_distance <= env_cfg.unsafe_descent_distance and curr_distance > prev_distance + 40:
            return env_cfg.obstacle_clear_bonus
        return 0.0

    def _unsafe_descent_penalty(self, observation: dict[str, Any], terminated: bool) -> float:
        if terminated:
            return 0.0

        env_cfg = self.config.game.env
        dino = observation.get("dino") or {}
        obstacle = observation.get("nextObstacle")
        if not obstacle:
            return 0.0

        descending = float(dino.get("velocityY", 0.0)) > 150.0
        on_ground = bool(dino.get("onGround", False))
        distance = float(obstacle.get("distance", 1e9))

        if descending and not on_ground and 0.0 <= distance <= env_cfg.unsafe_descent_distance:
            return env_cfg.unsafe_descent_penalty
        return 0.0

    def render(self) -> None:
        return None

    def close(self) -> None:
        if self.page is not None:
            self.page.context.close()
        if self.browser is not None:
            self.browser.close()
        if self.playwright is not None:
            self.playwright.stop()
        if self.server is not None:
            self.server.stop()
