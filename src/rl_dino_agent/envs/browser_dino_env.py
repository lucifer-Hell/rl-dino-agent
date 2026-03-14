from __future__ import annotations

from collections import deque
from pathlib import Path
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
        resized = cv2.resize(
            image,
            (self.config.game.env.frame_width, self.config.game.env.frame_height),
            interpolation=cv2.INTER_AREA,
        )
        if self.config.game.env.grayscale:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            return gray
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

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
        self.frame_buffer.clear()
        observation = self._reset_game()
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

        reward = float(result["reward"])
        terminated = bool(result["done"])
        truncated = self.current_step >= self.config.game.env.max_episode_steps
        info = {
            "score": int(result["score"]),
            "game_observation": result["observation"],
        }
        if terminated or truncated:
            info["episode"] = {"r": float(result["observation"]["totalReward"]), "l": self.current_step}

        return self._stacked_observation(), reward, terminated, truncated, info

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

