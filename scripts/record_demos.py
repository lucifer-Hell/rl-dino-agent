from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright
from rich.console import Console


RECORDER_BOOTSTRAP = """
() => {
  window.DemoRecorder = {
    actionEvents: [],
    startedAt: performance.now(),
  };

  document.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if (event.code === "Space" || event.code === "ArrowUp") {
      window.DemoRecorder.actionEvents.push({
        kind: "jump",
        ts: performance.now()
      });
    }
    if (key === "r") {
      window.DemoRecorder.actionEvents.push({
        kind: "reset",
        ts: performance.now()
      });
    }
  });
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record human demos for RL Dino.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/index.html")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("demos"))
    parser.add_argument("--frame-width", type=int, default=84)
    parser.add_argument("--frame-height", type=int, default=84)
    return parser.parse_args()


def capture_frame(page, frame_width: int, frame_height: int) -> np.ndarray:
    encoded = page.evaluate(
        """
() => {
  const canvas = document.querySelector('#game');
  return canvas.toDataURL('image/png').split(',')[1];
}
"""
    )
    image = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    return cv2.resize(image, (frame_width, frame_height), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    console = Console()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 960, "height": 540})
        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_function("() => Boolean(window.RLDino)")
        page.evaluate(RECORDER_BOOTSTRAP)
        console.print("Browser ready. Play with Space/ArrowUp, reset with R.")

        for episode_idx in range(1, args.episodes + 1):
            console.print(f"Recording episode {episode_idx}/{args.episodes}. Click the browser and play.")
            page.evaluate("() => window.RLDino.reset({ autoplay: false })")
            time.sleep(0.5)

            frames: list[np.ndarray] = []
            actions: list[int] = []
            rewards: list[float] = []
            dones: list[bool] = []
            scores: list[float] = []

            last_event_count = 0
            last_done = False
            sample_period = 1.0 / args.fps

            while True:
                state = page.evaluate("() => window.RLDino.getState()")
                events = page.evaluate("() => window.DemoRecorder.actionEvents.slice()")
                jump_count = sum(1 for event in events[last_event_count:] if event["kind"] == "jump")
                last_event_count = len(events)

                frames.append(capture_frame(page, args.frame_width, args.frame_height))
                actions.append(1 if jump_count > 0 else 0)
                rewards.append(float(state["reward"]))
                dones.append(bool(state["gameOver"]))
                scores.append(float(state["score"]))

                if state["gameOver"] and not last_done:
                    break
                last_done = bool(state["gameOver"])
                time.sleep(sample_period)

            episode_dir = args.output_dir / f"episode_{episode_idx:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                episode_dir / "trajectory.npz",
                frames=np.stack(frames).astype(np.uint8),
                actions=np.array(actions, dtype=np.int64),
                rewards=np.array(rewards, dtype=np.float32),
                dones=np.array(dones, dtype=np.bool_),
                scores=np.array(scores, dtype=np.float32),
            )
            (episode_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "fps": args.fps,
                        "frame_width": args.frame_width,
                        "frame_height": args.frame_height,
                        "num_steps": len(actions),
                        "final_score": scores[-1] if scores else 0,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            console.print(f"Saved {episode_dir}")

        browser.close()


if __name__ == "__main__":
    main()
