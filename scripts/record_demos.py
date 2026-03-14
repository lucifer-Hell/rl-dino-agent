from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright
from rich.console import Console


RECORDER_BOOTSTRAP = """
() => {
  if (window.DemoRecorder && window.DemoRecorder.installed) {
    return true;
  }

  const recorder = {
    installed: true,
    intervalId: null,
    pendingAction: 0,
    active: false,
    frames: [],
    actions: [],
    rewards: [],
    dones: [],
    scores: [],
    frameWidth: 84,
    frameHeight: 84,
    fps: 12,
    install() {
      this.sourceCanvas = document.querySelector('#game');
      this.bufferCanvas = document.createElement('canvas');
      this.bufferCtx = this.bufferCanvas.getContext('2d', { willReadFrequently: true });
    },
    startEpisode(config) {
      this.stopEpisode();
      this.frameWidth = config.frameWidth;
      this.frameHeight = config.frameHeight;
      this.fps = config.fps;
      this.frames = [];
      this.actions = [];
      this.rewards = [];
      this.dones = [];
      this.scores = [];
      this.pendingAction = 0;
      this.active = true;

      this.bufferCanvas.width = this.frameWidth;
      this.bufferCanvas.height = this.frameHeight;

      const samplePeriod = Math.max(1, Math.round(1000 / this.fps));
      this.intervalId = window.setInterval(() => {
        if (!this.active) {
          return;
        }

        const state = window.RLDino.getState();
        this.bufferCtx.drawImage(this.sourceCanvas, 0, 0, this.frameWidth, this.frameHeight);
        const imageData = this.bufferCtx.getImageData(0, 0, this.frameWidth, this.frameHeight).data;
        const grayscale = new Uint8Array(this.frameWidth * this.frameHeight);

        for (let src = 0, dst = 0; src < imageData.length; src += 4, dst += 1) {
          grayscale[dst] = Math.round(
            imageData[src] * 0.299 +
            imageData[src + 1] * 0.587 +
            imageData[src + 2] * 0.114
          );
        }

        this.frames.push(Array.from(grayscale));
        this.actions.push(this.pendingAction);
        this.rewards.push(Number(state.reward || 0));
        this.dones.push(Boolean(state.gameOver));
        this.scores.push(Number(state.score || 0));
        this.pendingAction = 0;

        if (state.gameOver) {
          this.stopEpisode();
        }
      }, samplePeriod);
    },
    stopEpisode() {
      if (this.intervalId !== null) {
        window.clearInterval(this.intervalId);
        this.intervalId = null;
      }
      this.active = false;
    },
    getStatus() {
      const state = window.RLDino.getState();
      return {
        active: this.active,
        gameOver: Boolean(state.gameOver),
        score: Number(state.score || 0),
        samples: this.actions.length,
      };
    },
    consumeEpisode() {
      this.stopEpisode();
      return {
        frameWidth: this.frameWidth,
        frameHeight: this.frameHeight,
        fps: this.fps,
        frames: this.frames,
        actions: this.actions,
        rewards: this.rewards,
        dones: this.dones,
        scores: this.scores,
      };
    },
  };

  recorder.install();

  document.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if (event.code === "Space" || event.code === "ArrowUp") {
      recorder.pendingAction = 1;
    }
    if (key === "r" && !recorder.active) {
      recorder.pendingAction = 0;
    }
  });

  window.DemoRecorder = recorder;
  return true;
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record human demos for RL Dino.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/index.html")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("demos"))
    parser.add_argument("--frame-width", type=int, default=84)
    parser.add_argument("--frame-height", type=int, default=84)
    return parser.parse_args()


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
        console.print("Browser ready. Play with Space/ArrowUp. The recorder now samples inside the page for smoother play.")

        for episode_idx in range(1, args.episodes + 1):
            console.print(f"Recording episode {episode_idx}/{args.episodes}. Click the browser and play.")
            page.evaluate("() => window.RLDino.reset({ autoplay: false })")
            page.evaluate(
                "(config) => window.DemoRecorder.startEpisode(config)",
                {
                    "fps": args.fps,
                    "frameWidth": args.frame_width,
                    "frameHeight": args.frame_height,
                },
            )

            while True:
                status = page.evaluate("() => window.DemoRecorder.getStatus()")
                if status["gameOver"] and not status["active"]:
                    break
                time.sleep(0.2)

            episode = page.evaluate("() => window.DemoRecorder.consumeEpisode()")
            frames = np.array(episode["frames"], dtype=np.uint8).reshape(
                -1,
                episode["frameHeight"],
                episode["frameWidth"],
            )
            actions = np.array(episode["actions"], dtype=np.int64)
            rewards = np.array(episode["rewards"], dtype=np.float32)
            dones = np.array(episode["dones"], dtype=np.bool_)
            scores = np.array(episode["scores"], dtype=np.float32)

            episode_dir = args.output_dir / f"episode_{episode_idx:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                episode_dir / "trajectory.npz",
                frames=frames,
                actions=actions,
                rewards=rewards,
                dones=dones,
                scores=scores,
            )
            (episode_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "fps": int(episode["fps"]),
                        "frame_width": int(episode["frameWidth"]),
                        "frame_height": int(episode["frameHeight"]),
                        "num_steps": int(len(actions)),
                        "final_score": float(scores[-1]) if len(scores) else 0.0,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            console.print(f"Saved {episode_dir}")

        browser.close()


if __name__ == "__main__":
    main()
