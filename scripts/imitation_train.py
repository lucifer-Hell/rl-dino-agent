from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from stable_baselines3 import DQN
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from rl_dino_agent.config import AppConfig
from rl_dino_agent.training.factory import build_vector_env
from rl_dino_agent.utils.plotting import initialize_run_dir, persist_run_metadata


class DemoFrameStackDataset(Dataset):
    def __init__(self, demos_dir: Path, frame_stack: int) -> None:
        self.samples: list[tuple[np.ndarray, int]] = []
        for episode_dir in sorted(demos_dir.glob("episode_*")):
            trajectory = np.load(episode_dir / "trajectory.npz")
            frames = trajectory["frames"].astype(np.uint8)
            actions = trajectory["actions"].astype(np.int64)
            if len(frames) != len(actions):
                continue
            for idx in range(len(actions)):
                start = max(0, idx - frame_stack + 1)
                stack = frames[start : idx + 1]
                if len(stack) < frame_stack:
                    pad = np.repeat(stack[:1], frame_stack - len(stack), axis=0)
                    stack = np.concatenate([pad, stack], axis=0)
                stack = np.transpose(stack, (0, 1, 2))
                self.samples.append((stack, int(actions[idx])))

        if not self.samples:
            raise RuntimeError(f"No demo samples found in {demos_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames, action = self.samples[index]
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        return frames_tensor, torch.tensor(action, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavior clone demo episodes into the DQN policy.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--demos", type=Path, default=Path("demos"))
    parser.add_argument("--model", type=Path, required=True, help="Base DQN model zip to fine-tune from demos.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-split", type=float, default=0.9)
    return parser.parse_args()


def evaluate(model: DQN, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.policy.set_training_mode(False)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        for observations, actions in loader:
            observations = observations.to(device)
            actions = actions.to(device)
            logits = model.q_net(observations)
            loss = F.cross_entropy(logits, actions)
            total_loss += loss.item() * actions.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == actions).sum().item()
            total_items += actions.size(0)
    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


def print_summary(console: Console, model_path: Path, demos_dir: Path, run_dir: Path, dataset_size: int) -> None:
    table = Table(title="Imitation Learning")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Base model", str(model_path))
    table.add_row("Demos", str(demos_dir))
    table.add_row("Samples", str(dataset_size))
    table.add_row("Run dir", str(run_dir))
    console.print(table)


def main() -> None:
    args = parse_args()
    console = Console()
    config = AppConfig.load(args.config)
    config.run.name = f"{config.run.name}-imitation"
    run_dir = initialize_run_dir(config)
    persist_run_metadata(config, args.config.resolve(), run_dir)

    dataset = DemoFrameStackDataset(args.demos.resolve(), config.game.env.frame_stack)
    print_summary(console, args.model.resolve(), args.demos.resolve(), run_dir, len(dataset))

    train_size = int(len(dataset) * args.train_split)
    val_size = max(1, len(dataset) - train_size)
    if train_size == 0:
        train_size = len(dataset) - 1
        val_size = 1
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.run.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    env = build_vector_env(config)
    model = DQN.load(
        str(args.model.resolve()),
        env=env,
        device=config.training.device,
        tensorboard_log=str(run_dir / config.training.tensorboard_log_subdir),
    )
    device = model.device
    optimizer = torch.optim.Adam(model.q_net.parameters(), lr=args.learning_rate)

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_model_path = run_dir / "best_imitation_model"

    try:
        for epoch in range(1, args.epochs + 1):
            model.policy.set_training_mode(True)
            epoch_loss = 0.0
            correct = 0
            count = 0

            for observations, actions in train_loader:
                observations = observations.to(device)
                actions = actions.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model.q_net(observations)
                loss = F.cross_entropy(logits, actions)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * actions.size(0)
                correct += (logits.argmax(dim=1) == actions).sum().item()
                count += actions.size(0)

            train_loss = epoch_loss / max(count, 1)
            train_acc = correct / max(count, 1)
            val_loss, val_acc = evaluate(model, val_loader, device)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            history.append(row)
            console.print(
                f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

            model.q_net_target.load_state_dict(model.q_net.state_dict())

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                model.save(str(best_model_path))

        model.save(str(run_dir / "final_imitation_model"))
        (run_dir / "imitation_metrics.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        console.print(f"[bold green]Imitation training complete.[/bold green] Best val acc: {best_val_acc:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
