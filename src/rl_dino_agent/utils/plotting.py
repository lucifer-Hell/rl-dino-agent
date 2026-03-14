from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

import yaml

from rl_dino_agent.config import AppConfig


def initialize_run_dir(config: AppConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path.cwd() / config.run.output_root / f"{timestamp}-{config.run.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def persist_run_metadata(config: AppConfig, config_path: Path, run_dir: Path) -> None:
    resolved = config.as_dict()
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False),
        encoding="utf-8",
    )
    shutil.copy2(config_path, run_dir / config_path.name)
