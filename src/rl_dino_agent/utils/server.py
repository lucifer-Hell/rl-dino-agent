from __future__ import annotations

from pathlib import Path
import socket
import subprocess
import sys
import time


class LocalStaticServer:
    def __init__(self, root: Path, host: str, port: int) -> None:
        self.root = Path(root).resolve()
        self.host = host
        self.port = port
        self.process: subprocess.Popen[str] | None = None
        self.python_executable = Path(sys.executable)

    def start(self) -> None:
        if self.process is not None:
            return
        self.process = subprocess.Popen(
            [
                str(self.python_executable),
                "-m",
                "http.server",
                str(self.port),
                "--bind",
                self.host,
            ],
            cwd=self.root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._wait_until_ready()

    def _wait_until_ready(self, timeout_seconds: float = 10.0) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.25)
                if probe.connect_ex((self.host, self.port)) == 0:
                    return
            time.sleep(0.1)
        raise RuntimeError(
            f"Timed out waiting for local server at http://{self.host}:{self.port}"
        )

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        self.process.wait(timeout=5)
        self.process = None
