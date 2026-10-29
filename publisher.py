from pathlib import Path
from typing import Protocol


class Publisher(Protocol):
    def publish(self, manifest_path: Path) -> None: ...
