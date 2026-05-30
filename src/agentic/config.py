from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    def __init__(self, config_path: str | Path | None = None):
        self._data: dict[str, Any] = {}
        if config_path:
            self.load(config_path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            self._data = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        data = self._data
        for k in keys:
            if isinstance(data, dict):
                data = data.get(k)
                if data is None:
                    return default
            else:
                return default
        return data

    def set(self, key: str, value: Any) -> None:
        keys = key.split(".")
        data = self._data
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        data[keys[-1]] = value

    def to_dict(self) -> dict:
        return self._data.copy()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False)

    @classmethod
    def from_env(cls) -> Config:
        config = cls()
        if os.getenv("OPENAI_API_KEY"):
            config.set("llm.api_key", os.getenv("OPENAI_API_KEY"))
        if os.getenv("OPENAI_MODEL"):
            config.set("llm.model", os.getenv("OPENAI_MODEL"))
        if os.getenv("AGENT_MAX_ITERATIONS"):
            config.set("agent.max_iterations", int(os.getenv("AGENT_MAX_ITERATIONS")))
        if os.getenv("LOG_LEVEL"):
            config.set("logging.level", os.getenv("LOG_LEVEL"))
        return config
