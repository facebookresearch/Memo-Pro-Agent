# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Configuration loading.

A single YAML file describes a run. Values are plain nested dicts until they are
handed to the component that owns them, so adding a knob means touching the
component's dataclass and the YAML, not a central schema.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class ConfigError(RuntimeError):
    pass


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a config file and apply ``key.path=value`` overrides."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for override in overrides or []:
        apply_override(config, override)
    return config


def apply_override(config: dict[str, Any], override: str) -> None:
    """Apply one ``a.b.c=value`` override in place.

    The key must already exist. Creating keys on the fly turns a typo into a
    silently ignored setting, which is the worst way to discover a config bug.
    """
    if "=" not in override:
        raise ConfigError(f"override must be key.path=value, got {override!r}")
    dotted, raw = override.split("=", 1)
    parts = dotted.strip().split(".")
    node: Any = config
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"unknown config key: {dotted}")
        node = node[part]
    leaf = parts[-1]
    if not isinstance(node, dict) or leaf not in node:
        raise ConfigError(f"unknown config key: {dotted}")
    node[leaf] = _coerce(raw)


def _coerce(raw: str) -> Any:
    text = raw.strip()
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"config section {name!r} is missing or not a mapping")
    return copy.deepcopy(value)
