"""Top-level package exports for qml_benchmarks."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from qml_benchmarks.hyperparameter_settings import hyper_parameter_settings


def __getattr__(name: str) -> Any:
	"""Lazily import optional heavy subpackages."""
	if name in {"data", "models"}:
		return import_module(f"qml_benchmarks.{name}")
	raise AttributeError(f"module 'qml_benchmarks' has no attribute '{name}'")

__all__ = ["data", "models", "hyper_parameter_settings"]
