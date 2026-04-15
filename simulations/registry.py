import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from django.conf import settings


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    kind: str
    default: float | int
    min_value: float | int | None = None
    max_value: float | int | None = None
    help_text: str = ""


@dataclass(frozen=True)
class ExpressionSpec:
    name: str
    label: str
    variables: tuple[str, ...]
    default: str
    help_text: str = ""


@dataclass(frozen=True)
class SimulationSpec:
    slug: str
    title: str
    description: str
    expressions: tuple[ExpressionSpec, ...] = field(default_factory=tuple)
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)
    runner: Callable | None = None


SIMULATION_SPECS: dict[str, SimulationSpec] = {}


def _order_file_path() -> Path:
    return Path(settings.BASE_DIR) / "config" / "simulation_order.json"


def _load_order_index() -> dict[str, int]:
    try:
        loaded = json.loads(_order_file_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}

    if not isinstance(loaded, list):
        return {}

    order_index: dict[str, int] = {}
    for idx, slug in enumerate(loaded):
        if isinstance(slug, str) and slug not in order_index:
            order_index[slug] = idx
    return order_index


def register_spec(spec: SimulationSpec) -> None:
    SIMULATION_SPECS[spec.slug] = spec


def list_specs() -> list[SimulationSpec]:
    order_index = _load_order_index()
    return sorted(
        SIMULATION_SPECS.values(),
        key=lambda item: (order_index.get(item.slug, len(order_index)), item.title),
    )


def get_spec(slug: str) -> SimulationSpec:
    if slug not in SIMULATION_SPECS:
        raise KeyError(f"No existe simulacion registrada para '{slug}'")
    return SIMULATION_SPECS[slug]

