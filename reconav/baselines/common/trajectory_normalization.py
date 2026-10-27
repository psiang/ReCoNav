"""Lossless evaluator-route normalization for external baselines."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


# This tolerance is only for floating-point equality/no-motion noise.  It is
# deliberately many orders of magnitude below any simulator action distance.
POSITION_EQUALITY_EPSILON_M = 1e-9


def _xyz(point: Mapping[str, Any]) -> tuple[float, float, float]:
    return float(point["x"]), float(point["y"]), float(point["z"])


def normalize_consecutive_duplicate_positions(
    waypoints: Sequence[Mapping[str, Any]],
    *,
    epsilon_m: float = POSITION_EQUALITY_EPSILON_M,
) -> list[dict[str, Any]]:
    """Remove only consecutive positions equal within ``epsilon_m``.

    No interpolation, spatial simplification, or movement-based resampling is
    performed.  The complete input sequence remains the caller's raw trace.
    """

    if epsilon_m < 0:
        raise ValueError("epsilon_m must be non-negative")
    normalized: list[dict[str, Any]] = []
    previous_xyz: tuple[float, float, float] | None = None
    for point in waypoints:
        current_xyz = _xyz(point)
        if previous_xyz is None or math.dist(previous_xyz, current_xyz) > epsilon_m:
            normalized.append(dict(point))
            previous_xyz = current_xyz
    return normalized


def normalize_evaluator_route(
    route: Mapping[str, Any],
    *,
    prediction_empty: bool = False,
) -> dict[str, Any]:
    """Return a route copy whose evaluator-visible waypoints are normalized."""

    normalized = dict(route)
    raw_waypoints = route.get("waypoints", [])
    if not isinstance(raw_waypoints, list):
        raise TypeError("route.waypoints must be a list")
    normalized["waypoints"] = (
        []
        if prediction_empty or not raw_waypoints
        else normalize_consecutive_duplicate_positions(raw_waypoints)
    )
    return normalized
