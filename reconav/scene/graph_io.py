"""Lossless, non-executable JSON representation of the paper navigation geometry."""
from __future__ import annotations

import dataclasses
import gzip
import json
import math
from pathlib import Path

import numpy as np
from shapely import wkb
from shapely.geometry.base import BaseGeometry


def encode(value):
    if dataclasses.is_dataclass(value):
        return {"type": "dataclass", "class": type(value).__name__, "fields": {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value, BaseGeometry):
        return {"type": "geometry", "wkb": value.wkb_hex}
    if isinstance(value, dict):
        return {"type": "dict", "items": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [encode(x) for x in value]}
    if isinstance(value, (set, frozenset)):
        return {"type": type(value).__name__, "items": [encode(x) for x in sorted(value)]}
    if isinstance(value, list):
        return [encode(x) for x in value]
    if isinstance(value, np.ndarray):
        return {"type": "array", "dtype": str(value.dtype), "items": value.tolist()}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"type": "float", "value": repr(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported geometry value: {type(value).__name__}")


def decode(value):
    if isinstance(value, list):
        return [decode(x) for x in value]
    if not isinstance(value, dict):
        return value
    kind = value["type"]
    if kind == "dict":
        return {decode(k): decode(v) for k, v in value["items"]}
    if kind in {"tuple", "set", "frozenset"}:
        return {"tuple": tuple, "set": set, "frozenset": frozenset}[kind](decode(x) for x in value["items"])
    if kind == "geometry":
        return wkb.loads(value["wkb"], hex=True)
    if kind == "array":
        return np.asarray(value["items"], dtype=value["dtype"])
    if kind == "float":
        return float(value["value"])
    if kind == "dataclass":
        from reconav.scene import navigation as geometry_graph, mesh_physical
        classes = {name: cls for module in (geometry_graph, mesh_physical) for name, cls in vars(module).items() if isinstance(cls, type) and dataclasses.is_dataclass(cls)}
        cls = classes.get(value["class"])
        if cls is None:
            raise ValueError(f"Unknown navigation geometry class: {value['class']}")
        return cls(**{k: decode(v) for k, v in value["fields"].items()})
    raise ValueError(f"Unknown navigation geometry encoding: {kind}")


def asg_identity(asg):
    import hashlib
    value = [asg.to_entity_dict(), asg.to_geometry_dict(), asg.to_topology_dict()]
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path: Path, graph, asg) -> None:
    from reconav.scene.navigation import GEOMETRY_ALGORITHM_VERSION
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "reconav-navigation-v1", "algorithm_version": GEOMETRY_ALGORITHM_VERSION, "asg_identity": asg_identity(asg), "graph": encode(graph)}
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=False)


_CACHE = {}


def load_if_available(asg, config):
    from reconav.scene.navigation import GEOMETRY_ALGORITHM_VERSION
    from reconav.paths import navigation_root
    path = navigation_root() / asg.data_source / asg.scene_id / "navigation/graph.json.gz"
    if not path.is_file():
        return None
    signature = asg_identity(asg)
    key = (str(path.resolve()), path.stat().st_mtime_ns, signature, repr(config))
    if key in _CACHE:
        return _CACHE[key]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        doc = json.load(stream)
    if doc["format"] != "reconav-navigation-v1" or doc["algorithm_version"] != GEOMETRY_ALGORITHM_VERSION or doc["asg_identity"] != signature:
        raise ValueError(f"Navigation geometry does not match this ASG/code version: {path}")
    graph = decode(doc["graph"])
    if graph.config != config or graph.scene_id != asg.scene_id or graph.data_source != asg.data_source:
        raise ValueError(f"Navigation geometry has a different physical configuration: {path}")
    graph.cache_dir = str(path.parent)
    graph.cache_hit = True
    setattr(asg, "_geometry_graph_handle", graph)
    _CACHE[key] = graph
    return graph
