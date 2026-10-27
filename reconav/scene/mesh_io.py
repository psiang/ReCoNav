"""Read scene mesh triangles and vertex colors from GLB files."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

_COMPONENT_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}

_TYPE_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

def _glb(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"not a GLB file: {path}")
    offset = 12
    model = None
    binary = None
    while offset + 8 <= len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + length]
        offset += length
        if kind == 0x4E4F534A:
            model = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
    if model is None or binary is None:
        raise ValueError(f"GLB is missing JSON or BIN chunk: {path}")
    return model, binary

def _accessor(model: dict[str, Any], binary: bytes, index: int) -> np.ndarray:
    accessor = model["accessors"][index]
    view = model["bufferViews"][accessor["bufferView"]]
    dtype = _COMPONENT_DTYPES[accessor["componentType"]]
    width = _TYPE_WIDTH[accessor["type"]]
    count = int(accessor["count"])
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", dtype.itemsize * width))
    if stride == dtype.itemsize * width:
        return np.frombuffer(binary, dtype=dtype, count=count * width, offset=start).reshape(count, width)
    return np.ndarray(
        shape=(count, width),
        dtype=dtype,
        buffer=binary,
        offset=start,
        strides=(stride, dtype.itemsize),
    ).copy()

def load_glb_triangles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return world-space triangles and their RGB colors from the viewer GLB."""
    model, binary = _glb(path)
    triangles: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()
    for node in model.get("nodes", []):
        mesh_index = node.get("mesh")
        if mesh_index is None:
            continue
        for primitive_index, primitive in enumerate(model["meshes"][mesh_index].get("primitives", [])):
            if int(primitive.get("mode", 4)) != 4 or (mesh_index, primitive_index) in seen:
                continue
            seen.add((mesh_index, primitive_index))
            attributes = primitive.get("attributes", {})
            position_index = attributes.get("POSITION")
            if position_index is None:
                continue
            positions = _accessor(model, binary, position_index).astype(np.float32)
            if "indices" in primitive:
                indices = _accessor(model, binary, primitive["indices"]).reshape(-1).astype(np.int64)
            else:
                indices = np.arange(len(positions), dtype=np.int64)
            usable = len(indices) - (len(indices) % 3)
            indexed = indices[:usable].reshape(-1, 3)
            triangles.append(positions[indexed])

            color_index = attributes.get("COLOR_0")
            if color_index is None:
                vertex_colors = np.full((len(positions), 3), 155, dtype=np.uint8)
            else:
                vertex_colors = _accessor(model, binary, color_index)[:, :3]
                if np.issubdtype(vertex_colors.dtype, np.floating):
                    vertex_colors = np.clip(vertex_colors * 255.0, 0, 255).astype(np.uint8)
                else:
                    vertex_colors = vertex_colors.astype(np.uint8)
            colors.append(vertex_colors[indexed].mean(axis=1).astype(np.uint8))
    if not triangles:
        raise ValueError(f"GLB contains no triangle POSITION data: {path}")
    return np.concatenate(triangles), np.concatenate(colors)
