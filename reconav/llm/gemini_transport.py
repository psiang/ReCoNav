"""Lossless, non-recursive Gemini serialization of the common JSON contract.

Only the wire representation changes. No source-text inference, default values,
reference truncation, semantic repair, or instance selection happens here.
The caller still validates the reconstructed common contract.
"""
from __future__ import annotations

import json
import math
from typing import Any

VERSION = "gemini-flat-paths-v3-qualified-references"
TABLE_TYPES = {"strings": str, "integers": int, "numbers": float, "booleans": bool}
CONTAINERS = ("objects", "arrays", "nulls")


def needs_transport(response_format: dict | None) -> bool:
    schema = (response_format or {}).get("json_schema", {}).get("schema", {})
    return "NominalReference" in schema.get("$defs", {})


def response_format() -> dict:
    def array(items):
        return {"type": "array", "items": items}
    properties = {name: array({"type": "string"}) for name in CONTAINERS}
    for name, kind in {"strings": "string", "integers": "integer", "numbers": "number", "booleans": "boolean"}.items():
        properties[name] = array({
            "type": "object", "properties": {"path": {"type": "string"}, "value": {"type": kind}},
            "required": ["path", "value"], "additionalProperties": False,
        })
    return {"type": "json_schema", "json_schema": {
        "name": "flat_semantic_transport", "strict": True,
        "schema": {"type": "object", "properties": properties,
                   "required": list(properties), "additionalProperties": False},
    }}


def _escape(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def encode(value: Any) -> dict:
    result = {name: [] for name in (*CONTAINERS, *TABLE_TYPES)}
    def visit(item, path):
        if isinstance(item, dict):
            result["objects"].append(path)
            for key, child in item.items():
                visit(child, path + "/" + _escape(key))
        elif isinstance(item, list):
            result["arrays"].append(path)
            for index, child in enumerate(item):
                visit(child, path + "/" + str(index))
        elif item is None:
            result["nulls"].append(path)
        else:
            name = next((name for name, kind in TABLE_TYPES.items() if type(item) is kind), None)
            if name is None or isinstance(item, float) and not math.isfinite(item):
                raise ValueError("unsupported JSON value")
            result[name].append({"path": path, "value": item})
    visit(value, "")
    return result


def decode(value: Any) -> Any:
    names = set(CONTAINERS) | set(TABLE_TYPES)
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError("flat transport requires exactly seven tables")
    nodes = {}
    def register(path, item):
        if not isinstance(path, str) or path and not path.startswith("/"):
            raise ValueError("invalid JSON pointer")
        if any("~" in part.replace("~1", "").replace("~0", "") for part in path.split("/")):
            raise ValueError("invalid JSON pointer escape")
        if path in nodes:
            raise ValueError(f"duplicate flat path: {path}")
        nodes[path] = item
    for name in names:
        if not isinstance(value[name], list):
            raise ValueError("flat tables must be arrays")
        for row in value[name]:
            if name in CONTAINERS:
                register(row, {} if name == "objects" else [] if name == "arrays" else None)
            else:
                if not isinstance(row, dict) or set(row) != {"path", "value"}:
                    raise ValueError("flat value row requires path and value")
                item = row["value"]
                if type(item) is not TABLE_TYPES[name] and not (name == "numbers" and type(item) is int):
                    raise ValueError(f"incorrect primitive type in {name}")
                if name == "numbers":
                    item = float(item)
                    if not math.isfinite(item):
                        raise ValueError("nonfinite number")
                register(row["path"], item)
    if "" not in nodes:
        raise ValueError("missing root node")
    array_children = {}
    for path, item in nodes.items():
        if not path:
            continue
        parent_path, _, key = path.rpartition("/")
        if parent_path not in nodes:
            raise ValueError(f"missing parent node: {parent_path}")
        parent = nodes[parent_path]
        if isinstance(parent, dict):
            parent[key.replace("~1", "/").replace("~0", "~")] = item
        elif isinstance(parent, list):
            if not key.isascii() or not key.isdigit() or str(int(key)) != key:
                raise ValueError("array indexes must be nonnegative canonical integers")
            array_children.setdefault(parent_path, {})[int(key)] = item
        else:
            raise ValueError("a primitive node cannot have children")
    for path, children in array_children.items():
        if sorted(children) != list(range(len(children))):
            raise ValueError(f"noncontiguous array indexes: {path}")
        nodes[path].extend(children[index] for index in range(len(children)))
    return nodes[""]


def prepare(system: str, user: str, kwargs: dict) -> tuple[str, str, dict]:
    schema = kwargs["response_format"]["json_schema"]["schema"]
    example = {"route": {"via_events": []}, "status": "unresolved", "semantic_contract": None}
    patch = (
        "\n\nGemini wire serialization (changes JSON format only):\n"
        "Preserve every semantic requirement above. Represent the requested canonical JSON as seven flat tables. "
        "All paths are RFC 6901 JSON pointers: root is the empty string, object keys use /key, array indexes use /0. "
        "Declare EVERY object and array, including the root and empty containers, in objects and arrays. "
        "Put null-valued paths in nulls. Put primitive leaves in strings, integers, numbers, or booleans as "
        '{"path": pointer, "value": actual_value}. Each path occurs exactly once; array indexes are contiguous. '
        "Always include all seven tables. Do not emit the canonical nested object directly. "
        "The decoder only reconstructs JSON and supplies no omitted semantic fields or values. "
        "Nominal qualifier reference quantifiers must explicitly be unique or exists.\n"
        "Nominal references may carry their own qualifiers; preserve nested reference qualifications "
        "as specified by the canonical schema. An unqualified reference may use qualifiers=null.\n"
        "Serialization example (illustrates format only, not a task answer):\n"
        + json.dumps(example, ensure_ascii=False) + "\nbecomes\n"
        + json.dumps(encode(example), ensure_ascii=False) + "\n"
        "Canonical JSON schema to represent (all original fields, meanings, and required constraints apply):\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    )
    prepared = {**kwargs, "response_format": response_format()}
    if "max_tokens" in prepared:
        # Flat paths repeat structural names; Gemini's cap also covers thinking.
        # Fixed for every semantic method, independent of score or retry number.
        prepared["max_tokens"] = max(int(prepared["max_tokens"]), 32768)
    return system + patch, user, prepared

