"""Configurable data, output, model and optional baseline locations."""
from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root if (root / "pyproject.toml").is_file() else Path.cwd().resolve()


def _configured(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def data_root() -> Path:
    return _configured("RECONAV_DATA_ROOT", Path.cwd() / "data")


def cache_root() -> Path:
    return _configured("RECONAV_CACHE_ROOT", repository_root() / ".cache")


def output_root() -> Path:
    return _configured("RECONAV_OUTPUT_ROOT", repository_root() / "outputs")


def external_root() -> Path:
    return _configured("RECONAV_EXTERNAL_ROOT", repository_root() / "external")


def model_root() -> Path:
    return _configured("RECONAV_MODEL_ROOT", external_root() / "models")


def lang2ltl_embeddings() -> Path:
    return _configured("RECONAV_LANG2LTL_EMBEDDINGS", external_root() / "resources/lang2ltl2/embeddings.zip")


def habitat_navigation_root() -> Path:
    return _configured("RECONAV_HABITAT_NAVIGATION_ROOT", external_root() / "resources/habitat-navigation")


def navigation_root() -> Path:
    return _configured("RECONAV_NAVIGATION_ROOT", data_root() / "benchmark/scenes")


def benchmark_path(*parts: str | Path) -> Path:
    return data_root() / "benchmark" / Path(*parts)


def hm3d_root() -> Path:
    return _configured("RECONAV_HM3D_ROOT", external_root() / "hm3d/val")


def upstream_path(*parts: str | Path) -> Path:
    return external_root() / "upstream" / Path(*parts)


def runtime_path(*parts: str | Path) -> Path:
    return external_root() / "runtime_assets" / Path(*parts)


def compat_path(*parts: str | Path) -> Path:
    return external_root() / "compat" / Path(*parts)


def resource_path(*parts: str | Path) -> Path:
    return Path(__file__).resolve().parent / "baselines/resources" / Path(*parts)


def environment_path(label: str, *parts: str | Path) -> Path:
    name = "RECONAV_ENV_" + label.upper().replace("-", "_")
    return _configured(name, repository_root() / ".environments" / label) / Path(*parts)


def configure(data: str | Path | None = None) -> Path:
    if data is not None:
        os.environ["RECONAV_DATA_ROOT"] = str(Path(data).expanduser().resolve())
    # Keep relative user settings stable when a worker changes its working directory.
    for name, resolve in (("RECONAV_DATA_ROOT", data_root),
                          ("RECONAV_CACHE_ROOT", cache_root),
                          ("RECONAV_OUTPUT_ROOT", output_root),
                          ("RECONAV_EXTERNAL_ROOT", external_root),
                          ("RECONAV_MODEL_ROOT", model_root),
                          ("RECONAV_HM3D_ROOT", hm3d_root),
                          ("RECONAV_LANG2LTL_EMBEDDINGS", lang2ltl_embeddings),
                          ("RECONAV_HABITAT_NAVIGATION_ROOT", habitat_navigation_root),
                          ("RECONAV_NAVIGATION_ROOT", navigation_root)):
        os.environ[name] = str(resolve())
    root = data_root()
    os.environ.setdefault("HM3DSEM_OBJECT_VOCABULARY", str(root / "benchmark/vocabularies/hm3dsem_category_mappings.tsv"))
    os.environ.setdefault("NAV_GEOMETRY_CACHE_ROOT", str(cache_root() / "geometry"))
    return root
