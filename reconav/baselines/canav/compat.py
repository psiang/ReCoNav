"""Runtime compatibility helpers for CA-Nav model artifacts."""

from __future__ import annotations

import torch


def configure_groundingdino_extension() -> str:
    """Use the extension built for this environment alongside pinned source."""
    import importlib.util
    import site
    import sys
    from pathlib import Path
    from reconav import paths
    from reconav.baselines.common.upstream import verify_recorded_source
    source = paths.upstream_path('GroundingDINO-canav')
    verify_recorded_source(source, 'GroundingDINO-canav')
    sys.path.insert(0, str(source))
    import groundingdino

    for root in site.getsitepackages():
        package = Path(root) / "groundingdino"
        if next(package.glob("_C*.so"), None):
            binary = next(package.glob('_C*.so'))
            name = 'groundingdino._C'
            spec = importlib.util.spec_from_file_location(name, binary)
            extension = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(extension)
            sys.modules[name] = extension
            groundingdino._C = extension
            break
    try:
        from groundingdino import _C
    except ImportError as error:
        raise RuntimeError("build and install GroundingDINO in the CA-Nav worker environment") from error
    return str(_C.__file__)


def repair_legacy_gelu_modules(module: torch.nn.Module) -> int:
    """Restore the old GELU default omitted from PyTorch 1.10 pickles."""

    repaired = 0
    for child in module.modules():
        if isinstance(child, torch.nn.GELU) and not hasattr(child, "approximate"):
            child.approximate = "none"
            repaired += 1
    return repaired


def install_habitat_sensor_spec_compat() -> bool:
    """Bridge Habitat-Lab 0.1.7's SensorSpec API to Habitat-Sim 0.2.1."""

    import habitat_sim

    if hasattr(habitat_sim.SensorSpec(), "parameters"):
        return False

    class _ParameterProxy:
        def __init__(self, spec: habitat_sim.CameraSensorSpec) -> None:
            self._spec = spec

        def __setitem__(self, key: str, value: str) -> None:
            if key != "hfov":
                raise KeyError(f"unsupported legacy camera parameter: {key}")
            self._spec.hfov = float(value)

    class LegacyCameraSensorSpec(habitat_sim.CameraSensorSpec):
        @property
        def parameters(self) -> _ParameterProxy:
            return _ParameterProxy(self)

    habitat_sim.SensorSpec = LegacyCameraSensorSpec
    return True
