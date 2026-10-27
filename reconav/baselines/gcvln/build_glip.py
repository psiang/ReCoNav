"""Build the pinned GLIP extension with PyTorch 2.5's CUDA compatibility APIs."""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from reconav import paths
from reconav.baselines.gcvln.run_single import verify_upstream


HEADER = """#pragma once
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDACachingAllocator.h>
#define THCudaCheck C10_CUDA_CHECK
template <typename T> __host__ __device__ inline T THCCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}
"""


def prepare_source(upstream: Path, destination: Path) -> None:
    """Export committed source and replace removed THC names; retain kernels."""
    verify_upstream(upstream)
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", "HEAD:third_party/GLIP"], cwd=upstream
    )
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("GLIP build source directory must be empty")
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk():
                raise ValueError("unexpected path in pinned GLIP source")
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.extractfile(member).read())

    csrc = destination / "maskrcnn_benchmark/csrc"
    (csrc / "torch_compat.h").write_text(HEADER)
    for path in csrc.rglob("*.cu"):
        text = path.read_text()
        for header in ("THC/THC.h", "THC/THCDeviceUtils.cuh", "THC/THCAtomics.cuh"):
            text = text.replace(f"#include <{header}>", '#include "torch_compat.h"')
        text = text.replace(
            "THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState", ""
        )
        text = text.replace(
            "THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long))",
            "c10::cuda::CUDACachingAllocator::raw_alloc(boxes_num * col_blocks * sizeof(unsigned long long))",
        )
        text = text.replace("THCudaFree(state, mask_dev)", "c10::cuda::CUDACachingAllocator::raw_delete(mask_dev)")
        path.write_text(text)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=paths.external_root() / "upstream/GC-VLN")
    args = parser.parse_args(argv)
    import torch
    from torch.utils.cpp_extension import CUDA_HOME
    if not torch.cuda.is_available() or CUDA_HOME is None:
        raise RuntimeError("install CUDA-enabled Torch and its CUDA toolkit before building GLIP")
    cache = paths.cache_root() / "external/build"
    cache.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix="glip-", dir=cache))
    prepare_source(args.upstream, build / "source")
    env = dict(os.environ)
    env.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "--wheel-dir", str(build / "wheels"), str(build / "source")], env=env, check=True
    )
    wheel, = (build / "wheels").glob("*.whl")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)], check=True)
    print(f"Installed GLIP CUDA extension; build retained in {build}")


if __name__ == "__main__":
    main()
