"""Smoke-check that the PyTorch install in the active conda env is sane.

Run from CI after `setup-miniconda` to catch obvious env rot:

    python env/verify_cuda_support.py

By default this asserts a CUDA-compiled PyTorch build is present. It does
NOT assert ``torch.cuda.is_available()`` because hosted CI runners have
no GPU. To enforce the runtime check (e.g. on a self-hosted GPU runner)
set the environment variable ``REQUIRE_CUDA_RUNTIME=1``.

To disable the CUDA-build assertion (e.g. for the CPU-only conda env)
set ``REQUIRE_CUDA_BUILD=0``.
"""

import os
import sys

import torch


def _truthy(env_var, default):
    return os.environ.get(env_var, default).lower() in {"1", "true", "yes"}


require_cuda_build = _truthy("REQUIRE_CUDA_BUILD", "1")
require_cuda_runtime = _truthy("REQUIRE_CUDA_RUNTIME", "0")

print("PyTorch version:    ", torch.__version__)
print("CUDA build version: ", torch.version.cuda)
print("CUDA available:     ", torch.cuda.is_available())

# Always assert the PyTorch version is recognisably 2.x; this catches a
# silent downgrade or a broken install where torch imports as something
# unexpected.
assert torch.__version__.startswith("2."), (
    f"Expected PyTorch 2.x, got {torch.__version__}"
)

if require_cuda_build:
    assert torch.version.cuda is not None, (
        "PyTorch was installed without CUDA support "
        "(torch.version.cuda is None). Check env/conda_gpu_env.yml."
    )

if require_cuda_runtime:
    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False on a runner that was "
        "expected to have a GPU. Set REQUIRE_CUDA_RUNTIME=0 if running "
        "on a CPU-only host."
    )
    # Touch the device to confirm a kernel can launch.
    _ = torch.zeros(1, device="cuda") + 1
    print("CUDA runtime smoke test: ok")

print("verify_cuda_support: ok", file=sys.stderr)
