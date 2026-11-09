"""Check that the torch inside the vLLM environment can use the GPUs (run with that env's python).

Exits non-zero with a hint when the CUDA build of torch does not match the driver, or a GPU is
missing. Run by the Kaggle notebook's setup stage; usable anywhere:
    /kaggle/working/vllm-env/bin/python scripts/check_vllm_gpu.py
"""
from __future__ import annotations

import sys

EXPECTED_GPU_COUNT = 2  # Kaggle "GPU T4 x2"


def main() -> int:
    import torch

    print(f"torch {torch.__version__} built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not usable from the vLLM environment. The NVIDIA driver is probably "
              "older than this torch build needs. Pin an older vLLM in the notebook "
              "(VLLM_SPEC, e.g. \"vllm==0.10.2\"), delete /kaggle/working/vllm-env, and re-run setup.")
        return 1
    gpu_count = torch.cuda.device_count()
    for gpu_index in range(gpu_count):
        major, minor = torch.cuda.get_device_capability(gpu_index)
        print(f"GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}, compute capability {major}.{minor}")
    # a real kernel launch catches builds that see the GPU but ship no kernels for sm_75
    product = torch.ones(64, 64, device="cuda", dtype=torch.float16) @ torch.ones(64, 64, device="cuda",
                                                                                  dtype=torch.float16)
    torch.cuda.synchronize()
    print(f"fp16 matmul on GPU OK ({float(product[0, 0])})")
    if gpu_count < EXPECTED_GPU_COUNT:
        print(f"ERROR: {gpu_count} GPU(s) visible, the config uses tensor_parallel_size=2. "
              "Select accelerator 'GPU T4 x2'.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
