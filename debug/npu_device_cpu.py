import numpy as np
import torch
import torch.cpu

import triton
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver
from triton.backends import ascend


class FakeAscendBackend(ascend.compiler.AscendBackend):
    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu"


class FakeAscendDriver(GPUDriver):
    def __init__(self):
        super().__init__()
        import torch
        import torch.cpu

        self.get_current_device = torch.cpu.current_device
        self.set_current_device = torch.cpu.set_device
        self.get_current_stream = torch.cpu.current_stream

    def get_current_target(self):
        warp_size = 0
        # ascend/backend/utils.py
        # arch = "Ascend910B4"
        arch = "Ascend910_9599"
        # return GPUTarget("cpu", arch, warp_size)
        return GPUTarget("npu", arch, warp_size)

    def get_active_torch_device(self):
        import torch

        return torch.device("cpu")

    def get_device_interface(self):
        import torch

        return torch.cpu

    @staticmethod
    def is_active():
        return True

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device="cpu")

    def clear_cache(self, cache):
        cache.zero_()


ascend.compiler.AscendBackend = FakeAscendBackend  # noqa
# not work
ascend.compiler.supports_target = FakeAscendBackend.supports_target  # noqa
ascend.driver.XtensaDriver = FakeAscendDriver  # noqa

triton.runtime.driver.set_active(FakeAscendDriver())

DEVICE = "cpu"
torch.cpu.set_device(DEVICE)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
