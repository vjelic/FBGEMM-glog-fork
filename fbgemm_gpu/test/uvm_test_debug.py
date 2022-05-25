#!/usr/bin/env python3

# pyre-ignore-all-errors[56]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List

import fbgemm_gpu
import hypothesis.strategies as st
import torch

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_available, gpu_unavailable

if gpu_available:
    # pyre-ignore[21]
    from fbgemm_gpu.uvm import cudaMemAdvise, hipMemoryAdvise, cudaMemPrefetchAsync

from hypothesis import given, settings, Verbosity

MAX_EXAMPLES = 40


class UvmTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @given(
        sizes=st.just([794, 735, 868, 604]),
        vanilla=st.just(True),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_is_uvm_tensor(self, sizes: List[int], vanilla: bool) -> None:
        op = (
            torch.ops.fbgemm.new_managed_tensor
            if not vanilla
            else torch.ops.fbgemm.new_vanilla_managed_tensor
        )
        uvm_t = op(torch.empty(0, device="cuda:0", dtype=torch.float), sizes)
        assert torch.ops.fbgemm.is_uvm_tensor(uvm_t)
        assert torch.ops.fbgemm.uvm_storage(uvm_t)


if __name__ == "__main__":
    unittest.main()
