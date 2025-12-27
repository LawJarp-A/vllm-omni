# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CPU offloading support for transformer models in vLLM-Omni.

This module provides hook-based CPU offloading for transformer/autoregressive
models, enabling memory-efficient inference without requiring changes to model code.
"""

from vllm_omni.model_executor.cpu_offload.backend import TransformerCPUOffloadBackend
from vllm_omni.model_executor.cpu_offload.hook import (
    TransformerCPUOffloadHook,
    apply_transformer_cpu_offload_hook,
)

__all__ = [
    "TransformerCPUOffloadBackend",
    "TransformerCPUOffloadHook",
    "apply_transformer_cpu_offload_hook",
]

