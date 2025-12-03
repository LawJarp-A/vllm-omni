# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TeaCache: Timestep Embedding Aware Cache for diffusion model acceleration.

TeaCache speeds up diffusion inference by reusing transformer block computations
when consecutive timestep embeddings are similar.
"""

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.core import TeaCacheWrapper, apply_teacache
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState

__all__ = [
    "TeaCacheConfig",
    "TeaCacheWrapper",
    "TeaCacheState",
    "apply_teacache",
]
