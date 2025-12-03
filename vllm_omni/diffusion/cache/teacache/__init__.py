# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TeaCache: Timestep Embedding Aware Cache for diffusion model acceleration.

TeaCache speeds up diffusion inference by reusing transformer block computations
when consecutive timestep embeddings are similar.

This implementation uses a hooks-based approach that requires zero changes to
model code. Model developers only need to add an extractor function to support
new models.

Usage:
    # Recommended: Use enable_teacache flag
    omni = Omni(model="Qwen/Qwen-Image")
    images = omni.generate(prompt, enable_teacache=True)

    # Advanced: Direct pipeline access
    pipeline.transformer.enable_teacache(rel_l1_thresh=0.2)
    images = pipeline(prompt)
"""

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook, apply_teacache_hook
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState
from vllm_omni.diffusion.cache.teacache.extractors import register_extractor

__all__ = [
    "TeaCacheConfig",
    "TeaCacheState",
    "TeaCacheHook",
    "apply_teacache_hook",
    "register_extractor",
]
