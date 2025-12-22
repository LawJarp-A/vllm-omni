# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CPU Offload backend implementation.

This module provides a backend that applies CPU offloading hooks to pipeline
components, enabling memory-efficient inference without requiring changes
to individual pipeline code.
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cpu_offload.hook import CPUOffloadHook, apply_cpu_offload_hook
from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


class CPUOffloadBackend:
    """
    Backend for applying CPU offloading hooks to pipeline components.

    This backend applies CPU offloading hooks to text_encoder, transformer,
    and VAE components based on configuration flags. It implements an
    alternating offload strategy where components are moved to GPU only
    when actively used.

    Example:
        >>> from vllm_omni.diffusion.data import OmniDiffusionConfig
        >>> config = OmniDiffusionConfig(
        ...     dit_cpu_offload=True,
        ...     text_encoder_cpu_offload=True,
        ...     vae_cpu_offload=True,
        ... )
        >>> backend = CPUOffloadBackend(config, device=torch.device("cuda:0"))
        >>> backend.enable(pipeline)
    """

    def __init__(self, config: OmniDiffusionConfig, device: torch.device):
        """
        Initialize CPU offload backend.

        Args:
            config: OmniDiffusionConfig with CPU offload flags
            device: Execution device (typically GPU)
        """
        self.config = config
        self.device = device
        self.enabled = False
        self.hooks: dict[str, CPUOffloadHook] = {}

    def enable(self, pipeline: Any) -> None:
        """
        Enable CPU offloading on pipeline components using hooks.

        Applies hooks to text_encoder, transformer, and VAE based on
        configuration flags. Hooks coordinate to implement alternating
        offload strategy.

        Args:
            pipeline: Diffusion pipeline instance. Extracts:
                     - text_encoder: pipeline.text_encoder (if exists)
                     - transformer: pipeline.transformer
                     - vae: pipeline.vae (if exists)
        """
        hooks_list: list[CPUOffloadHook] = []

        # Apply hook to text_encoder if enabled
        if self.config.text_encoder_cpu_offload and hasattr(pipeline, "text_encoder"):
            text_encoder = pipeline.text_encoder
            if text_encoder is not None:
                hook = apply_cpu_offload_hook(text_encoder, self.device)
                self.hooks["text_encoder"] = hook
                hooks_list.append(hook)
                logger.info("CPU offloading enabled for text_encoder")

        # Apply hook to transformer (DIT) if enabled
        if self.config.dit_cpu_offload and hasattr(pipeline, "transformer"):
            transformer = pipeline.transformer
            if transformer is not None:
                hook = apply_cpu_offload_hook(transformer, self.device, other_hooks=hooks_list)
                self.hooks["transformer"] = hook
                hooks_list.append(hook)
                logger.info("CPU offloading enabled for transformer (DIT)")

        # Update all hooks to know about each other for coordination
        for hook in hooks_list:
            hook.other_hooks = [h for h in hooks_list if h is not hook]

        # Apply hook to VAE if enabled (VAE doesn't need coordination with others)
        if self.config.vae_cpu_offload and hasattr(pipeline, "vae"):
            vae = pipeline.vae
            if vae is not None:
                hook = apply_cpu_offload_hook(vae, self.device)
                self.hooks["vae"] = hook
                logger.info("CPU offloading enabled for VAE")

        # Apply hook to image_encoder if enabled
        if self.config.image_encoder_cpu_offload and hasattr(pipeline, "image_encoder"):
            image_encoder = pipeline.image_encoder
            if image_encoder is not None:
                hook = apply_cpu_offload_hook(image_encoder, self.device)
                self.hooks["image_encoder"] = hook
                logger.info("CPU offloading enabled for image_encoder")

        self.enabled = True
        logger.info(
            f"CPU offloading enabled for components: {list(self.hooks.keys())}"
        )

    def is_enabled(self) -> bool:
        """
        Check if CPU offloading is enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self.enabled

