# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU offloading utilities for diffusion models.

This module provides mutual-exclusion CPU offloading between DiT and encoders.
When dit_cpu_offload is enabled:
- Text encoders run on GPU while DiT is on CPU
- DiT runs on GPU while encoders are offloaded to CPU

This allows running large models on limited GPU memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


class SequentialOffloader:
    """Sequential offloader: DiT and encoders take turns on GPU.

    Uses PyTorch's forward pre-hooks to automatically swap models:
    - Before encoder runs: move DiT to CPU, move encoder to GPU
    - Before DiT runs: move encoders to CPU, move DiT to GPU

    This ensures only one large model is on GPU at a time.
    """

    def __init__(
        self,
        dit: nn.Module,
        encoders: list[nn.Module],
        device: torch.device,
        pin_memory: bool = True,
    ):
        self.dit = dit
        self.encoders = encoders
        self.device = device
        self.pin_memory = pin_memory
        self._handles: list = []

    def _to_cpu(self, module: nn.Module) -> None:
        """Move module to CPU with optional memory pinning."""
        # Skip if already on CPU
        try:
            param = next(module.parameters())
            if param.device.type == "cpu":
                return
        except StopIteration:
            return

        previous_device = param.device
        module.to("cpu", non_blocking=True)

        # Release allocator blocks when tensors leave the GPU.
        if previous_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.pin_memory:
            for p in module.parameters():
                if p.data.device.type == "cpu" and not p.data.is_pinned():
                    p.data = p.data.pin_memory()

    def _to_gpu(self, module: nn.Module) -> None:
        """Move module to GPU."""
        # Skip if already on target device
        try:
            if next(module.parameters()).device == self.device:
                return
        except StopIteration:
            return

        module.to(self.device, non_blocking=True)

    def _dit_pre_hook(self, module: nn.Module, args: tuple) -> None:
        """Before DiT forward: offload encoders, load DiT."""
        for enc in self.encoders:
            self._to_cpu(enc)
        self._to_gpu(module)
        torch.cuda.synchronize()
        logger.debug("Swapped: encoders -> CPU, DiT -> GPU")

    def _encoder_pre_hook(self, module: nn.Module, args: tuple) -> None:
        """Before encoder forward: offload DiT, load encoder."""
        self._to_cpu(self.dit)
        self._to_gpu(module)
        torch.cuda.synchronize()
        logger.debug("Swapped: DiT -> CPU, encoder -> GPU")

    def register(self) -> None:
        """Register forward pre-hooks on DiT and encoders."""
        # Hook on DiT
        h = self.dit.register_forward_pre_hook(self._dit_pre_hook)
        self._handles.append(h)
        logger.debug("Registered offload hook for DiT")

        # Hook on each encoder
        for enc in self.encoders:
            h = enc.register_forward_pre_hook(self._encoder_pre_hook)
            self._handles.append(h)
            logger.debug("Registered offload hook for %s", enc.__class__.__name__)

    def remove(self) -> None:
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []


def apply_offload_hooks(model: nn.Module, od_config: OmniDiffusionConfig) -> None:
    """Apply mutual-exclusion offload hooks based on config.

    When dit_cpu_offload is enabled, DiT and encoders swap GPU access:
    - Encoders (text_encoder, text_encoder_2, text_encoder_3, image_encoder)
      run on GPU while DiT is on CPU
    - DiT runs on GPU while encoders are on CPU

    Args:
        model: Diffusion pipeline model
        od_config: OmniDiffusionConfig with offload settings
    """
    if not getattr(od_config, "dit_cpu_offload", False):
        return

    # Find DiT/transformer/unet
    dit = None
    dit_attr = None
    for attr in ["transformer", "dit", "unet"]:
        if hasattr(model, attr) and getattr(model, attr) is not None:
            dit = getattr(model, attr)
            dit_attr = attr
            break

    if dit is None:
        logger.warning("dit_cpu_offload enabled but no transformer/dit/unet found")
        return

    # Capture execution device BEFORE moving DiT to CPU
    try:
        device = next(dit.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect all encoders
    encoders = []
    encoder_names = []
    for attr in ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder"]:
        if hasattr(model, attr) and getattr(model, attr) is not None:
            encoders.append(getattr(model, attr))
            encoder_names.append(attr)

    if not encoders:
        logger.warning("dit_cpu_offload enabled but no encoders found")
        return

    # Initial state: DiT on CPU (encoders run first, they stay on GPU)
    pin = getattr(od_config, "pin_cpu_memory", True)
    dit.to("cpu")
    print("Moved DiT to CPU for offloading.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if pin and torch.cuda.is_available():
        for p in dit.parameters():
            if not p.data.is_pinned():
                p.data = p.data.pin_memory()

    # Register sequential offload hooks
    SequentialOffloader(dit, encoders, device, pin).register()

    logger.info(
        "CPU offload enabled: %s <-> %s (mutual exclusion)",
        dit_attr,
        ", ".join(encoder_names),
    )
