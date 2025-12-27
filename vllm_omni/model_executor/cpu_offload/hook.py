# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Hook-based CPU offloading implementation for transformer models in vLLM-Omni.

This module implements a hook system that automatically manages device transfers
for transformer model components, enabling memory-efficient inference without
requiring changes to individual model code.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook


class TransformerCPUOffloadHook(ModelHook):
    """
    ModelHook implementing CPU offloading for transformer model components.

    This hook automatically moves modules between CPU and GPU:
    - Moves to CPU in initialize_hook (storage)
    - Moves to GPU in new_forward (before computation)
    - Optionally coordinates with other hooks for alternating offload strategy
    - Implements latency-minimizing mode (keeps on GPU after forward)

    Key features:
    - Zero changes to model code
    - Automatic device management
    - Works with any torch.nn.Module
    - Latency-minimizing: keeps components on GPU after forward pass
    """

    _HOOK_NAME = "transformer_cpu_offload"

    def __init__(
        self,
        execution_device: torch.device,
        other_hooks: list[TransformerCPUOffloadHook] | None = None,
        keep_on_gpu: bool = True,
    ):
        """
        Initialize TransformerCPUOffloadHook.

        Args:
            execution_device: Device to move module to during forward pass
            other_hooks: List of other TransformerCPUOffloadHooks to coordinate with
                        (for alternating offload strategy)
            keep_on_gpu: If True, keep module on GPU after forward (latency-minimizing).
                        If False, offload to CPU after forward (memory-saving).
        """
        super().__init__()
        self.execution_device = execution_device
        self.other_hooks = other_hooks or []
        self.keep_on_gpu = keep_on_gpu
        self._module: nn.Module | None = None

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        """
        Initialize hook by moving module to CPU.

        Args:
            module: The module to initialize the hook for.

        Returns:
            The initialized module (now on CPU).
        """
        self._module = module
        # Move to CPU for storage
        module.to("cpu")
        return module

    def new_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        """
        Execute forward pass with automatic device management.

        Args:
            module: The module to execute forward on
            *args: Forward arguments
            **kwargs: Forward keyword arguments

        Returns:
            Module output
        """
        # Check if module needs to be moved to execution device
        # Check parameters recursively to handle modules with submodules
        first_param = next(module.parameters(recurse=True), None)
        if first_param is not None:
            current_device = first_param.device
            if current_device != self.execution_device:
                # Offload other hooks to CPU first (alternating strategy)
                for other_hook in self.other_hooks:
                    if other_hook._module is not None:
                        other_hook.offload_to_cpu(other_hook._module)

                # Move this module to execution device
                module.to(self.execution_device)

        # Execute original forward
        result = module._original_forward(*args, **kwargs)

        # Latency-minimizing mode: keep on GPU after forward
        # (don't offload to CPU unless explicitly requested)
        if not self.keep_on_gpu:
            # Memory-saving mode: offload after forward
            self.offload_to_cpu(module)

        return result

    def offload_to_cpu(self, module: nn.Module) -> None:
        """
        Explicitly offload module to CPU.

        Note: pin_memory support (from config) is reserved for future implementation
        to enable faster CPU↔GPU transfers.

        Args:
            module: Module to offload
        """
        module.to("cpu")

    def reset_state(self, module: nn.Module) -> nn.Module:
        """
        Reset hook state (no-op for CPU offload).

        Args:
            module: The module to reset state for

        Returns:
            The module
        """
        return module


def apply_transformer_cpu_offload_hook(
    module: nn.Module,
    execution_device: torch.device,
    other_hooks: list[TransformerCPUOffloadHook] | None = None,
    keep_on_gpu: bool = True,
) -> TransformerCPUOffloadHook:
    """
    Apply CPU offloading to a transformer module using hooks.

    This function registers a TransformerCPUOffloadHook that automatically manages
    device transfers without requiring changes to the module code.

    Args:
        module: Module to apply CPU offloading to
        execution_device: Device to move module to during forward passes
        other_hooks: List of other TransformerCPUOffloadHooks to coordinate with
        keep_on_gpu: If True, keep on GPU after forward (latency-minimizing).
                     If False, offload after forward (memory-saving).

    Returns:
        The created TransformerCPUOffloadHook instance

    Example:
        >>> device = torch.device("cuda:0")
        >>> hook = apply_transformer_cpu_offload_hook(thinker_model, device)
        >>> # Module now automatically moves to GPU when forward() is called
    """
    registry = HookRegistry.get_or_create(module)
    hook = TransformerCPUOffloadHook(
        execution_device=execution_device, other_hooks=other_hooks, keep_on_gpu=keep_on_gpu
    )
    registry.register_hook(TransformerCPUOffloadHook._HOOK_NAME, hook)
    return hook

