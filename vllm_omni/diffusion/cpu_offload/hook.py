# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Hook-based CPU offloading implementation for vLLM-Omni.

This module implements a hook system that automatically manages device transfers
for model components, enabling memory-efficient inference without requiring
changes to individual pipeline code.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook


class CPUOffloadHook(ModelHook):
    """
    ModelHook implementing CPU offloading for model components.

    This hook automatically moves modules between CPU and GPU:
    - Moves to CPU in initialize_hook (storage)
    - Moves to GPU in pre_forward (before computation)
    - Optionally coordinates with other hooks for alternating offload strategy

    Key features:
    - Zero changes to model code
    - Automatic device management
    - Works with any torch.nn.Module
    """

    _HOOK_NAME = "cpu_offload"

    def __init__(
        self,
        execution_device: torch.device,
        other_hooks: list[CPUOffloadHook] | None = None,
    ):
        """
        Initialize CPUOffloadHook.

        Args:
            execution_device: Device to move module to during forward pass
            other_hooks: List of other CPUOffloadHooks to coordinate with
                        (for alternating offload strategy)
        """
        super().__init__()
        self.execution_device = execution_device
        self.other_hooks = other_hooks or []
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
        if next(module.parameters(recurse=False), None) is not None:
            current_device = next(module.parameters(recurse=False)).device
            if current_device != self.execution_device:
                # Offload other hooks to CPU first (alternating strategy)
                for other_hook in self.other_hooks:
                    if other_hook._module is not None:
                        other_hook.offload_to_cpu(other_hook._module)

                # Move this module to execution device
                module.to(self.execution_device)

        # Execute original forward
        return module._original_forward(*args, **kwargs)

    def offload_to_cpu(self, module: nn.Module) -> None:
        """
        Explicitly offload module to CPU.

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


def apply_cpu_offload_hook(
    module: nn.Module,
    execution_device: torch.device,
    other_hooks: list[CPUOffloadHook] | None = None,
) -> CPUOffloadHook:
    """
    Apply CPU offloading to a module using hooks.

    This function registers a CPUOffloadHook that automatically manages
    device transfers without requiring changes to the module code.

    Args:
        module: Module to apply CPU offloading to
        execution_device: Device to move module to during forward passes
        other_hooks: List of other CPUOffloadHooks to coordinate with

    Returns:
        The created CPUOffloadHook instance

    Example:
        >>> device = torch.device("cuda:0")
        >>> hook = apply_cpu_offload_hook(text_encoder, device)
        >>> # Module now automatically moves to GPU when forward() is called
    """
    registry = HookRegistry.get_or_create(module)
    hook = CPUOffloadHook(execution_device=execution_device, other_hooks=other_hooks)
    registry.register_hook(CPUOffloadHook._HOOK_NAME, hook)
    return hook

