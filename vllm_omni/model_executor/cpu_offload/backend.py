# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CPU Offload backend implementation for transformer models.

This module provides a backend that applies CPU offloading hooks to transformer
model components, enabling memory-efficient inference without requiring changes
to individual model code.
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.cpu_offload.hook import (
    TransformerCPUOffloadHook,
    apply_transformer_cpu_offload_hook,
)

logger = init_logger(__name__)


class TransformerCPUOffloadBackend:
    """
    Backend for applying CPU offloading hooks to transformer model components.

    This backend applies CPU offloading hooks to model components (thinker, talker,
    code2wav, visual, audio_tower, language_model) based on configuration flags.
    It implements an alternating offload strategy where components are moved to
    GPU only when actively used.

    Example:
        >>> from vllm_omni.config.model import OmniModelConfig
        >>> config = OmniModelConfig(
        ...     cpu_offload_enabled=True,
        ...     cpu_offload_components=["thinker", "talker"],
        ... )
        >>> backend = TransformerCPUOffloadBackend(config, device=torch.device("cuda:0"))
        >>> backend.enable(model)
    """

    def __init__(self, config: Any, device: torch.device):
        """
        Initialize transformer CPU offload backend.

        Args:
            config: Configuration object with CPU offload flags (OmniModelConfig)
            device: Execution device (typically GPU)
        """
        self.config = config
        self.device = device
        self.enabled = False
        self.hooks: dict[str, TransformerCPUOffloadHook] = {}

    def _get_component(self, model: Any, component_name: str) -> Any | None:
        """
        Get a component from the model by name.

        Args:
            model: The model instance
            component_name: Name of the component (e.g., "thinker", "visual", "audio_tower")

        Returns:
            The component if found, None otherwise
        """
        if hasattr(model, component_name):
            return getattr(model, component_name)
        return None

    def _matches_component_pattern(self, component_name: str, patterns: list[str]) -> bool:
        """
        Check if a component name matches any of the given patterns.

        Args:
            component_name: Name of the component
            patterns: List of patterns (supports wildcard matching with "*")

        Returns:
            True if component matches any pattern, False otherwise
        """
        if not patterns:
            return False

        for pattern in patterns:
            if pattern == component_name:
                return True
            # Simple wildcard matching: "visual.*" matches "visual" and "visual.encoder"
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if component_name.startswith(prefix):
                    return True
            elif "*" in pattern:
                # Basic wildcard support
                import fnmatch

                if fnmatch.fnmatch(component_name, pattern):
                    return True

        return False

    def enable(self, model: Any) -> None:
        """
        Enable CPU offloading on model components using hooks.

        Applies hooks to model components based on configuration flags.
        Hooks coordinate to implement alternating offload strategy.

        Args:
            model: Transformer model instance. Supports:
                  - Multi-stage components: thinker, talker, code2wav
                  - Sub-components: visual, audio_tower, language_model
        """
        if not self.config.cpu_offload_enabled:
            return

        components_to_offload = self.config.cpu_offload_components
        if components_to_offload is None:
            # If no components specified, offload all available components
            components_to_offload = []

        hooks_list: list[TransformerCPUOffloadHook] = []
        keep_on_gpu = True  # Latency-minimizing mode (default)

        # Multi-stage components
        stage_components = ["thinker", "talker", "code2wav", "token2wav"]
        for comp_name in stage_components:
            if components_to_offload and not self._matches_component_pattern(comp_name, components_to_offload):
                continue

            component = self._get_component(model, comp_name)
            if component is not None:
                hook = apply_transformer_cpu_offload_hook(
                    component, self.device, other_hooks=hooks_list, keep_on_gpu=keep_on_gpu
                )
                self.hooks[comp_name] = hook
                hooks_list.append(hook)

        # Sub-components (visual, audio_tower, language_model)
        sub_components = ["visual", "audio_tower", "language_model"]
        for comp_name in sub_components:
            if components_to_offload and not self._matches_component_pattern(comp_name, components_to_offload):
                continue

            # Try to get from model directly
            component = self._get_component(model, comp_name)
            if component is None:
                # Try to get from thinker/talker if model has them
                for stage_comp in ["thinker", "talker"]:
                    stage = self._get_component(model, stage_comp)
                    if stage is not None:
                        component = self._get_component(stage, comp_name)
                        if component is not None:
                            break

            if component is not None:
                hook = apply_transformer_cpu_offload_hook(
                    component, self.device, other_hooks=hooks_list, keep_on_gpu=keep_on_gpu
                )
                self.hooks[comp_name] = hook
                hooks_list.append(hook)

        # Update all hooks to know about each other for coordination
        for hook in hooks_list:
            hook.other_hooks = [h for h in hooks_list if h is not hook]

        self.enabled = True
        if not self.hooks:
            logger.warning("CPU offloading enabled but no components found to offload")

    def is_enabled(self) -> bool:
        """
        Check if CPU offloading is enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self.enabled

    def ensure_component_on_gpu(self, component_name: str) -> None:
        """
        Ensure a specific component is on GPU before cross-stage data transfer.

        This is useful when preparing outputs for the next stage to ensure
        the component is on GPU even if it was offloaded after forward pass.

        Note: Currently not used in the codebase but available for future
        cross-stage data transfer optimizations.

        Args:
            component_name: Name of the component to ensure is on GPU
        """
        if not self.enabled:
            return

        hook = self.hooks.get(component_name)
        if hook is not None and hook._module is not None:
            # Check if module is on CPU and move to GPU if needed
            # Use recurse=True to handle modules with submodules
            first_param = next(hook._module.parameters(recurse=True), None)
            if first_param is not None:
                current_device = first_param.device
                if current_device != self.device:
                    hook._module.to(self.device)

