from __future__ import annotations

"""
Hook-based TeaCache implementation for vLLM-Omni.

This module implements a diffusers-style hook system that completely intercepts
the transformer forward pass, eliminating the need for any TeaCache-specific
code in model definitions. Model developers only need to add an extractor function
to support new models.
"""

from typing import Any, Optional, Union

import numpy as np
import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import get_extractor
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook, StateManager


class TeaCacheHook(ModelHook):
    """
    ModelHook implementing TeaCache for transformer models.

    This hook completely intercepts the transformer's forward pass and implements
    adaptive caching based on timestep embedding similarity. It's model-agnostic
    and supports multiple model types through extractor functions.

    Key features:
    - Zero changes to model code
    - CFG-aware with separate states for positive/negative branches
    - Model-specific polynomial rescaling
    - Auto-detection of model types

    Attributes:
        config: TeaCache configuration with thresholds and callbacks
        rescale_func: Polynomial function for rescaling L1 distances
        state_manager: Manages TeaCacheState across forward passes
        extractor_fn: Model-specific function to extract modulated input
    """

    _HOOK_NAME = "teacache"

    def __init__(self, config: TeaCacheConfig):
        super().__init__()
        self.config = config
        self.rescale_func = np.poly1d(config.coefficients)
        self.state_manager = StateManager(TeaCacheState)
        self.extractor_fn = None

    def initialize_hook(self, module):
        """Initialize hook with auto-detected extractor and model type."""
        # Auto-detect extractor function for this model
        self.extractor_fn = get_extractor(module)

        # Set default context
        self.state_manager.set_context("teacache")

        return module

    def new_forward(self, module, *args: Any, **kwargs: Any):
        """
        Route to model-specific forward handler based on module type.

        This allows the hook to support multiple model architectures without
        changing the hook registry system.
        """
        module_class_name = module.__class__.__name__

        if "QwenImage" in module_class_name:
            return self._handle_qwen_forward(module, *args, **kwargs)
        else:
            # For unsupported models, fall back to original forward
            # This allows graceful degradation
            raise NotImplementedError(
                f"TeaCache hook does not support model type: {module_class_name}. "
                f"Please add a handler method for this model."
            )

    def _handle_qwen_forward(
        self,
        module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        img_shapes: torch.Tensor,
        txt_seq_lens: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[dict[str, Any]] = None,
        cache_branch: Optional[str] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Handle QwenImageTransformer2DModel forward pass with TeaCache.

        This method completely replicates the model's forward pass while adding
        adaptive caching logic. The original model code remains untouched.

        Args:
            module: The QwenImageTransformer2DModel instance
            hidden_states: Input latent tensor
            encoder_hidden_states: Text encoder outputs
            encoder_hidden_states_mask: Mask for text encoder
            timestep: Current diffusion timestep
            img_shapes: Image shapes for position embedding
            txt_seq_lens: Text sequence lengths
            guidance: Optional guidance scale for CFG
            attention_kwargs: Additional attention arguments
            cache_branch: CFG branch identifier ("positive", "negative", or None)
            return_dict: Whether to return a dict or tuple

        Returns:
            Transformer2DModelOutput or tuple with denoised output
        """
        # ============================================================================
        # PREPROCESSING (same as original forward)
        # ============================================================================
        hidden_states = module.img_in(hidden_states)

        # Ensure timestep tensor is on the same device and dtype as hidden_states
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype)
        encoder_hidden_states = module.txt_norm(encoder_hidden_states)
        encoder_hidden_states = module.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            module.time_text_embed(timestep, hidden_states)
            if guidance is None
            else module.time_text_embed(timestep, guidance, hidden_states)
        )

        image_rotary_emb = module.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        # ============================================================================
        # TEACACHE LOGIC (hook-based, model-agnostic)
        # ============================================================================
        # Set context based on CFG branch for separate state tracking
        branch = cache_branch if cache_branch is not None else "default"
        context_name = f"teacache_{branch}"
        self.state_manager.set_context(context_name)
        state = self.state_manager.get_state()

        # Extract modulated input from first transformer block
        inp = hidden_states.clone()
        temb_clone = temb.clone()
        modulated_inp = self.extractor_fn(module, inp, temb_clone)

        # Decide whether to compute or cache based on modulated input similarity
        should_compute = self._should_compute_full_transformer(state, modulated_inp)

        if not should_compute and state.previous_residual is not None:
            # ============================================================================
            # FAST PATH: Reuse cached residuals
            # ============================================================================
            hidden_states = hidden_states + state.previous_residual
            if state.previous_residual_encoder is not None:
                encoder_hidden_states = encoder_hidden_states + state.previous_residual_encoder
        else:
            # ============================================================================
            # SLOW PATH: Full transformer computation
            # ============================================================================
            ori_hidden_states = hidden_states.clone()
            ori_encoder_hidden_states = encoder_hidden_states.clone()

            # Run all transformer blocks
            for block in module.transformer_blocks:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )

            # Cache residuals for next timestep
            state.previous_residual = (hidden_states - ori_hidden_states).detach()
            state.previous_residual_encoder = (encoder_hidden_states - ori_encoder_hidden_states).detach()

        # Update state
        state.previous_modulated_input = modulated_inp.detach()
        state.cnt += 1

        # ============================================================================
        # POSTPROCESSING (same as original forward)
        # ============================================================================
        hidden_states = module.norm_out(hidden_states, temb)
        output = module.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def _should_compute_full_transformer(
        self, state: TeaCacheState, modulated_inp: torch.Tensor
    ) -> bool:
        """
        Determine whether to compute full transformer or reuse cached residual.

        This implements the core TeaCache algorithm:
        1. Always compute first timestep
        2. For intermediate steps:
           - Compute relative L1 distance between current and previous modulated inputs
           - Apply polynomial rescaling with model-specific coefficients
           - Accumulate rescaled distances
           - Compare to threshold: below = cache, above = compute

        Args:
            state: Current TeaCacheState containing counters and cached values
            modulated_inp: Modulated input extracted from first transformer block

        Returns:
            True to compute full transformer, False to reuse cached residual
        """
        # First timestep: always compute
        if state.cnt == 0:
            state.accumulated_rel_l1_distance = 0.0
            return True

        # Need previous input for comparison
        if state.previous_modulated_input is None:
            return True

        # Compute relative L1 distance between consecutive modulated inputs
        rel_distance = (
            (
                (modulated_inp - state.previous_modulated_input).abs().mean()
                / (state.previous_modulated_input.abs().mean() + 1e-8)
            )
            .cpu()
            .item()
        )

        # Apply model-specific polynomial rescaling
        rescaled_distance = float(self.rescale_func(rel_distance))
        state.accumulated_rel_l1_distance += abs(rescaled_distance)

        # Decision: below threshold = cache, above = compute
        if state.accumulated_rel_l1_distance < self.config.rel_l1_thresh:
            return False  # Use cache
        else:
            state.accumulated_rel_l1_distance = 0.0  # Reset accumulator
            return True  # Compute

    def reset_state(self, module):
        """Reset all cached states for a new inference run."""
        self.state_manager.reset()
        return module


def apply_teacache_hook(module, config: TeaCacheConfig) -> None:
    """
    Apply TeaCache optimization to a transformer module.

    This function registers a TeaCacheHook that completely intercepts the
    module's forward pass, implementing adaptive caching without any changes
    to the model code.

    Args:
        module: Transformer model to optimize (e.g., QwenImageTransformer2DModel)
        config: TeaCacheConfig specifying caching parameters

    Example:
        >>> config = TeaCacheConfig(rel_l1_thresh=0.2, model_type="Qwen")
        >>> apply_teacache_hook(transformer, config)
        >>> # Model now uses TeaCache automatically, no code changes needed!
    """
    registry = HookRegistry.get_or_create(module)
    hook = TeaCacheHook(config)
    registry.register_hook(TeaCacheHook._HOOK_NAME, hook)
