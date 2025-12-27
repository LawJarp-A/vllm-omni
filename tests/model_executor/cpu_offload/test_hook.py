# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for TransformerCPUOffloadHook.

Tests hook registration, device transfer logic, and alternating offload strategy.
"""

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.hooks import HookRegistry
from vllm_omni.model_executor.cpu_offload.hook import (
    TransformerCPUOffloadHook,
    apply_transformer_cpu_offload_hook,
)


class SimpleModule(nn.Module):
    """Simple test module for CPU offload testing."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture
def device():
    """Fixture providing CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def simple_module():
    """Fixture providing a simple test module."""
    return SimpleModule()


class TestTransformerCPUOffloadHook:
    """Test suite for TransformerCPUOffloadHook."""

    def test_hook_initialization(self, device: torch.device):
        """Test hook initialization with default parameters."""
        hook = TransformerCPUOffloadHook(execution_device=device)
        assert hook.execution_device == device
        assert hook.other_hooks == []
        assert hook.keep_on_gpu is True
        assert hook._module is None

    def test_hook_initialization_with_other_hooks(self, device: torch.device):
        """Test hook initialization with other hooks for coordination."""
        hook1 = TransformerCPUOffloadHook(execution_device=device)
        hook2 = TransformerCPUOffloadHook(execution_device=device)
        hook3 = TransformerCPUOffloadHook(execution_device=device, other_hooks=[hook1, hook2])
        assert len(hook3.other_hooks) == 2
        assert hook1 in hook3.other_hooks
        assert hook2 in hook3.other_hooks

    def test_initialize_hook_moves_to_cpu(self, simple_module: nn.Module):
        """Test that initialize_hook moves module to CPU."""
        # Move module to GPU first
        if torch.cuda.is_available():
            simple_module.to("cuda:0")
            assert next(simple_module.parameters()).device.type == "cuda"

        hook = TransformerCPUOffloadHook(execution_device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        result = hook.initialize_hook(simple_module)
        assert result is simple_module
        assert next(simple_module.parameters()).device.type == "cpu"
        assert hook._module is simple_module

    def test_new_forward_moves_to_gpu(self, simple_module: nn.Module, device: torch.device):
        """Test that new_forward moves module to GPU before forward pass."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Use apply_transformer_cpu_offload_hook to ensure registry is created
        apply_transformer_cpu_offload_hook(simple_module, device)

        # Module should be on CPU after initialization
        assert next(simple_module.parameters()).device.type == "cpu"

        # Create dummy input
        x = torch.randn(2, 128).to(device)

        # Forward should move to GPU (via the hook system)
        result = simple_module(x)
        assert next(simple_module.parameters()).device == device
        assert result.shape == (2, 128)

    def test_alternating_offload_strategy(self, device: torch.device):
        """Test alternating offload strategy with multiple hooks."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        module1 = SimpleModule()
        module2 = SimpleModule()
        module3 = SimpleModule()

        # Create hooks with coordination
        hook1 = apply_transformer_cpu_offload_hook(module1, device)
        hook2 = apply_transformer_cpu_offload_hook(module2, device, other_hooks=[hook1])
        hook3 = apply_transformer_cpu_offload_hook(module3, device, other_hooks=[hook1, hook2])

        # Set up coordination for all hooks
        hook1.other_hooks = [hook2, hook3]
        hook2.other_hooks = [hook1, hook3]
        hook3.other_hooks = [hook1, hook2]

        x = torch.randn(2, 128).to(device)

        # When module1 is used, module2 and module3 should be offloaded
        _ = module1(x)
        assert next(module1.parameters()).device == device
        # hook2 and hook3 should remain on CPU (or be moved there)
        assert next(module2.parameters()).device.type == "cpu"
        assert next(module3.parameters()).device.type == "cpu"

    def test_latency_minimizing_mode(self, simple_module: nn.Module, device: torch.device):
        """Test that latency-minimizing mode keeps module on GPU after forward."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        apply_transformer_cpu_offload_hook(simple_module, device, keep_on_gpu=True)

        x = torch.randn(2, 128).to(device)
        _ = simple_module(x)

        # Module should still be on GPU after forward (latency-minimizing)
        assert next(simple_module.parameters()).device == device

    def test_memory_saving_mode(self, simple_module: nn.Module, device: torch.device):
        """Test that memory-saving mode offloads module after forward."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        apply_transformer_cpu_offload_hook(simple_module, device, keep_on_gpu=False)

        x = torch.randn(2, 128).to(device)
        _ = simple_module(x)

        # Module should be offloaded to CPU after forward (memory-saving)
        assert next(simple_module.parameters()).device.type == "cpu"

    def test_offload_to_cpu(self, simple_module: nn.Module, device: torch.device):
        """Test explicit offload_to_cpu method."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        hook = TransformerCPUOffloadHook(execution_device=device)
        simple_module.to(device)
        assert next(simple_module.parameters()).device == device

        hook.offload_to_cpu(simple_module)
        assert next(simple_module.parameters()).device.type == "cpu"

    def test_reset_state(self, simple_module: nn.Module):
        """Test reset_state method (should be no-op)."""
        hook = TransformerCPUOffloadHook(execution_device=torch.device("cpu"))
        result = hook.reset_state(simple_module)
        assert result is simple_module


class TestApplyTransformerCPUOffloadHook:
    """Test suite for apply_transformer_cpu_offload_hook function."""

    def test_apply_hook_registers_with_registry(self, simple_module: nn.Module, device: torch.device):
        """Test that apply_transformer_cpu_offload_hook registers hook with registry."""
        hook = apply_transformer_cpu_offload_hook(simple_module, device)

        registry = HookRegistry.get_or_create(simple_module)
        # Check that hook is registered using get_hook (registry doesn't have has_hook)
        registered_hook = registry.get_hook(TransformerCPUOffloadHook._HOOK_NAME)
        assert registered_hook is not None
        assert registered_hook is hook
        assert isinstance(hook, TransformerCPUOffloadHook)

    def test_apply_hook_with_other_hooks(self, device: torch.device):
        """Test applying hook with coordination to other hooks."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        module1 = SimpleModule()
        module2 = SimpleModule()

        hook1 = apply_transformer_cpu_offload_hook(module1, device)
        hook2 = apply_transformer_cpu_offload_hook(module2, device, other_hooks=[hook1])

        assert hook2.other_hooks == [hook1]

    def test_apply_hook_initializes_module(self, simple_module: nn.Module, device: torch.device):
        """Test that applying hook initializes module (moves to CPU)."""
        if torch.cuda.is_available():
            simple_module.to(device)
            assert next(simple_module.parameters()).device == device

        apply_transformer_cpu_offload_hook(simple_module, device)
        # Module should be on CPU after hook initialization
        assert next(simple_module.parameters()).device.type == "cpu"

