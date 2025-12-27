# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for TransformerCPUOffloadBackend.

Tests component detection, hook application, and configuration handling.
"""

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.cpu_offload.backend import TransformerCPUOffloadBackend


class MockModelConfig:
    """Mock model config for testing."""

    def __init__(
        self,
        cpu_offload_enabled: bool = False,
        cpu_offload_components: list[str] | None = None,
        cpu_offload_strategy: str = "alternating",
        cpu_offload_pin_memory: bool = True,
    ):
        self.cpu_offload_enabled = cpu_offload_enabled
        self.cpu_offload_components = cpu_offload_components
        self.cpu_offload_strategy = cpu_offload_strategy
        self.cpu_offload_pin_memory = cpu_offload_pin_memory


class MockModel(nn.Module):
    """Mock model with multiple components for testing."""

    def __init__(self):
        super().__init__()
        self.thinker = nn.Linear(128, 128)
        self.talker = nn.Linear(128, 128)
        self.visual = nn.Linear(128, 128)
        self.audio_tower = nn.Linear(128, 128)


@pytest.fixture
def device():
    """Fixture providing CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def mock_config():
    """Fixture providing mock config with CPU offload disabled."""
    return MockModelConfig(cpu_offload_enabled=False)


@pytest.fixture
def mock_model():
    """Fixture providing mock model with components."""
    return MockModel()


class TestTransformerCPUOffloadBackend:
    """Test suite for TransformerCPUOffloadBackend."""

    def test_backend_initialization(self, mock_config: MockModelConfig, device: torch.device):
        """Test backend initialization."""
        backend = TransformerCPUOffloadBackend(mock_config, device)
        assert backend.config is mock_config
        assert backend.device == device
        assert backend.enabled is False
        assert backend.hooks == {}

    def test_enable_when_disabled(self, mock_config: MockModelConfig, device: torch.device, mock_model: nn.Module):
        """Test that enable does nothing when CPU offload is disabled."""
        backend = TransformerCPUOffloadBackend(mock_config, device)
        backend.enable(mock_model)
        assert backend.enabled is False
        assert len(backend.hooks) == 0

    def test_enable_with_no_components_specified(self, device: torch.device, mock_model: nn.Module):
        """Test enabling with no components specified (should offload all available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = MockModelConfig(cpu_offload_enabled=True, cpu_offload_components=None)
        backend = TransformerCPUOffloadBackend(config, device)
        backend.enable(mock_model)

        # Should find and offload available components
        assert backend.enabled is True
        # Should have found at least thinker and talker
        assert len(backend.hooks) > 0

    def test_enable_with_specific_components(self, device: torch.device, mock_model: nn.Module):
        """Test enabling with specific components."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = MockModelConfig(cpu_offload_enabled=True, cpu_offload_components=["thinker", "visual"])
        backend = TransformerCPUOffloadBackend(config, device)
        backend.enable(mock_model)

        assert backend.enabled is True
        assert "thinker" in backend.hooks
        assert "visual" in backend.hooks
        # talker should not be in hooks since not specified
        assert "talker" not in backend.hooks

    def test_get_component(self, mock_config: MockModelConfig, device: torch.device, mock_model: nn.Module):
        """Test _get_component method."""
        backend = TransformerCPUOffloadBackend(mock_config, device)
        component = backend._get_component(mock_model, "thinker")
        assert component is not None
        assert component is mock_model.thinker

        component = backend._get_component(mock_model, "nonexistent")
        assert component is None

    def test_matches_component_pattern(self, mock_config: MockModelConfig, device: torch.device):
        """Test _matches_component_pattern method."""
        backend = TransformerCPUOffloadBackend(mock_config, device)

        # Exact match
        assert backend._matches_component_pattern("thinker", ["thinker"]) is True
        assert backend._matches_component_pattern("thinker", ["talker"]) is False

        # Wildcard pattern
        assert backend._matches_component_pattern("visual", ["visual.*"]) is True
        assert backend._matches_component_pattern("visual.encoder", ["visual.*"]) is True
        assert backend._matches_component_pattern("audio_tower", ["visual.*"]) is False

    def test_is_enabled(self, mock_config: MockModelConfig, device: torch.device):
        """Test is_enabled method."""
        backend = TransformerCPUOffloadBackend(mock_config, device)
        assert backend.is_enabled() is False

        backend.enabled = True
        assert backend.is_enabled() is True

    def test_ensure_component_on_gpu(self, device: torch.device, mock_model: nn.Module):
        """Test ensure_component_on_gpu method."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = MockModelConfig(cpu_offload_enabled=True, cpu_offload_components=["thinker"])
        backend = TransformerCPUOffloadBackend(config, device)
        backend.enable(mock_model)

        # Component should be on CPU after enable
        assert next(mock_model.thinker.parameters()).device.type == "cpu"

        # Ensure component is on GPU
        backend.ensure_component_on_gpu("thinker")
        assert next(mock_model.thinker.parameters()).device == device

    def test_enable_with_nested_components(self, device: torch.device):
        """Test enabling with nested components (visual inside thinker)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create model with nested structure
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.thinker = nn.Module()
                self.thinker.visual = nn.Linear(128, 128)

        model = NestedModel()
        config = MockModelConfig(cpu_offload_enabled=True, cpu_offload_components=["visual"])
        backend = TransformerCPUOffloadBackend(config, device)
        backend.enable(model)

        # Should find visual inside thinker
        assert "visual" in backend.hooks

