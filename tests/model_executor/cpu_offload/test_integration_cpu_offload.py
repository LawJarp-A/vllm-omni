# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Integration test for CPU offloading with actual Qwen image model.

This test verifies that CPU offloading works correctly with a real model.
Models are downloaded to /mnt/nvme to avoid space issues on /mnt.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

# Set model cache to /mnt/nvme to avoid space issues
os.environ["HF_HOME"] = "/mnt/nvme/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/mnt/nvme/.cache/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/nvme/.cache/huggingface/transformers"

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ruff: noqa: E402
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

from tests.e2e.offline_inference.conftest import OmniRunner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_offload_with_qwen_image():
    """Test CPU offloading with Qwen2-VL image model."""
    model_name = "Qwen/Qwen2-VL-2B-Instruct"

    # Create stage config with CPU offload enabled
    stage_config_content = """
stage_args:
  - stage_id: 0
    runtime:
      devices: "0"
      max_batch_size: 1
    engine_args:
      model_arch: Qwen2VLForConditionalGeneration
      cpu_offload_enabled: true
      cpu_offload_components: ["visual"]  # Offload visual encoder
      cpu_offload_strategy: "alternating"
      gpu_memory_utilization: 0.5
      enforce_eager: true
      trust_remote_code: true
      max_model_len: 2048
    final_output: true
    final_output_type: text
    is_comprehension: true
    default_sampling_params:
      temperature: 0.7
      top_p: 0.9
      max_tokens: 100
      seed: 42
      detokenize: True
"""

    # Write temporary stage config
    stage_config_path = Path("/tmp/test_cpu_offload_stage_config.yaml")
    stage_config_path.write_text(stage_config_content)

    try:
        # Initialize OmniRunner with CPU offload enabled
        print(f"\n{'='*60}")
        print(f"Testing CPU offloading with {model_name}")
        print(f"{'='*60}")
        print(f"Model cache: {os.environ.get('HF_HOME', 'default')}")

        with OmniRunner(
            model_name,
            seed=42,
            stage_configs_path=str(stage_config_path),
            init_sleep_seconds=30,
        ) as runner:
            # Prepare image input
            image_asset = ImageAsset("cherry_blossom")
            image = convert_image_mode(image_asset.pil_image.resize((224, 224)), "RGB")

            # Test inference with CPU offload
            print("\nRunning inference with CPU offload enabled...")
            outputs = runner.generate_multimodal(
                prompts="What is in this image?",
                images=image,
            )

            # Verify output
            assert len(outputs) > 0
            assert outputs[0].final_output_type == "text"
            assert len(outputs[0].request_output) > 0
            text_output = outputs[0].request_output[0].outputs[0].text
            assert text_output is not None
            assert len(text_output.strip()) > 0

            print(f"\n✅ SUCCESS! Generated text: {text_output[:100]}...")
            print(f"\n{'='*60}")
            print("CPU offloading integration test passed!")
            print(f"{'='*60}")

    finally:
        # Clean up temporary config
        if stage_config_path.exists():
            stage_config_path.unlink()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_offload_disabled_comparison():
    """Test that model works correctly with CPU offload disabled."""
    model_name = "Qwen/Qwen2-VL-2B-Instruct"

    # Create stage config with CPU offload disabled
    stage_config_content = """
stage_args:
  - stage_id: 0
    runtime:
      devices: "0"
      max_batch_size: 1
    engine_args:
      model_arch: Qwen2VLForConditionalGeneration
      cpu_offload_enabled: false  # CPU offload disabled
      gpu_memory_utilization: 0.5
      enforce_eager: true
      trust_remote_code: true
      max_model_len: 2048
    final_output: true
    final_output_type: text
    is_comprehension: true
    default_sampling_params:
      temperature: 0.7
      top_p: 0.9
      max_tokens: 100
      seed: 42
      detokenize: True
"""

    # Write temporary stage config
    stage_config_path = Path("/tmp/test_cpu_offload_disabled_stage_config.yaml")
    stage_config_path.write_text(stage_config_content)

    try:
        print(f"\n{'='*60}")
        print(f"Testing baseline (CPU offload disabled) with {model_name}")
        print(f"{'='*60}")

        with OmniRunner(
            model_name,
            seed=42,
            stage_configs_path=str(stage_config_path),
            init_sleep_seconds=30,
        ) as runner:
            # Prepare image input
            image_asset = ImageAsset("cherry_blossom")
            image = convert_image_mode(image_asset.pil_image.resize((224, 224)), "RGB")

            # Test inference without CPU offload
            print("\nRunning inference with CPU offload disabled...")
            outputs = runner.generate_multimodal(
                prompts="What is in this image?",
                images=image,
            )

            # Verify output
            assert len(outputs) > 0
            assert outputs[0].final_output_type == "text"
            assert len(outputs[0].request_output) > 0
            text_output = outputs[0].request_output[0].outputs[0].text
            assert text_output is not None
            assert len(text_output.strip()) > 0

            print(f"\n✅ SUCCESS! Generated text: {text_output[:100]}...")
            print(f"\n{'='*60}")
            print("Baseline test (CPU offload disabled) passed!")
            print(f"{'='*60}")

    finally:
        # Clean up temporary config
        if stage_config_path.exists():
            stage_config_path.unlink()

