# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Integration test for CPU offloading with Qwen2.5-Omni model.

This test verifies that CPU offloading works correctly with a real multi-stage model.
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
from tests.e2e.offline_inference.conftest import OmniRunner
from tests.e2e.offline_inference.utils import create_new_process_for_each_test


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.core_model
@create_new_process_for_each_test()
def test_cpu_offload_with_qwen25_omni(omni_runner: type[OmniRunner]) -> None:
    """Test CPU offloading with Qwen2.5-Omni multi-stage model."""
    model_name = "Qwen/Qwen2.5-Omni-3B"

    # Create stage config with CPU offload enabled for talker stage
    stage_config_content = """
stage_args:
  - stage_id: 0
    runtime:
      devices: "0"
      max_batch_size: 1
    engine_args:
      model_stage: thinker
      model_arch: Qwen2_5OmniForConditionalGeneration
      worker_cls: vllm_omni.worker.gpu_ar_worker.GPUARWorker
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      cpu_offload_enabled: false  # Keep thinker on GPU
      gpu_memory_utilization: 0.6
      enforce_eager: true
      trust_remote_code: true
      engine_output_type: latent
      enable_prefix_caching: false
      max_num_batched_tokens: 32768
    is_comprehension: true
    final_output: true
    final_output_type: text
    default_sampling_params:
      temperature: 0.0
      top_p: 1.0
      top_k: -1
      max_tokens: 100
      seed: 42
      detokenize: True
      repetition_penalty: 1.1
  - stage_id: 1
    runtime:
      devices: "0"
      max_batch_size: 1
    engine_args:
      model_stage: talker
      model_arch: Qwen2_5OmniForConditionalGeneration
      worker_cls: vllm_omni.worker.gpu_ar_worker.GPUARWorker
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      cpu_offload_enabled: true  # Enable CPU offload for talker
      cpu_offload_components: ["visual"]  # Offload visual encoder
      cpu_offload_strategy: "alternating"
      gpu_memory_utilization: 0.3
      enforce_eager: true
      trust_remote_code: true
      enable_prefix_caching: false
      max_num_batched_tokens: 32768
      engine_output_type: latent
    engine_input_source: [0]
    custom_process_input_func: vllm_omni.model_executor.stage_input_processors.qwen2_5_omni.thinker2talker
    default_sampling_params:
      temperature: 0.9
      top_p: 0.8
      top_k: 40
      max_tokens: 100
      seed: 42
      detokenize: False
      repetition_penalty: 1.05
      stop_token_ids: [8294]

runtime:
  enabled: true
  defaults:
    window_size: -1
    max_inflight: 1
  edges:
    - from: 0
      to: 1
      window_size: -1
"""

    # Write temporary stage config
    stage_config_path = Path("/tmp/test_cpu_offload_qwen25_omni.yaml")
    stage_config_path.write_text(stage_config_content)

    try:
        print(f"\n{'='*60}")
        print(f"Testing CPU offloading with {model_name}")
        print(f"{'='*60}")
        print(f"Model cache: {os.environ.get('HF_HOME', 'default')}")

        with omni_runner(
            model_name,
            seed=42,
            stage_configs_path=str(stage_config_path),
            init_sleep_seconds=30,
        ) as runner:
            # Test with simple text prompt
            print("\nRunning inference with CPU offload enabled on talker stage...")
            outputs = runner.generate_multimodal(
                prompts="Hello, how are you?",
            )

            # Verify output
            assert len(outputs) > 0
            text_output = None
            for stage_output in outputs:
                if stage_output.final_output_type == "text":
                    text_output = stage_output
                    break

            assert text_output is not None
            assert len(text_output.request_output) > 0
            text_content = text_output.request_output[0].outputs[0].text
            assert text_content is not None
            assert len(text_content.strip()) > 0

            print(f"\n✅ SUCCESS! Generated text: {text_content[:100]}...")
            print(f"\n{'='*60}")
            print("CPU offloading integration test passed!")
            print(f"{'='*60}")

    finally:
        # Clean up temporary config
        if stage_config_path.exists():
            stage_config_path.unlink()

