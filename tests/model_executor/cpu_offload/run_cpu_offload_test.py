#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Standalone script to test CPU offloading with Qwen2.5-Omni model.

Usage:
    # Set model cache to /mnt/nvme
    export HF_HOME=/mnt/nvme/.cache/huggingface
    export HF_HUB_CACHE=/mnt/nvme/.cache/huggingface/hub
    export TRANSFORMERS_CACHE=/mnt/nvme/.cache/huggingface/transformers

    # Run the test
    python tests/model_executor/cpu_offload/run_cpu_offload_test.py
"""

import os
import sys
from pathlib import Path

# Set model cache to /mnt/nvme to avoid space issues
os.environ.setdefault("HF_HOME", "/mnt/nvme/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/nvme/.cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/nvme/.cache/huggingface/transformers")

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ruff: noqa: E402
import torch

from vllm_omni import Omni


def main():
    """Run CPU offloading test with Qwen2.5-Omni."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires a GPU.")
        return 1

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
        print(f"Stage config: {stage_config_path}")

        # Initialize Omni with CPU offload enabled
        print("\nInitializing model with CPU offload enabled on talker stage...")
        omni = Omni(
            model=model_name,
            stage_configs_path=str(stage_config_path),
            init_sleep_seconds=30,
        )

        # Test with simple text prompt
        print("\nRunning inference with CPU offload enabled...")
        from vllm.sampling_params import SamplingParams

        # Create sampling params for each stage
        sampling_params_list = [
            SamplingParams(temperature=0.0, max_tokens=100),  # Stage 0 (thinker)
            SamplingParams(temperature=0.9, max_tokens=100),  # Stage 1 (talker)
        ]
        outputs = omni.generate(
            prompts=["Hello, how are you?"],
            sampling_params_list=sampling_params_list,
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

        # Cleanup
        omni.close()
        return 0

    except Exception as e:
        print(f"\n❌ ERROR: Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Clean up temporary config
        if stage_config_path.exists():
            stage_config_path.unlink()


if __name__ == "__main__":
    sys.exit(main())

