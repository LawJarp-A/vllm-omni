# CPU Offload Tests

This directory contains tests for the CPU offloading feature for transformer models.

## Test Files

- **`test_hook.py`**: Unit tests for `TransformerCPUOffloadHook`
  - Tests hook registration and removal
  - Tests device transfer logic (CPU ↔ GPU)
  - Tests alternating offload strategy
  - Tests latency-minimizing mode

- **`test_backend.py`**: Unit tests for `TransformerCPUOffloadBackend`
  - Tests component detection and matching
  - Tests hook application with specific components
  - Tests pattern matching (wildcards, exact matches)
  - Tests `ensure_component_on_gpu` method

- **`test_cpu_offload_qwen25_omni.py`**: Integration test with Qwen2.5-Omni
  - Tests CPU offloading with a real multi-stage model
  - Uses `OmniRunner` test infrastructure
  - Tests CPU offload on talker stage with visual encoder offloading

- **`test_integration_cpu_offload.py`**: Integration test with Qwen2-VL
  - Tests CPU offloading with Qwen2-VL image model
  - Includes baseline comparison test (CPU offload disabled)

- **`run_cpu_offload_test.py`**: Standalone test script
  - Can be run directly without pytest
  - Automatically sets model cache to `/mnt/nvme`

## Prerequisites

1. **vllm must be installed and compiled**
   - vllm requires C extensions (`vllm._C`) to be built
   - Install vllm in development mode: `cd /path/to/vllm && pip install -e .`
   - Or use a pre-built vllm package

2. **vllm-omni must be installed**
   - Install in development mode: `cd /path/to/vllm-omni && pip install -e .`

3. **Model cache configuration**
   - Models should be downloaded to `/mnt/nvme` (not `/mnt`) due to space constraints
   - Set environment variables:
     ```bash
     export HF_HOME=/mnt/nvme/.cache/huggingface
     export HF_HUB_CACHE=/mnt/nvme/.cache/huggingface/hub
     export TRANSFORMERS_CACHE=/mnt/nvme/.cache/huggingface/transformers
     ```

4. **CUDA availability**
   - Tests require CUDA-capable GPU
   - Tests will skip if CUDA is not available

## Running Tests

### Unit Tests (No Model Download Required)

```bash
# Run all unit tests
pytest tests/model_executor/cpu_offload/test_hook.py tests/model_executor/cpu_offload/test_backend.py -v

# Run specific test
pytest tests/model_executor/cpu_offload/test_hook.py::test_hook_registration -v
```

### Integration Tests (Requires Model Download)

```bash
# Set model cache
export HF_HOME=/mnt/nvme/.cache/huggingface
export HF_HUB_CACHE=/mnt/nvme/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/mnt/nvme/.cache/huggingface/transformers

# Run integration test with Qwen2.5-Omni
pytest tests/model_executor/cpu_offload/test_cpu_offload_qwen25_omni.py -v -s

# Or use the standalone script
python tests/model_executor/cpu_offload/run_cpu_offload_test.py
```

## Test Status

✅ **Unit tests**: Written and ready to run (require vllm installation)
✅ **Integration tests**: Written and ready to run (require vllm installation and model download)
✅ **Code quality**: All tests pass `ruff lint`

## Troubleshooting

### `ModuleNotFoundError: No module named 'vllm'`
- Install vllm: `cd /path/to/vllm && pip install -e .`
- Ensure vllm is in your Python path

### `ModuleNotFoundError: No module named 'vllm._C'`
- vllm C extensions need to be compiled
- Reinstall vllm: `cd /path/to/vllm && pip install -e . --force-reinstall`
- Ensure CUDA toolkit is installed

### `CUDA not available`
- Tests will automatically skip if CUDA is not available
- Ensure CUDA-capable GPU is present and drivers are installed

### Model download fails
- Check that `/mnt/nvme` has sufficient space
- Verify environment variables are set correctly
- Check network connectivity for HuggingFace Hub

## Notes

- All tests are designed to work with the existing vllm-omni test infrastructure
- Integration tests use real models and will download them on first run
- Model cache is set to `/mnt/nvme` to avoid space issues on `/mnt`
- Tests follow pytest conventions and can be run individually or as a suite

