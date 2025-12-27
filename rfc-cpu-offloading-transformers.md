# RFC: CPU Offloading for Transformer Models in vLLM-Omni

## Summary

This RFC proposes adding CPU offloading support for transformer/autoregressive (AR) models in vLLM-Omni, leveraging the existing hook-based architecture already used for diffusion models. This feature will enable memory-efficient inference for large transformer models without requiring modifications to model code.

## Motivation

### Problem Statement

Large transformer models (e.g., Qwen3-Omni-MoE, Qwen2.5-Omni) can consume significant GPU memory, making it challenging to run them on hardware with limited GPU memory. While vLLM-Omni already supports CPU offloading for diffusion models, transformer/AR models currently lack this capability.

### Use Cases

1. **Memory-Constrained Environments**: Users with limited GPU memory can offload model components to CPU, enabling inference on smaller GPUs
2. **Multi-Stage Models**: For multi-stage models (thinker, talker, code2wav), components can be offloaded when not actively used, reducing peak memory usage
3. **Component-Level Optimization**: Users can selectively offload specific components (e.g., visual encoder, audio tower) while keeping frequently-used components on GPU

### Current State

- ✅ CPU offloading exists for diffusion models (`vllm_omni/diffusion/cpu_offload/`)
- ✅ Hook-based architecture is implemented and working
- ❌ No CPU offloading support for transformer/AR models
- ❌ No configuration flags for transformer CPU offloading

## Proposed Change

### Design: Hook-Based Approach

We propose extending the existing hook-based CPU offloading system to support transformer models. This approach:

- **Reuses existing infrastructure**: Leverages `HookRegistry` and `ModelHook` from diffusion CPU offloading
- **Zero model code changes**: Works with any `torch.nn.Module` without modifications
- **Automatic device management**: Hooks automatically handle CPU↔GPU transfers
- **Consistent with existing patterns**: Follows the same architecture as diffusion CPU offloading

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Model Runner                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         TransformerCPUOffloadBackend              │  │
│  │  - Applies hooks to model components              │  │
│  │  - Coordinates alternating offload strategy      │  │
│  └──────────────────────────────────────────────────┘  │
│                          │                               │
│                          ▼                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │         TransformerCPUOffloadHook                │  │
│  │  - Moves module to GPU before forward()          │  │
│  │  - Coordinates with other hooks                   │  │
│  │  - Keeps on GPU (latency-minimizing mode)        │  │
│  └──────────────────────────────────────────────────┘  │
│                          │                               │
│                          ▼                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │              HookRegistry                         │  │
│  │  - Manages hooks attached to modules             │  │
│  │  - Intercepts forward() calls                    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Key Components

1. **`TransformerCPUOffloadHook`**: Hook implementation that manages device transfers
   - Moves modules to GPU before forward pass
   - Coordinates with other hooks for alternating strategy
   - Implements latency-minimizing mode (keeps on GPU after forward)

2. **`TransformerCPUOffloadBackend`**: Backend that applies hooks to model components
   - Detects model components (thinker, talker, code2wav, visual, audio_tower)
   - Applies hooks based on configuration
   - Coordinates hooks for alternating offload strategy

3. **Configuration Flags**: New flags in `OmniModelConfig`
   - `cpu_offload_enabled: bool = False` (default: disabled to minimize latency)
   - `cpu_offload_components: list[str] | None = None`
   - `cpu_offload_strategy: str = "alternating"`
   - `cpu_offload_pin_memory: bool = True`

### Hook Architecture Analysis

**Current Hook System**:
- `HookRegistry.dispatch()` supports single hook per module
- CPU offloading coordination happens via `other_hooks` list in `CPUOffloadHook`
- **Hook chaining is NOT needed** for CPU offloading (coordination happens within hook)

**Decision**: No changes to hook architecture required. Current single-hook-per-module design is sufficient.

### Implementation Details

#### 1. Module Structure

```
vllm_omni/model_executor/cpu_offload/
├── __init__.py
├── hook.py              # TransformerCPUOffloadHook
└── backend.py           # TransformerCPUOffloadBackend
```

#### 2. Integration Points

- **Model Runners**: Apply hooks in `GPUARModelRunner.__init__()` and `GPUGenerationModelRunner.__init__()`
- **Workers**: Pass configuration from `GPUARWorker` and `GPUGenerationWorker` to model runners
- **Configuration**: Add flags to `OmniModelConfig` in `vllm_omni/config/model.py`

#### 3. Offloading Strategy

**Alternating Strategy** (default):
- When component A is used, offload components B, C to CPU
- Move component A to GPU before forward pass
- After forward pass, keep on GPU (latency-minimizing mode)

**Latency-Minimizing Mode** (default behavior):
- Components stay on GPU after forward pass
- Only offload when explicitly needed or memory pressure detected
- Default: `cpu_offload_enabled = False` (disabled to minimize latency)

#### 4. Component Detection

The backend will automatically detect and support:
- **Multi-stage components**: thinker, talker, code2wav
- **Sub-components**: visual, audio_tower, language_model
- **Pattern matching**: Support component name patterns (e.g., "visual.*")

### Configuration Examples

#### Example 1: Single-Stage Model with Component Offloading

```yaml
stage_args:
  - stage_id: 0
    engine_args:
      model_arch: Qwen2_5OmniForConditionalGeneration
      cpu_offload_enabled: true
      cpu_offload_components: ["visual", "audio_tower"]
      cpu_offload_strategy: "alternating"
```

#### Example 2: Multi-Stage Model with Per-Stage Offloading

```yaml
stage_args:
  - stage_id: 0
    engine_args:
      model_stage: thinker
      cpu_offload_enabled: false  # Keep thinker on GPU
  - stage_id: 1
    engine_args:
      model_stage: talker
      cpu_offload_enabled: true
      cpu_offload_components: ["visual"]
```

#### Example 3: CLI Usage

```bash
vllm-omni --model qwen2.5-omni \
  --cpu-offload-enabled \
  --cpu-offload-components thinker,talker \
  --cpu-offload-strategy alternating
```

## Alternatives Considered

### Option 1: Hook-Based Approach ✅ (Selected)

**Pros**:
- Reuses existing infrastructure
- Zero changes to model code
- Automatic device management
- Consistent with diffusion CPU offloading

**Cons**:
- Requires hook registration point
- May need coordination for multi-stage models

### Option 2: Model Runner Wrapper

**Pros**:
- Centralized control point
- Easy to configure per-stage

**Cons**:
- Requires changes to model runner code
- Less flexible than hooks
- May need modifications for each model type

### Option 3: Weight Loading Hook

**Pros**:
- Minimal changes
- Works at initialization time

**Cons**:
- Less dynamic control
- Doesn't handle runtime offloading

**Decision**: Option 1 (hook-based) selected for consistency, flexibility, and zero model code changes.

## Implementation Plan

### Phase 1: Foundation (Week 1)
- Create `vllm_omni/model_executor/cpu_offload/` module
- Implement `TransformerCPUOffloadHook` and `TransformerCPUOffloadBackend`
- Add configuration flags to `OmniModelConfig`
- Analyze hook architecture (confirmed: no changes needed)

### Phase 2: Model Runner Integration (Week 2)
- Integrate hooks in `GPUARModelRunner` and `GPUGenerationModelRunner`
- Update workers to pass configuration
- Implement alternating offload strategy

### Phase 3: Multi-Stage Support (Week 3)
- Add per-stage CPU offload configuration
- Support component-level offloading
- Ensure cross-stage coordination

### Phase 4: Testing & Documentation (Week 4)
- Write unit and integration tests
- Create performance benchmarks
- Write user documentation and examples

**Total Estimated Time**: 4 weeks for MVP implementation

## Open Questions

1. **Memory Pressure Detection**: Should we implement automatic memory pressure detection to trigger offloading, or rely on explicit configuration?
   - **Proposed**: Start with explicit configuration, add automatic detection in future if needed

2. **Offloading Granularity**: Should we support per-layer offloading in addition to per-component?
   - **Proposed**: Start with per-component, add per-layer in future if needed

3. **Performance Impact**: What is the acceptable latency overhead for CPU offloading?
   - **Proposed**: With latency-minimizing mode (default), overhead should be minimal. Benchmark and document trade-offs.

4. **Integration with Other Features**: How should CPU offloading interact with tensor parallelism, pipeline parallelism, and quantization?
   - **Proposed**: CPU offload should work with TP/PP (offload happens per-rank). Quantization should work transparently.

## Success Criteria

1. ✅ CPU offloading works for transformer models without modifying model code
2. ✅ New models automatically support CPU offloading via hooks
3. ✅ Minimal latency impact (default: keep on GPU, only offload when needed)
4. ✅ Demonstrable memory reduction when enabled (target: 30-50% for multi-stage models)
5. ✅ All existing tests pass, new tests added
6. ✅ Complete user documentation and examples
7. ✅ Code passes all linting and style checks

## Backward Compatibility

- **Default behavior**: CPU offloading is **disabled by default** (`cpu_offload_enabled = False`)
- **No breaking changes**: Existing code continues to work without modifications
- **Opt-in feature**: Users must explicitly enable CPU offloading

## Testing Strategy

### Unit Tests
- Hook registration and removal
- Device transfer logic
- Alternating offload strategy
- Edge cases (empty components, all offloaded, etc.)

### Integration Tests
- Qwen2.5-Omni (single-stage)
- Qwen3-Omni-MoE (multi-stage)
- Memory usage verification
- Inference correctness
- Performance benchmarks

## Documentation

- User guide section on CPU offloading
- Configuration options documentation
- Examples in `examples/` directory
- API documentation updates
- Hook architecture decisions documented

## Risks and Mitigations

### Risk 1: Performance Overhead
- **Mitigation**: Default to latency-minimizing mode (keep on GPU), only offload when explicitly needed

### Risk 2: Complexity in Multi-Stage Models
- **Mitigation**: Start with simple cases, add multi-stage support incrementally

### Risk 3: Hook Architecture Limitations
- **Mitigation**: Analyzed and confirmed current architecture is sufficient (no hook chaining needed)

## Timeline

- **Week 1**: Foundation (hook implementation, config support)
- **Week 2**: Model runner integration
- **Week 3**: Multi-stage support
- **Week 4**: Testing & documentation

**Target Completion**: 4 weeks from start

## Feedback Period

**Requested Feedback Period**: 1 week

Please provide feedback on:
- Design approach (hook-based)
- Configuration interface
- Default behavior (latency-minimizing)
- Implementation timeline

## CC List

- vLLM-Omni maintainers
- Contributors working on memory optimization
- Users interested in running large models on memory-constrained hardware

## References

- Existing CPU offloading implementation: `vllm_omni/diffusion/cpu_offload/`
- Hook architecture: `vllm_omni/diffusion/hooks.py`
- Model configuration: `vllm_omni/config/model.py`
- Detailed implementation plan: `plan.md`

---

**Status**: Draft - Ready for Review

