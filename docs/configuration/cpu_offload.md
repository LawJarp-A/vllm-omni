# CPU Offloading for Transformer Models

CPU offloading allows you to move model components to CPU memory, enabling inference on hardware with limited GPU memory.

## Configuration

CPU offloading is configured via the following parameters in `OmniModelConfig` or stage config YAML:

### Parameters

- **`cpu_offload_enabled`** (bool, default: `False`): Enable CPU offloading for transformer model components. Default is disabled to minimize latency.

- **`cpu_offload_components`** (list[str] | None, default: `None`): List of component names to offload. Examples:
  - `["thinker", "talker"]` - Offload multi-stage components
  - `["visual", "audio_tower"]` - Offload sub-components
  - `None` - Offload all available components

- **`cpu_offload_strategy`** (str, default: `"alternating"`): Offloading strategy:
  - `"alternating"`: Move component to GPU when needed, offload others (default)
  - `"sequential"`: Load components on demand

- **`cpu_offload_pin_memory`** (bool, default: `True`): Pin CPU memory for faster CPU↔GPU transfers.

## Stage Config YAML Example

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
      cpu_offload_components: ["visual"]  # Offload visual, keep language_model on GPU
```

## CLI Usage

```bash
vllm-omni serve Qwen/Qwen2.5-Omni-7B --omni \
  --cpu-offload-enabled \
  --cpu-offload-components thinker,talker \
  --cpu-offload-strategy alternating
```

## Supported Components

### Multi-Stage Components
- `thinker`: Thinker stage model
- `talker`: Talker stage model
- `code2wav`: Code2Wav stage model
- `token2wav`: Token2Wav stage model

### Sub-Components
- `visual`: Visual encoder
- `audio_tower`: Audio encoder
- `language_model`: Language model

## Behavior

- **Latency-Minimizing Mode** (default): Components stay on GPU after forward pass, only offload when explicitly needed
- **Alternating Strategy**: When component A is used, components B and C are offloaded to CPU
- **Automatic Device Management**: Hooks automatically handle CPU↔GPU transfers without model code changes

## Performance Considerations

### Memory Savings
- **Multi-stage models**: Can reduce peak GPU memory by 30-50% when offloading unused stages
- **Component-level**: Can reduce memory by selectively offloading large encoders (visual, audio_tower)

### Latency Impact
- **Latency-minimizing mode** (default): Minimal latency impact as components stay on GPU after forward
- **Memory-saving mode**: Slight latency increase due to CPU↔GPU transfers, but significant memory reduction
- **Alternating strategy**: Components are moved to GPU before use, ensuring no computation delay

### Trade-offs
- **Memory vs. Latency**: Choose based on your constraints
  - Low memory: Enable CPU offload with memory-saving mode
  - Low latency: Keep CPU offload disabled or use latency-minimizing mode
- **Best Practices**:
  - Offload stages/components that are used infrequently
  - Keep frequently-used components on GPU
  - Use alternating strategy for multi-component models

## Examples

### Example: Multi-Stage Model with Selective Offloading

```yaml
stage_args:
  - stage_id: 0
    engine_args:
      model_stage: thinker
      cpu_offload_enabled: false  # Keep thinker on GPU (used frequently)
  - stage_id: 1
    engine_args:
      model_stage: talker
      cpu_offload_enabled: true
      cpu_offload_components: ["visual"]  # Offload visual encoder only
      cpu_offload_strategy: "alternating"
  - stage_id: 2
    engine_args:
      model_stage: code2wav
      cpu_offload_enabled: true  # Offload code2wav when not in use
```

### Example: Component-Level Offloading

```yaml
stage_args:
  - stage_id: 0
    engine_args:
      model_arch: Qwen2_5OmniForConditionalGeneration
      cpu_offload_enabled: true
      cpu_offload_components: ["visual", "audio_tower"]  # Offload encoders
      # language_model stays on GPU
```

## Notes

- CPU offloading is opt-in (default: disabled) to maintain backward compatibility
- Offloading adds CPU↔GPU transfer overhead, but reduces peak GPU memory usage
- Works with tensor parallelism and pipeline parallelism (offload happens per-rank)
- Hooks automatically manage device transfers - no model code changes required
- Components are moved to GPU before forward pass, ensuring no computation delay

