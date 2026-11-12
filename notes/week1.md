# Week 1: Environment Setup & Baseline Benchmark

## Hardware & Environment
- **Model:** `openai/whisper-small`
- **Dataset:** LibriSpeech `test-clean` (all 2,620 utterances)
- **Precision Variants:** `fp32`, `fp16`
- **Setup Notes:** Enabled read access to `/sys/class/powercap/intel-rapl:0/energy_uj` for CPU RAPL energy telemetry alongside NVML GPU power sampling.

## Baseline Results (n = 2,620)

| Variant | WER (%) | p50 (ms) | p95 (ms) | RTFx | GPU J/min | CPU J/min | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fp32` | 3.45 | 314.5 | 609.5 | 21.2 | 117.46 | 46.04 | 1053 |
| `fp16` | 3.45 | 157.5 | 350.9 | 41.1 | 53.11 | 28.14 | 546 |

## Key Takeaways
- **Accuracy:** Both `fp32` and `fp16` achieved **3.45% WER**, matching the published Whisper `small` benchmark (~3.4%) with zero precision loss.
- **Speed & Efficiency:** `fp16` nearly doubled throughput (**41.1× RTFx** vs. **21.2×**), halved median latency (**157.5 ms**), cut total system energy by **50.3%** (**81.25 J/min** vs. **163.50 J/min**), and reduced peak VRAM by **48.1%** (**546 MB**).
- Baseline verified and tagged as `v0.1-baseline`.
