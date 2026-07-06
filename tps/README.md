# Decode-speed optimizations for Gemma-4-E2B on the Hexagon NPU

Autoregressive decode on the NPU is **KV-attention-bound, not weight-bound** (in the profile, weight matmuls
are only ~6% of decode). These optimizations attack the actual bottleneck. All latencies are measured
on the physical Samsung Galaxy S26 Ultra (Snapdragon 8 Elite Gen 5) Hexagon NPU, A16W4, and are
precision-independent (they apply at any weight bitwidth).

## The ladder (S26 Ultra, 4096-ctx decode, 100% NPU)

| decode graph | latency | tokens/s | speedup |
|---|---|---|---|
| full-KV baseline | 210 ms | 4.8 | 1.0× |
| + windowed-KV | 101 ms | 9.9 | 2.08× |
| + int8-KV (full-attn slots) | 93 ms | 10.8 | 2.27× |
| + broadcast-GQA (removes `expand`) | 51 ms | 19.8 | 4.15× |
| + broadcast-GQA + int8-KV | **44 ms** | **22.7** | **4.77×** |

For reference, Qualcomm's shipped `geniex_llamacpp` q4_0 reports ~18.2 tps @ 4096 on the same NPU.

## What each lever does

**Windowed-KV** — Gemma-4-E2B has hybrid attention: 28 sliding-window layers (window 512) + 7 full-attention.
The naive graph kept a full 4096 KV buffer for every layer and masked out-of-window keys. Instead, sliding
layers use a **512-entry ring buffer** (write at `pos % 512`); only the 7 full-attention layers keep 4096.
Mathematically identical to the masked version (proven == HF greedy, including a 2298-token prompt that wraps
the ring ~4×), but stops computing attention over positions that were being thrown away. 2.08×.

**Broadcast-GQA** — the single biggest lever. Profiling showed the GQA `repeat_kv` **`expand` op was 38% of
decode** (materializing the 1 KV head into 8 copies before attention). A custom `gqa_bcast` attention
interface does the grouped attention with broadcasting matmuls (1 KV head × 8 query heads, never copied),
eliminating the `expand` entirely — verified 0 `Expand` nodes in the exported ONNX. ~2× on top of windowing.

**int8-KV** — stores the KV cache reads at int8 on the full-attention slots (which still attend over 4096),
halving their bandwidth. Modest (+~9%): after windowing + broadcast-GQA, decode is compute-bound, not
KV-read-bound.

## Files
- [`decode_windowed.py`](decode_windowed.py) — windowed-KV decode graph (512 ring buffer for sliding layers); validates == HF greedy, exports ONNX.
- [`decode_windowed_gqa.py`](decode_windowed_gqa.py) — adds the broadcast-GQA attention interface (removes the `expand`); the 22.7-tps graph.
- [`decode_wgqa_prof.py`](decode_wgqa_prof.py) — quantize (A16W4) + compile + profile the graphs on the S26 (the A/B ladder above).
- [`decode_int8kv_profile.py`](decode_int8kv_profile.py) — int8-KV on the full-attention slots, profiled on-device.

## Correctness-harness note
HF `generate` stops on `<eos>` while a raw argmax loop may emit `<end_of_turn>` — compare **content** tokens
(everything before the first special token), not raw sequences, or a correct decode reads as a mismatch.
