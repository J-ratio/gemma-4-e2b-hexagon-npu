# Baseline: A16W8 decode with **no** inference optimizations

This branch (`baseline-a16w8-unoptimized`) is the deliberate *before* state for demonstrating the
three decode optimizations. It is a **correct, runnable, hardware-verified** A16W8 build that is
simply slow.

Measured on physical Snapdragon 8 Elite (SM8750 / HTP v79), AI Hub profile job:

| | value |
|---|---|
| decode latency | **307.9 ms/step** |
| throughput | **3.25 tok/s** (NPU compute only) |
| KV bytes crossing the graph per step | **~288 MB** |
| on-device output | token-exact vs float — `'The capital of France is **Paris**.'` |

## Why "correct but slow" is the point

The obvious baseline would be `main` minus `tps/`. That doesn't work: `src/decode_pipeline.py`
calibrates activation encodings on `np.random.randn`, which compiles and profiles perfectly and
then emits **pure noise** on device (final hidden norm measured at 0.0000). A speedup measured
against a broken baseline says nothing.

So this branch adds `src/decode_pipeline_baseline.py`, which keeps the naive graph but fixes the
three things that make an A16W8 build actually run:

1. **Real calibration** — activations captured from a working float decode loop, not `randn`.
2. **Chat-format calibration** — the same prompt format used at inference. The raw completion
   format makes this `-it` model degenerate (`' France is France is …'`) *even at fp32*, and
   raw-calibrated + chat-inference yields fluent-but-unfaithful output, which is much harder to
   catch than noise.
3. **`NEG = -1e4`**, not `-inf`/`float32.min`, which cannot survive int16 activation quantization.
   The host must use the identical value.

Both baseline and optimized builds therefore differ **only** in the optimizations — which is what
makes 307.9 ms → 65.3 ms attributable.

## What is deliberately absent

| optimization | absent here | how you can tell |
|---|---|---|
| **Windowed KV** | every layer allocates a full `[1,nkv,4096,hd]` buffer; sliding layers mask out-of-window keys instead of using a ring | `runtime/hostlib.py` has no `KV_BUF`/`WIN`; `run_gate.py` seeds `[CTX]*NC`; both masks are `[1,1,1,CTX]` |
| **Broadcast-GQA** | stock `repeat_kv` materializes 1 KV head into 8 copies | exported graph contains `Expand` nodes (count them) |
| **int8-KV** | all KV tensors stay at int16 | `decode_pipeline_baseline.py` never touches `qc_quantize_op_dict` |

`tps/` has been removed from this branch — it *is* the answer.

## Files

```
src/decode_fixed.py                naive full-KV decode graph export (unchanged from main)
src/decode_pipeline_baseline.py    A16W8 quantization that actually runs  [added]
src/reexport_trunk.py              prefill graph export (unchanged)
runtime/hostlib.py                 host embeddings, chat template, lm_head + softcap
runtime/run_gate.py                host autoregressive loop
runtime/gate_ondevice.sh           on-device qnn-net-run step + KV rotation
runtime/requirements.txt
```

`quant/` is left in place — it is A16W4 *accuracy* research and orthogonal to these three
(precision-independent) speed changes.

## Reproduce the baseline

```bash
# 1. export the float naive decode graph
python src/decode_fixed.py

# 2. quantize + validate; add --compile to build and profile the .bin
python src/decode_pipeline_baseline.py --compile --targets v79
```

Expect phase A `CONTENT MATCH: True` and phase D content-token agreement at/near 100% on the
held-out prompts. Then on a device:

```bash
QAIRT_DIR=/path/to/qairt ./runtime/stage_device.sh <serial>   # or stage manually
python runtime/run_gate.py --prompt "The capital of France is" \
  --ntokens 14 --adb-serial <serial> --chat
```

`--chat` is **required** even on the baseline. Expected: `'The capital of France is **Paris**.'`

## Applying the optimizations

Use the `npu-decode-optimizations` skill. In order (windowing first — it shrinks what the other
two operate on):

1. **Windowed KV** — per-layer ring buffers from `cfg.layer_types`; ring index inside the graph;
   `sliding_mask` narrows to `[1,1,1,WIN]`. Touches the export script, `hostlib.masks`, and
   `run_gate.seed_kv`.
2. **Broadcast-GQA** — register a `gqa_bcast` attention interface; assert 0 `Expand` nodes.
   Touches the export script only.
3. **int8-KV** — `sim.qc_quantize_op_dict[...].set_bitwidth(8)` on the full-attention slots,
   **before** `compute_encodings`. Touches the quantization pipeline only.

Re-verify correctness at each rung, not just at the end — and compare **content tokens**, since a
raw-sequence comparison flags a correct graph as wrong once `<turn|>` and `<eos>` diverge.

Reference end state (`main`, same device, same A16W8 recipe):

| graph | ms/step | tok/s | speedup |
|---|---|---|---|
| this branch (naive full-KV) | 307.9 | 3.25 | 1.0× |
| + windowed KV + broadcast-GQA | 69.8 | 14.3 | 4.41× |
| + int8-KV | 65.3 | 15.3 | 4.71× |

## Caveat on the numbers

These are **NPU compute per step** from AI Hub profile jobs — not end-to-end throughput. Host-side
embedding lookup and `lm_head` are excluded, and the correctness harness reloads the ~1.9 GB
context binary every step, so its wall clock (~3.6 s/token) is not a throughput figure either.
Quote it as "307.9 ms NPU decode compute per token".
