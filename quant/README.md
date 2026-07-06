# A16W4 accuracy recipes — recovering 4-bit weights with AWQ + group-wise scales

Methods and measurements for pushing Gemma-4-E2B toward usable 4-bit weights (int16 activations). All
accuracy numbers here are **weight-only fake-quant simulation on CPU** (fidelity vs the fp32 parent); the
LPBQ sections report **QAIRT compile results** on the S26 Ultra HTP. No end-to-end on-device generation
claims — this is the recipe/exploration layer.

## Winning recipe: AWQ + group-64 + MSE-clip (`awq_g64c`)

Top-1 next-token agreement vs fp32 (teacher-forced), measured across methods:

| method | top-1 agree | note |
|---|---|---|
| RTN per-group-128 (naive A16W4) | 60.9% | baseline |
| RTN per-group-64 | 66.3% | finer groups |
| GPTQ per-group-128 (act_order) | 67.4% | Hessian error-comp |
| AWQ per-group-128 | 68.5% | activation-aware scaling |
| **AWQ + group-64 + MSE-clip** | **79.3%** (KL 0.593) | best |
| rot + AWQ + GPTQ group-64 | 79.3% | ties; needs online rotation matmul |

Also `awq_g64c` → MMLU 0.549 (96% of fp32 0.571). AWQ's per-input-channel scale folds into the preceding
RMSNorm (no extra runtime op).

Findings: stacking matters (AWQ 68.5%@g128 → 79.3%@g64+clip); MSE-clip helps AWQ but hurts GPTQ;
random-orthogonal rotation hurts AWQ/RTN but rescues GPTQ; group-64 is the sweet spot (group-32 was worse).

> Caveat on interpreting these: top-1 agreement / MMLU are **teacher-forced** — they measure per-token
> fidelity against ground-truth context, not free-running generation quality. A given top-1 % does not
> linearly translate to generation coherence.

## Group-wise int4 on the Hexagon NPU: BQ vs LPBQ

Per-block int4 has two encoding forms in QAIRT; only one has an HTP kernel:
- **Raw block-quant (BQ / `PER_BLOCK`)** — no HTP kernel (`QNN_FullyConnected_w_blk_scale: no properties
  registered`), fails to compile.
- **LPBQ** (Low-Power Blockwise: per-channel float scale × per-block int scale) — supported. AIMET emits it
  via `set_lpbq_for_params(sim, bitwidth=4, block_size=64, op_types={"MatMul","Gemm","Conv"})`.

`lpbq_probe.py` confirms LPBQ-64 int4 **compiles on the S26 HTP** (raw BQ does not). `awq_lpbq_deploy.py`
compiles the AWQ-scaled group-64 trunk to a QNN context binary.

### Known issue: LPBQ int4 crashes the HTP compiler on autoregressive *decode* graphs (QAIRT 2.45)

LPBQ-64 int4 compiles fine on the **prefill/trunk** graph (static SDPA), but **`qnn-context-binary-generator`
segfaults (SIGSEGV, exit code -11)** when compiling the **autoregressive decode** graph with the same LPBQ
encoding — targeting Hexagon v81 (Snapdragon 8 Elite Gen 5), QAIRT 2.45.

- The crash is in **context-binary generation**, *after* `qairt-converter` succeeds — i.e. the AIMET LPBQ
  encodings and the ONNX→DLC conversion are valid; the HTP backend dies building the on-device kernels. It is
  a **silent SIGSEGV with no op-level diagnostic** (just "failed with exit code -11").
- **Root cause: the LPBQ 4-bit weight-packing kernels** (`q::pack_4bit_lpbq_weights_2x`,
  `q::pack_4bit_lpbq_scales`, `*_w_scale` op variants) are incompletely registered for the decode graph's op
  mix. Public match: [mllm #678](https://github.com/UbiquitousLearning/mllm/issues/678) —
  `no properties registered for q::GroupedConv2d_w_scale` → `Selecting disabled op
  q::pack_4bit_lpbq_weights_2x` → SIGSEGV on isolated LPBQ-int4 attention projections.
- **Not our graph — verified by elimination.** The segfault persists after: rewriting the KV cache to
  Qualcomm's shipped Concat / new-slice-output pattern (0 `ScatterND` nodes); excluding the dynamic attention
  MatMuls from LPBQ; and restricting LPBQ to MLP linears only. Meanwhile **per-channel int4 compiles and runs
  on the identical decode graph** (it uses HTP's fully-populated per-channel kernels, not the LPBQ packing
  path).
- **Consistent with Qualcomm's own recipes:** `qai-hub-models` ships **per-channel int4** for Llama/Gemma
  decode; LPBQ is opt-in and only wired into the Phi recipe.

**Practical consequence:** on QAIRT 2.45, group-wise/LPBQ int4 (the higher-accuracy recipe) is **prefill-only**;
the decode graph is capped at **per-channel int4**. Deploying group-64 end-to-end on decode needs a newer
QAIRT with broader `pack_4bit_lpbq` kernel coverage — there is no graph-side workaround.

## Files
- [`w4_explore.py`](w4_explore.py) — composable sweep (rotation / AWQ / GPTQ / group-size / MSE-clip).
- [`w4_better.py`](w4_better.py) — AWQ + GPTQ (act_order) implementations, MMLU/GSM8K via lm-eval.
- [`awq_deploy.py`](awq_deploy.py) — folds AWQ per-channel scales into weights (+ 1/s input hook), re-exports.
- [`awq_lpbq_deploy.py`](awq_lpbq_deploy.py) — quantizes the AWQ trunk with LPBQ-64 and compiles for HTP.
- [`lpbq_probe.py`](lpbq_probe.py) — proves LPBQ block-int4 compiles on HTP (raw BQ fails).
- [`layer_sensitivity.py`](layer_sensitivity.py) — per-layer / per-projection W4 sensitivity map.
- [`agreement.py`](agreement.py) — top-1 agreement + KL fidelity metric.
- [`lmeval_quant.py`](lmeval_quant.py) — MMLU + GSM8K across fp32 / W8 / W4.
- [`run_awq_lmeval.py`](run_awq_lmeval.py) — **self-contained**: quantize to `awq_g64c` and run MMLU+GSM8K via lm-eval on CPU (reproduces [`RESULTS.md`](RESULTS.md)).

## Notes
- Measure fidelity with **top-1 agreement + KL**, not perplexity — Gemma-4-E2B-it on raw text gives
  confounded PPL (one low-prob token dominates the geometric mean).
- Gemma needs `tokenizer.add_bos_token=True` in lm-eval or loglikelihood-MC tasks score at random.
