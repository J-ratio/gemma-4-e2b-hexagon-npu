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

## Files
- [`w4_explore.py`](w4_explore.py) — composable sweep (rotation / AWQ / GPTQ / group-size / MSE-clip).
- [`w4_better.py`](w4_better.py) — AWQ + GPTQ (act_order) implementations, MMLU/GSM8K via lm-eval.
- [`awq_deploy.py`](awq_deploy.py) — folds AWQ per-channel scales into weights (+ 1/s input hook), re-exports.
- [`awq_lpbq_deploy.py`](awq_lpbq_deploy.py) — quantizes the AWQ trunk with LPBQ-64 and compiles for HTP.
- [`lpbq_probe.py`](lpbq_probe.py) — proves LPBQ block-int4 compiles on HTP (raw BQ fails).
- [`layer_sensitivity.py`](layer_sensitivity.py) — per-layer / per-projection W4 sensitivity map.
- [`agreement.py`](agreement.py) — top-1 agreement + KL fidelity metric.
- [`lmeval_quant.py`](lmeval_quant.py) — MMLU + GSM8K across fp32 / W8 / W4.

## Notes
- Measure fidelity with **top-1 agreement + KL**, not perplexity — Gemma-4-E2B-it on raw text gives
  confounded PPL (one low-prob token dominates the geometric mean).
- Gemma needs `tokenizer.add_bos_token=True` in lm-eval or loglikelihood-MC tasks score at random.
