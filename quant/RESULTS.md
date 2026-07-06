# awq_g64c — measured accuracy (CPU lm-evaluation-harness)

Best pure-4-bit recipe: **AWQ + group-64 + per-group MSE-clip** (`awq_g64c`), int16 activations, weight-only
fake-quant. Reproduce with [`run_awq_lmeval.py`](run_awq_lmeval.py). Baseline is fp32 Gemma-4-E2B-it.

## MMLU + GSM8K (lm-eval-harness)

MMLU = 10% subsample (~1430 questions, reliable ±~1.3%). GSM8K = 100 questions (high variance, ±~10 pts).
Gemma needs `tokenizer.add_bos_token=True` or MMLU scores at random; MMLU run raw, GSM8K with chat template.

| config | MMLU | (% of fp32) | GSM8K | top-1 agree vs fp32 | KL |
|---|---|---|---|---|---|
| fp32 (baseline) | 0.571 | — | 0.54 | 100% | 0 |
| A16W8 per-channel | 0.573 | 100% | 0.53 | 93.5% | 0.035 |
| naive A16W4 (RTN g128) | 0.497 | 87% | 0.36 | 60.9% | 1.42 |
| **awq_g64c (AWQ+g64+clip)** | **0.549** | **96%** | **0.41** | **79.3%** | **0.593** |

`awq_g64c` recovers MMLU to **96% of fp32** and lifts top-1 agreement from 60.9% (naive W4) to **79.3%**, at
pure 4-bit weights — AWQ's per-channel scale folds into the preceding RMSNorm (no runtime cost).

## How it was reached (ablation, top-1 agreement vs fp32)

| method | top-1 |
|---|---|
| RTN per-group-128 (naive) | 60.9% |
| RTN per-group-64 | 66.3% |
| GPTQ per-group-128 (act_order) | 67.4% |
| AWQ per-group-128 | 68.5% |
| **AWQ + group-64 + MSE-clip** | **79.3%** |

Findings: stacking matters (AWQ 68.5%@g128 → 79.3%@g64+clip); MSE-clip helps AWQ but hurts GPTQ; group-64 is
the sweet spot (group-32 was worse on both MMLU and GSM8K).

## Caveats
- These are **teacher-forced** metrics (per-token fidelity / MC likelihood), not free-running generation
  quality — they don't translate linearly to generated-text coherence.
- GSM8K@100 is high-variance (0.13–0.67 across otherwise-good configs); trust **MMLU + top-1 agreement + KL**.
- Perplexity is unreliable here: Gemma-4-E2B-**it** on raw text gives confounded PPL (one low-prob token
  dominates the geometric mean) — that's why fidelity is measured by agreement/KL, not PPL.
