# Running Gemma-4-E2B on the phone's NPU, at int16 activations

I wanted to run Google's **Gemma-4-E2B-it** on a **Samsung Galaxy S26 Ultra** — specifically on its
**Hexagon NPU**, not the CPU or GPU — and I wanted to keep activations at **int16** rather than dropping
to the int8 that the off-the-shelf recipe uses. Qualcomm's shipped version only offers `q4_0` (GGUF-style,
int8 activations). Keeping activations at 16 bits costs a little memory but holds onto a lot of accuracy,
so that's the path this repo takes: **A16W8** and **A16W4** (int16 activations, int8/int4 weights), built
with ONNX + [AIMET](https://github.com/quic/aimet) and pushed to real hardware through
[Qualcomm AI Hub](https://aihub.qualcomm.com/).

This is the **baseline** version — the part that just makes it *work*: a prefill graph that exports
cleanly, a decode graph you can actually loop for autoregressive generation, a KV cache that survives
ONNX tracing, and quantize→compile→profile runs on an actual S26 Ultra. It generates the same tokens as
Hugging Face does, greedily, token for token. The accuracy-recovery tricks (AWQ/GPTQ) and the decode
speedups live elsewhere; this repo is deliberately the honest starting point they build on.

## Why this is harder than it sounds

Gemma-4-E2B is **dense** (not MoE, whatever some model cards say) — 35 layers, hidden size 1536, GQA with
8 query heads to 1 KV head, head_dim 256, and a 262,144-token vocab. It also has two features that quietly
wreck a naive ONNX export:

- **Hybrid attention** — 28 sliding-window layers (window 512) interleaved with 7 full-attention layers,
  and KV is shared across the last 20 layers.
- **A giant vocab.** The embedding and lm_head tensors are big enough to blow past ONNX's 2 GB protobuf
  limit and crash the compiler.

On top of that, Hugging Face builds its attention mask and its static sliding-window cache with Python
control flow (`create_causal_mask`, `cumulative_length`) that can't be traced into one reusable graph. So
the first real problem isn't quantization at all — it's getting a graph out the door.

## How it works

**1. A clean prefill "trunk" — [`src/reexport_trunk.py`](src/reexport_trunk.py)**
Instead of exporting the whole model, I drive `Gemma4TextModel.forward` directly with `inputs_embeds`,
`per_layer_inputs`, `position_ids`, and — the key move — the attention mask handed in **as a dict**
(`{"full_attention", "sliding_attention"}`). That sidesteps HF's mask-construction subgraph entirely. The
embedding gathers and lm_head stay on the host, so the >2 GB tensors never enter the graph and the fatal
`Concat`/`Sub` mask ops never get emitted. The output is just the final hidden state. One shape gotcha
worth writing down: `per_layer_inputs` has to be `[1, SEQ, 35, 256]`
(`[1, SEQ, num_hidden_layers, hidden_size_per_layer_input]`).

**2. A decode graph you can actually loop — [`src/decode_fixed.py`](src/decode_fixed.py)**
For generation you need a single-token graph you can run over and over, feeding the KV cache back in each
step. The trick that makes it export *once* and stay reusable: the write position `cache_position` is a
**tensor input**, not Python state, so the cache update is just `index_copy(2, cache_position, key)` into
fixed `[1, nkv, 4096, head_dim]` buffers (only the 15 non-shared layers store KV). The sliding window is
handled entirely by the mask you feed in, which keeps the cache logic dead simple and traceable. Per-layer
KV dims are read straight off a live prefill — full-attention layers use head_dim 512, sliding layers 256.

**3. Proof it's correct — [`src/host_generate.py`](src/host_generate.py)**
This drives the seq=1 decode graph token by token on ONNX Runtime (CPU) with a host-side KV loop:
`embed_tokens` + `embed_tokens_per_layer` → decode graph → feed the returned KV back as the past KV →
`lm_head` argmax. It's checked against Hugging Face greedy decoding and matches exactly (`MATCH: True`).
This same loop is the reference for how you'd drive the model on-device.

**4. Quantize, compile, profile on real hardware**
- [`src/trunk_quant.py`](src/trunk_quant.py) — AIMET A16W8 on the trunk (int8 weights, int16 activations,
  `min_max`). No tensor exclusions needed, because the giant vocab tensors were already left out in step 1.
- [`src/decode_pipeline.py`](src/decode_pipeline.py) — quantizes the decode graph to A16W8, then compiles
  and profiles it on the S26 Ultra.
- [`src/w4_metrics.py`](src/w4_metrics.py) — the A16W4 version (int4 weights) for both graphs; compiles to
  a `qnn_context_binary` and profiles on the physical device.

## What I measured

Everything below ran on a **Samsung Galaxy S26 Ultra, 100% on the Hexagon NPU.**

**Speed** — naive full-KV decode graph (latency here is calibration-independent):

| graph | precision | on-device latency |
|---|---|---|
| prefill (128 tok) | A16W4 | ~69 ms |
| decode (1 token) | A16W4 | ~206 ms (~4.8 tok/s) |
| decode (1 token) | A16W8 | ~224 ms |

The interesting bit: A16W4 isn't meaningfully faster than A16W8. Decode here is **bound by the KV
attention, not the weights**, so shrinking the weights barely moves the needle. (Fixing *that* is a
separate story.)

**Accuracy** — top-1 next-token agreement vs fp32, plus MMLU (10% subsample) and GSM8K (100 questions):

| precision | top-1 agree vs fp32 | MMLU | GSM8K |
|---|---|---|---|
| fp32 (reference) | 100% | 0.571 | 0.54 |
| **A16W8** (per-channel) | **95.5%** | **0.573** (100%) | 0.53 (98%) |
| **A16W4** (RTN, per-group-128) | 60.9% | 0.497 (87%) | 0.36 (67%) |

A16W8 is basically lossless. Plain A16W4 (round-to-nearest) is noticeably lossy, and it hurts most on
multi-step reasoning — GSM8K falls off harder than MMLU, which is what you'd expect when a single bad
token can derail a whole chain.

## Trying it yourself

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_hf_token>              # google/gemma-4-E2B-it is gated
export QAI_HUB_API_TOKEN=<your_aihub_token>  # from aihub.qualcomm.com
export GQ_DIR=$PWD                           # where ONNX/quantized outputs land

python src/reexport_trunk.py   # 1. export the clean prefill trunk
python src/decode_fixed.py     # 2. export the reusable decode graph
python src/host_generate.py    # 3. sanity check: must print MATCH: True (CPU only, no AI Hub needed)
python src/trunk_quant.py      # 4. A16W8 quantize the trunk
python src/decode_pipeline.py  # 5. A16W8 decode -> compile + profile on the S26
python src/w4_metrics.py       # 6. A16W4 prefill + decode -> on-device latency
```

Steps 1–3 are local CPU work and need no device. Steps 4–6 submit jobs to AI Hub and run on a physical
phone from the device farm, so you'll need a valid `QAI_HUB_API_TOKEN`.

Heads-up on resources: AIMET quantizing the ~7.5 GB graphs is memory-hungry — I ran it on a 512 GB-RAM
host. Point `TMPDIR` at a big tmpfs (e.g. `/dev/shm`) so the compile step doesn't run out of disk.

## Things that bit me (so they don't bite you)

- Gemma-4-E2B is **dense**. Some cards in the family are labelled "MoE" — ignore that here.
- AIMET settings that actually work: `quant_scheme="min_max"`, `param_type="int8"` or `"int4"`,
  `activation_type="int16"`.
- QAIRT chokes on `CastLike` and empty `Concat` nodes. The clean-trunk export avoids ever emitting them.
- If you benchmark this with lm-eval-harness, set the tokenizer to prepend BOS — Gemma scores at random on
  loglikelihood multiple-choice tasks otherwise.
- Closing the A16W4 accuracy gap and making decode faster are real, separate problems, and they're not in
  this repo on purpose.

## License

Apache-2.0 for this pipeline code. Gemma-4-E2B itself is covered by Google's Gemma license and terms.
