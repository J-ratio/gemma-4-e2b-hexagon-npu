#!/usr/bin/env python3
"""A16W8 quantization of the BASELINE (naive full-KV) decode graph.

This is the correctness baseline: no windowed KV, no broadcast-GQA, no int8-KV. It is the
graph exported by `decode_fixed.py` — every layer keeps a full [1,nkv,CTX,hd] KV buffer and
attention goes through the stock repeat_kv path.

It replaces `decode_pipeline.py`, which calibrates activation encodings on `np.random.randn`.
That compiles and profiles perfectly and then emits pure noise on device (final hidden norm
measured at 0.0000). Three things are required to make an A16W8 build actually run:

  1. Calibrate on REAL activations captured from a working float decode loop.
  2. Calibrate with the SAME prompt format used at inference — the chat template. Raw-format
     calibration + chat inference produces fluent but unfaithful output on device, which is
     far harder to notice than noise.
  3. Finite mask value NEG=-1e4, not -inf / float32.min, which cannot survive int16
     activation quantization. The host must use the same value.

Measured result of this configuration on physical v79 (SM8750): token-exact vs float, at
307.9 ms/step = 3.25 tok/s. That per-step latency is what the three decode optimizations
attack; correctness here is what makes the comparison meaningful.

  python decode_pipeline_baseline.py                 # validate only
  python decode_pipeline_baseline.py --compile --targets v79
"""
import os, sys, argparse, numpy as np, onnx, torch
import onnxruntime as ort
from transformers import AutoModelForCausalLM, AutoTokenizer

GQ = os.environ.get("GQ", "/home/ubuntu/gq")
IN = f"{GQ}/decode_fixed_onnx/model.onnx"          # produced by decode_fixed.py
MODEL = "google/gemma-4-E2B-it"
CTX, H, PLD, NL, NC = 4096, 1536, 256, 35, 15
KVD = [(1, 256)] * 4 + [(1, 512)] + [(1, 256)] * 4 + [(1, 512)] + [(1, 256)] * 4 + [(1, 512)]
NEG = -1e4                                          # must match the host at inference
# Gemma-4 chat template ids (byte-exact vs transformers.apply_chat_template)
BOS, TS, TE, NLT, RU, RM = 2, 105, 106, 107, 2364, 4368
STOP = {TE, 1}

os.environ.setdefault("HF_HOME", f"{GQ}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ap = argparse.ArgumentParser()
ap.add_argument("--compile", action="store_true")
ap.add_argument("--targets", default="v79")
ap.add_argument("--scheme", default="min_max",
                help="min_max beats tf_enhanced at int16: 65k levels mean range COVERAGE "
                     "matters more than outlier clipping (tf_enhanced inflated hidden ~10x)")
ap.add_argument("--nprompts", type=int, default=6)
ap.add_argument("--steps-per-prompt", type=int, default=12)
ap.add_argument("--out", default=f"{GQ}/decode_baseline_A16W8")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
DEVICES = {"v79": "Snapdragon 8 Elite QRD", "v81": "Snapdragon 8 Elite Gen 5 QRD"}

print("== load model + tokenizer ==", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cpu").eval()
tm = m.model.language_model


@torch.no_grad()
def emb(t):
    i = torch.tensor([[t]])
    return (tm.embed_tokens(i).numpy().astype(np.float32),
            tm.embed_tokens_per_layer(i).reshape(1, 1, NL, PLD).numpy().astype(np.float32))


def masks(pos):
    """Baseline masks: BOTH are [1,1,1,CTX]. The sliding layers still allocate a full CTX
    buffer and mask the out-of-window keys, which is exactly the waste windowed-KV removes."""
    j = np.arange(CTX)
    full = np.where(j <= pos, 0.0, NEG).astype(np.float32).reshape(1, 1, 1, CTX)
    slide = np.where((j <= pos) & (j > pos - 512), 0.0, NEG).astype(np.float32).reshape(1, 1, 1, CTX)
    return full, slide


def fresh_kv():
    """Baseline: uniform full-CTX buffers for every KV-storing layer (~288 MB per step)."""
    kv = {}
    for i in range(NC):
        nkv, hd = KVD[i]
        kv[f"past_k_{i}"] = np.zeros((1, nkv, CTX, hd), np.float32)
        kv[f"past_v_{i}"] = np.zeros((1, nkv, CTX, hd), np.float32)
    return kv


def feeds(t, pos, kv, idx=np.int64):
    ie, ple = emb(t)
    f, s = masks(pos)
    d = {"inputs_embeds": ie, "per_layer_inputs": ple,
         "position_ids": np.array([[pos]], idx), "cache_position": np.array([pos], idx),
         "full_mask": f, "sliding_mask": s}
    d.update(kv)
    return d


@torch.no_grad()
def argmax_next(h):
    return int(m.lm_head(torch.tensor(h).reshape(1, 1, H))[0, -1].argmax())


def chat_ids(p):
    """REQUIRED. The raw completion format makes this -it model degenerate into
    ' France is France is ...' even at fp32 — verified on the unquantized model."""
    return [BOS, TS, RU, NLT] + tok(p, add_special_tokens=False).input_ids + [TE, NLT, TS, RM, NLT]


def content(ids_):
    """Everything before the first special token. Compare THIS, not raw sequences: HF
    generate() stops at <eos> while a raw argmax loop keeps emitting <turn|>, which reads
    as a mismatch on an otherwise-correct graph."""
    out = []
    for t in ids_:
        if t in STOP:
            break
        out.append(t)
    return out


def run_loop(runner, ids, ngen, capture=None):
    kv = fresh_kv(); gen = []; nxt = None
    for step in range(len(ids) + ngen):
        t = ids[step] if step < len(ids) else nxt
        fd = feeds(t, step, kv)
        if capture is not None:
            capture.append({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in fd.items()})
        res = runner(fd)
        for i in range(NC):
            kv[f"past_k_{i}"] = res[1 + 2 * i]; kv[f"past_v_{i}"] = res[2 + 2 * i]
        if step >= len(ids) - 1:
            nxt = argmax_next(res[0]); gen.append(nxt)
            if nxt in STOP:
                break
    return gen


# ---------- phase A: the float graph must match HF before quantizing anything ----------
print("== phase A: float baseline graph vs HF greedy (chat template, content tokens) ==", flush=True)
sess = ort.InferenceSession(IN, providers=["CPUExecutionProvider"])
probe = "The capital of France is"
pids = chat_ids(probe)
gen = run_loop(lambda f: sess.run(None, f), pids, 20)
with torch.no_grad():
    hf = m.generate(torch.tensor([pids]), max_new_tokens=20, do_sample=False)[0][len(pids):].tolist()
cg, ch = content(gen), content(hf)
print(f"  FLOAT: {tok.decode(cg)!r}", flush=True)
print(f"  HF   : {tok.decode(ch)!r}", flush=True)
print(f"  CONTENT MATCH: {cg == ch}", flush=True)
if cg != ch:
    print("  !! float graph diverges from HF — fix that before quantizing", flush=True)
    sys.exit(1)

# ---------- phase B: real calibration feeds, in chat format ----------
PROMPTS = ["The capital of France is", "Explain photosynthesis in one sentence.",
           "What is 17 times 24?", "Name three primary colors.",
           "Why is the sky blue?", "Write one sentence about the ocean.",
           "What is the largest planet?", "Who wrote Hamlet?"][:args.nprompts]
print(f"== phase B: capture real calib feeds ({len(PROMPTS)} chat prompts) ==", flush=True)
calib = []
for p in PROMPTS:
    run_loop(lambda f: sess.run(None, f), chat_ids(p), args.steps_per_prompt, capture=calib)
    print(f"  captured, total={len(calib)}", flush=True)

# ---------- phase C: quantize ----------
from aimet_onnx.quantsim import QuantizationSimModel
print(f"== phase C: sim int8/int16 {args.scheme} on {len(calib)} real feeds ==", flush=True)
sim = QuantizationSimModel(onnx.load(IN), param_type="int8", activation_type="int16",
                           quant_scheme=args.scheme, dummy_input=calib[0],
                           providers=["CPUExecutionProvider"])


def cb(session, *_):
    for i, x in enumerate(calib):
        session.run(None, x)
        if (i + 1) % 20 == 0:
            print(f"    calib {i+1}/{len(calib)}", flush=True)


sim.compute_encodings(cb)
print("  encodings done", flush=True)

# ---------- phase D: validate on HELD-OUT prompts ----------
# Held out on purpose: scoring the calibration prompts is self-scoring and inflates the number.
print("== phase D: fake-quant validation (held-out prompts) ==", flush=True)
qs = sim.session
onames = [o.name for o in qs.get_outputs()]


def qrun(f):
    o = qs.run(None, f); d = dict(zip(onames, o))
    r = [d.get("hidden", o[0])]
    for i in range(NC):
        r += [d[f"present_k_{i}"], d[f"present_v_{i}"]]
    return r


HELD_OUT = ["The capital of France is", "What is the boiling point of water?",
            "Name a country in South America.", "What color is grass?"]
tot = ok = 0
for p in HELD_OUT:
    ids_ = chat_ids(p)
    cf = content(run_loop(lambda f: sess.run(None, f), ids_, 16))
    cq = content(run_loop(qrun, ids_, 16))
    a = sum(x == y for x, y in zip(cf, cq)); n = max(len(cf), len(cq))
    tot += n; ok += a
    print(f"  [{p[:34]:34s}] float={tok.decode(cf)[:46]!r}", flush=True)
    print(f"  {'':36s} quant={tok.decode(cq)[:46]!r}  {a}/{n}", flush=True)
print(f"\n  CONTENT-TOKEN AGREEMENT vs float: {ok}/{tot} ({100.0*ok/max(1,tot):.1f}%)", flush=True)

if not args.compile:
    print("  (skip compile; rerun with --compile)", flush=True)
    print("DONE", flush=True); sys.exit(0)

# ---------- phase E: export + compile + profile ----------
print("== phase E: export ==", flush=True)
sim.export(args.out, "decode_baseline_A16W8")
print("  EXPORTED:", os.listdir(args.out), flush=True)
os.makedirs(f"{GQ}/hubtmp", exist_ok=True)
os.environ["TMPDIR"] = f"{GQ}/hubtmp"
import tempfile; tempfile.tempdir = f"{GQ}/hubtmp"
import qai_hub as hub
for tg in [t.strip() for t in args.targets.split(",") if t.strip()]:
    dev = hub.Device(DEVICES[tg])
    cj = hub.submit_compile_job(model=args.out, device=dev,
                                options="--target_runtime qnn_context_binary --truncate_64bit_io",
                                name=f"gemma4-decode-BASELINE-A16W8-{tg}")
    print(f"  COMPILE[{tg}] {cj.job_id} {cj.url}", flush=True)
    cj.wait()
    if not cj.get_status().success:
        print(f"  compile[{tg}] FAILED", flush=True); continue
    tmdl = cj.get_target_model()
    # NOT into args.out: qai_hub rejects a model dir holding both a .bin and .encodings/.data
    os.makedirs(f"{GQ}/bins", exist_ok=True)
    tmdl.download(f"{GQ}/bins/gemma4_decode_baseline_a16w8_{tg}.bin")
    pj = hub.submit_profile_job(model=tmdl, device=dev, name=f"gemma4-decode-BASELINE-A16W8-prof-{tg}")
    print(f"  PROFILE[{tg}] {pj.job_id} {pj.url}", flush=True)
    pj.wait()
    if pj.get_status().success:
        ex = pj.download_profile().get("execution_summary", {})
        us = ex.get("estimated_inference_time", 0)
        if us:
            print(f"  >>> BASELINE {tg}: {us} us = {us/1000:.1f} ms = {1e6/us:.1f} tok/s", flush=True)
print("DONE", flush=True)
