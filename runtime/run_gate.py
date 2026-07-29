#!/usr/bin/env python3
"""Host orchestrator for the A16W8 v79 correctness gate.

Runs the autoregressive decode loop by driving qnn-net-run on-device once per token.
KV (~288MB) stays resident on device as files; only tiny per-step tensors cross adb.

Prereqs on device (staged by push_gate.sh):
  /data/local/tmp/gemma/{bin,lib,dsp,artifacts,step,kv,out}
Host math from hostlib.py (gemma3n scaling, tied softcapped lm_head).

Usage:
  python run_gate.py --prompt "The capital of France is" --ntokens 15 [--adb-serial X]
  Optionally --hf-check to compare against HF greedy on host (needs full model; heavy).
"""
import argparse, os, subprocess, sys, time, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hostlib

BASE = "/data/local/tmp/gemma"
STEP = f"{BASE}/step"
KV = f"{BASE}/kv"
OUTR = f"{BASE}/out/Result_0"
LOCAL_STEP = pathlib.Path("/tmp/gemma_step")
CTX = hostlib.CTX
H = hostlib.H
KV_HD = hostlib.KV_HD

def adb(args, serial=None, **kw):
    cmd = ["adb"] + (["-s", serial] if serial else []) + args
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def adb_shell(script, serial=None, timeout=600):
    return adb(["shell", script], serial=serial, timeout=timeout)

def push(local, remote, serial=None):
    r = adb(["push", str(local), remote], serial=serial, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"push failed {local}->{remote}: {r.stderr}")

def pull(remote, local, serial=None):
    r = adb(["pull", remote, str(local)], serial=serial, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"pull failed {remote}->{local}: {r.stderr}")

def seed_kv(serial):
    """Zero the 15 past_k/v buffers on device.

    Created ON DEVICE with dd rather than pushed: they are ~144MB of zeros, and pushing
    them over the QDC tunnel took minutes and was the flakiest part of the run (a reset
    mid-seed leaves a partial buffer and silently corrupts the whole generation).
    Baseline: every layer gets a full CTX-deep buffer.
    """
    LOCAL_STEP.mkdir(exist_ok=True)
    adb_shell(f"mkdir -p {KV} {STEP} {BASE}/out", serial=serial)
    depths = [CTX] * hostlib.NC
    cmds = [f"rm -f {KV}/*.raw"]
    for i in range(hostlib.NC):
        nbytes = depths[i] * KV_HD[i] * 4          # [1,1,depth,hd] float32
        for kind in ("k", "v"):
            cmds.append(f"dd if=/dev/zero of={KV}/past_{kind}_{i}.raw bs=4096 "
                        f"count={nbytes // 4096} 2>/dev/null")
    r = adb_shell(" && ".join(cmds), serial=serial, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"on-device KV seed failed: {r.stderr}")

def write_step_inputs(m, token_id, pos, serial):
    ie, ple = m.embeds(token_id)
    full, slide = m.masks(pos)
    files = {
        "inputs_embeds": ie.astype(np.float32),
        "per_layer_inputs": ple.astype(np.float32),
        "position_ids": np.array([[pos]], np.int32),
        "cache_position": np.array([pos], np.int32),
        "full_mask": full.astype(np.float32),
        "sliding_mask": slide.astype(np.float32),
    }
    for name, arr in files.items():
        p = LOCAL_STEP / f"{name}.raw"
        arr.tofile(p)
        push(p, f"{STEP}/{name}.raw", serial=serial)

def run_step(serial):
    r = adb_shell(f"sh {BASE}/gate_ondevice.sh", serial=serial, timeout=600)
    if "STEP_OK" not in r.stdout:
        raise RuntimeError(f"net-run step failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}")
    return r

def fetch_hidden(serial):
    pull(f"{OUTR}/hidden.raw", LOCAL_STEP / "hidden.raw", serial=serial)
    return np.fromfile(LOCAL_STEP / "hidden.raw", dtype=np.float32).reshape(H)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--ntokens", type=int, default=15)
    ap.add_argument("--adb-serial", default=os.environ.get("ADB_SERIAL"))
    ap.add_argument("--hf-check", action="store_true")
    ap.add_argument("--chat", action="store_true",
                    help="wrap the prompt in the Gemma-4 chat template (required for coherent "
                         "output from the -it model; raw completion format degenerates)")
    args = ap.parse_args()

    print("loading host model (embeddings + tokenizer)...", flush=True)
    m = hostlib.HostModel()
    ids = m.encode_chat(args.prompt) if args.chat else m.encode(args.prompt)
    print(f"prompt: {args.prompt!r}  (chat_template={args.chat})\nids: {ids}", flush=True)

    print("seeding KV buffers on device...", flush=True)
    seed_kv(args.adb_serial)

    seq = ids[:]
    gen_ids = []
    pos = 0
    t_steps = []
    # prefill+decode: feed prompt tokens one-by-one (pos advances), then greedy-generate
    total = len(seq) + args.ntokens
    nxt = None
    for step in range(total):
        t = seq[step] if step < len(seq) else nxt
        write_step_inputs(m, t, pos, args.adb_serial)
        t0 = time.time()
        run_step(args.adb_serial)
        dt = time.time() - t0
        t_steps.append(dt)
        hidden = fetch_hidden(args.adb_serial)
        pos += 1
        if step >= len(seq) - 1:  # last prompt token onward -> predict next
            nxt = m.argmax_next(hidden)
            gen_ids.append(nxt)
            print(f"  step {step:2d} pos {pos-1:2d}  {dt*1000:7.1f}ms  -> id {nxt:6d} {m.decode([nxt])!r}", flush=True)
            if nxt in hostlib.STOP_IDS:
                print("  (stop token reached)", flush=True)
                break

    text = m.decode([t for t in gen_ids if t not in hostlib.STOP_IDS])
    print("\n=== GENERATION ===")
    print("continuation:", repr(text))
    print(f"per-step wall (incl adb+netrun init): mean {1000*np.mean(t_steps):.0f}ms  min {1000*min(t_steps):.0f}ms")
    print("NOTE: this wall time is NOT throughput (net-run reloads context each step). Coherence/accuracy only.")

    if args.hf_check:
        hf_compare(m, ids, gen_ids)

def hf_compare(m, ids, gen_ids):
    print("\n=== HF greedy reference (host, CPU) ===", flush=True)
    import torch
    from transformers import AutoModelForCausalLM
    tok = (pathlib.Path.home() / ".cache/huggingface/token").read_text().strip()
    mdl = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it", token=tok,
                                               torch_dtype=torch.float32, device_map="cpu").eval()
    with torch.no_grad():
        out = mdl.generate(torch.tensor([ids]), max_new_tokens=len(gen_ids), do_sample=False)
    hf = out[0][len(ids):].tolist()
    print("HF  :", repr(m.decode(hf)))
    print("NPU :", repr(m.decode(gen_ids)))
    match = sum(a == b for a, b in zip(hf, gen_ids))
    print(f"token match: {match}/{len(gen_ids)}")

if __name__ == "__main__":
    main()
