#!/usr/bin/env python3
"""lm-eval-harness: fp32 vs A16W8(per-channel) vs A16W4(per-group-128) on MMLU + GSM8K.
Loads the model ONCE, re-quantizes Linear weights in place between configs. Chat template applied
(this is an -it model). Weight-only fake-quant; int16 activations are ~lossless so not simulated.
CPU, no NPU. Usage: python lmeval_quant.py <mmlu_limit> <gsm8k_limit> <batch_size> [configs csv]"""
import os, sys, json, time, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/gq/hf_cache")
# model is cached locally; keep hub ONLINE so lm-eval can fetch mmlu/gsm8k datasets
torch.set_num_threads(os.cpu_count())
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

MMLU_LIMIT = float(sys.argv[1]) if len(sys.argv)>1 else 0.1
GSM8K_LIMIT= float(sys.argv[2]) if len(sys.argv)>2 else 100
BS         = sys.argv[3] if len(sys.argv)>3 else "8"
CONFIGS    = sys.argv[4].split(",") if len(sys.argv)>4 else ["fp32","w8","w4"]
def L(x): return int(x) if x>1 else x     # >1 => count, <=1 => fraction

MODEL="google/gemma-4-E2B-it"
tok=AutoTokenizer.from_pretrained(MODEL)
# Gemma is BOS-sensitive: force the tokenizer itself to prepend BOS on every encode (mirrors CLI add_bos_token=True).
# Without this, loglikelihood MC tasks (MMLU/ARC/HellaSwag) score at random.
try: tok.add_bos_token=True
except Exception: pass
m=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, device_map="cpu").eval()

def fq(w,nbit,mode):
    q=2**(nbit-1)-1
    if mode=="channel": s=(w.abs().amax(1,keepdim=True)/q).clamp_min(1e-12); return torch.clamp(torch.round(w/s),-q-1,q)*s
    g=128;O,I=w.shape
    if I%g: return fq(w,nbit,"channel")
    wr=w.reshape(O,I//g,g); s=(wr.abs().amax(-1,keepdim=True)/q).clamp_min(1e-12)
    return (torch.clamp(torch.round(wr/s),-q-1,q)*s).reshape(O,I)

tm=m.model.language_model
lin=[mod for _,mod in tm.layers.named_modules() if isinstance(mod,torch.nn.Linear)]
orig=[mod.weight.data.clone() for mod in lin]
def apply_cfg(cfg):
    for mod,w0 in zip(lin,orig):
        if   cfg=="fp32": mod.weight.data=w0.clone()
        elif cfg=="w8":   mod.weight.data=fq(w0,8,"channel")
        elif cfg=="w4":   mod.weight.data=fq(w0,4,"group")

TASKS=[("mmlu", L(MMLU_LIMIT)), ("gsm8k", L(GSM8K_LIMIT))]
results={}
for cfg in CONFIGS:
    apply_cfg(cfg)
    lm=HFLM(pretrained=m, tokenizer=tok, batch_size=BS, add_bos_token=True)  # Gemma needs leading BOS for loglikelihood
    results[cfg]={}
    for task,lim in TASKS:
        t0=time.time()
        # MMLU is loglikelihood multiple-choice: chat template breaks " A"/" B" scoring -> run raw (leaderboard-style).
        # GSM8K is generative: chat template helps this -it model produce a parseable answer.
        gen_kw = {} if task=="mmlu" else {"apply_chat_template": True, "fewshot_as_multiturn": True}
        out=simple_evaluate(model=lm, tasks=[task], limit=lim, bootstrap_iters=0, **gen_kw)
        dt=time.time()-t0
        r=out["results"].get(task, {})
        # pick the primary metric
        acc=None
        for k in ("acc,none","exact_match,strict-match","exact_match,flexible-extract","acc_norm,none"):
            if k in r: acc=(k,r[k]); break
        results[cfg][task]={"metric":acc, "secs":round(dt,1), "raw":{k:v for k,v in r.items() if isinstance(v,(int,float))}}
        print(f"[{cfg}] {task}: {acc}  ({dt:.0f}s)", flush=True)
    del lm

print("\n===== SUMMARY =====", flush=True)
print(json.dumps(results, indent=2), flush=True)
print("LMEVAL_DONE", flush=True)
