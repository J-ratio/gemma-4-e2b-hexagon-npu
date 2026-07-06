#!/usr/bin/env python3
"""Run the BEST 4-bit recipe (awq_g64c = AWQ + group-64 + per-group MSE-clip) through lm-evaluation-harness
on CPU, and report MMLU + GSM8K vs the fp32 baseline. Weight-only fake-quant (int16 activations are
~lossless, so not simulated) — this measures the accuracy of the deployed weight scheme without needing
the NPU. Reproduces the headline numbers in RESULTS.md.

Usage:
  pip install torch transformers lm-eval datasets
  export HF_TOKEN=<token>              # google/gemma-4-E2B-it is gated
  python run_awq_lmeval.py [mmlu_frac] [gsm8k_n] [batch]     # defaults 0.1 100 16
"""
import os, sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

MMLU=float(sys.argv[1]) if len(sys.argv)>1 else 0.1        # fraction of MMLU (0.1 = ~1430 Q)
GSM=int(sys.argv[2]) if len(sys.argv)>2 else 100
BS=sys.argv[3] if len(sys.argv)>3 else "16"
MODEL="google/gemma-4-E2B-it"; GS=64

tok=AutoTokenizer.from_pretrained(MODEL, token=os.environ["HF_TOKEN"])
# Gemma is BOS-sensitive: without this, loglikelihood MC tasks (MMLU) score at RANDOM.
try: tok.add_bos_token=True
except Exception: pass
m=AutoModelForCausalLM.from_pretrained(MODEL, token=os.environ["HF_TOKEN"], torch_dtype=torch.float32,
                                       device_map="cpu").eval()
tm=m.model.language_model
lins=[(nm,mod) for nm,mod in tm.layers.named_modules() if isinstance(mod,torch.nn.Linear)]
orig={id(mod):mod.weight.data.clone() for _,mod in lins}
CLIP=[1.0,0.95,0.9,0.85,0.8,0.75,0.7]

def fq_clip(w,gs,nbit=4):
    """per-(row,group) MSE-optimal-clipped int4 fake-quant."""
    q=2**(nbit-1)-1; O,I=w.shape; G=1 if (gs is None or I%gs) else I//gs
    wr=w.reshape(O,G,I//G); base=wr.abs().amax(-1,keepdim=True).clamp_min(1e-12)
    best=berr=None
    for r in CLIP:
        s=base*r/q; qd=torch.clamp(torch.round(wr/s),-q-1,q)*s; e=((wr-qd)**2).sum(-1,keepdim=True)
        if best is None: best,berr=qd,e
        else: mk=e<berr; best=torch.where(mk,qd,best); berr=torch.where(mk,e,berr)
    return best.reshape(O,I)

def salience(nseq=16,sl=512):
    S={id(mod):torch.zeros(mod.weight.shape[1]) for _,mod in lins}; hks=[]
    def mk(mod):
        def h(mm,inp): x=inp[0].reshape(-1,inp[0].shape[-1]).float(); S[id(mod)]+=(x*x).sum(0)
        return h
    for _,mod in lins: hks.append(mod.register_forward_pre_hook(mk(mod)))
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")
    txt="\n\n".join(t for t in ds["text"] if t.strip()); enc=tok(txt,return_tensors="pt").input_ids[0]
    n=0;p=0
    with torch.no_grad():
        while n<nseq and p+sl<enc.numel(): m(enc[p:p+sl].unsqueeze(0),use_cache=False); p+=sl; n+=1
    for h in hks: h.remove()
    return {k:v/max(n,1) for k,v in S.items()}

def awq_g64c(W,sal):
    """AWQ per-channel scale search (Hessian-diag weighted) + group-64 MSE-clip quant. Returns effective W."""
    d=sal.clamp_min(1e-8); sa=d.sqrt(); best=None
    for a in [i/10 for i in range(11)]:
        s=(sa**a); s=(s/s.mean().clamp_min(1e-8)).clamp_min(1e-4)
        Wq=fq_clip(W*s[None,:],GS); E=Wq/s[None,:]-W; loss=(d[None,:]*E*E).sum().item()
        if best is None or loss<best[0]: best=(loss,s.clone(),Wq.clone())
    _,s,Wq=best; return Wq/s[None,:]     # deployed graph computes quant(W*s) then /s in activations

def evaluate(label):
    lm=HFLM(pretrained=m,tokenizer=tok,batch_size=BS,add_bos_token=True); out={}
    for task,lim in [("mmlu", int(MMLU) if MMLU>1 else MMLU),("gsm8k",GSM)]:
        gk={} if task=="mmlu" else {"apply_chat_template":True,"fewshot_as_multiturn":True}  # MMLU raw, GSM8K chat
        r=simple_evaluate(model=lm,tasks=[task],limit=lim,bootstrap_iters=0,**gk)["results"].get(task,{})
        acc=next((r[k] for k in ("acc,none","exact_match,strict-match","exact_match,flexible-extract") if k in r),None)
        out[task]=acc; print(f"[{label}] {task}: {acc}",flush=True)
    del lm; return out

res={}
print("=== fp32 baseline ===",flush=True); res["fp32"]=evaluate("fp32")
print("=== computing AWQ salience + quantizing to awq_g64c ===",flush=True)
SAL=salience()
for _,mod in lins: mod.weight.data=awq_g64c(orig[id(mod)],SAL[id(mod)])
res["awq_g64c"]=evaluate("awq_g64c")
print("\n===== SUMMARY (awq_g64c vs fp32) =====",flush=True)
print(json.dumps(res,indent=2,default=str),flush=True)
print("LMEVAL_DONE",flush=True)
