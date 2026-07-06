#!/usr/bin/env python3
"""Better W4 for Gemma-4-E2B, all WEIGHT-ONLY so it plugs into the same fake-quant benchmark:
  - gptq_g128 : GPTQ (Hessian error-compensated) per-group-128
  - rtn_g32   : plain round-to-nearest per-group-32 (cheap win)
  - mixed     : layers 8-15 kept at W8, rest W4 RTN g128 (from our per-layer sensitivity map)
Baselines already measured: fp32, w8(per-ch), rtn_g128(=our A16W4).
Modes:
  python w4_better.py sanity                      -> top-1 agreement + KL vs fp32 (fast)
  python w4_better.py full <mmlu> <gsm8k> <bs> <configs csv>
CPU, no NPU."""
import os, sys, json, time, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/gq/hf_cache")
torch.set_num_threads(int(os.environ.get("NT", os.cpu_count())))
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODE=sys.argv[1] if len(sys.argv)>1 else "sanity"
MODEL="google/gemma-4-E2B-it"
tok=AutoTokenizer.from_pretrained(MODEL)
try: tok.add_bos_token=True
except Exception: pass
m=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, device_map="cpu").eval()
tm=m.model.language_model
NL=len(tm.layers)

# ---- target linears (decoder layers only), remember their layer index ----
targets=[]          # (name, module, layer_idx)
for li,layer in enumerate(tm.layers):
    for nm,mod in layer.named_modules():
        if isinstance(mod, torch.nn.Linear): targets.append((f"L{li}.{nm}", mod, li))
orig={id(mod):mod.weight.data.clone() for _,mod,_ in targets}
print(f"target linears: {len(targets)} across {NL} layers", flush=True)

# ---------- fake-quant (RTN) ----------
def fq(w,nbit,gs):
    q=2**(nbit-1)-1
    O,I=w.shape
    if gs is None or I%gs:   # per-channel fallback
        s=(w.abs().amax(1,keepdim=True)/q).clamp_min(1e-12); return torch.clamp(torch.round(w/s),-q-1,q)*s
    wr=w.reshape(O,I//gs,gs); s=(wr.abs().amax(-1,keepdim=True)/q).clamp_min(1e-12)
    return (torch.clamp(torch.round(wr/s),-q-1,q)*s).reshape(O,I)

# ---------- GPTQ (weight-only, per-group) ----------
def gptq_layer(W, H, nbit=4, groupsize=128, percdamp=0.01, blocksize=128, act_order=True):
    W=W.clone().float(); O,I=W.shape; q=2**(nbit-1)-1
    H=H.clone().float()
    dead=torch.diag(H)==0; H[dead,dead]=1.0; W[:,dead]=0
    if act_order:                                     # process most-important input channels first
        perm=torch.argsort(torch.diag(H),descending=True)
        W=W[:,perm]; H=H[perm][:,perm]; invperm=torch.argsort(perm)
    damp=percdamp*torch.mean(torch.diag(H)).clamp_min(1e-8)
    idx=torch.arange(I); H[idx,idx]+=damp
    H=torch.linalg.cholesky(H); H=torch.cholesky_inverse(H); H=torch.linalg.cholesky(H,upper=True)
    Hinv=H
    Q=torch.zeros_like(W)
    scale_cache={}
    for i1 in range(0,I,blocksize):
        i2=min(i1+blocksize,I); cnt=i2-i1
        W1=W[:,i1:i2].clone(); Q1=torch.zeros_like(W1); Err1=torch.zeros_like(W1); Hinv1=Hinv[i1:i2,i1:i2]
        for i in range(cnt):
            col=i1+i
            if col%groupsize==0:
                a=col; b=min(col+groupsize,I); scale_cache["cur"]=W[:,a:b].abs().amax(1,keepdim=True).div(q).clamp_min(1e-12)
            s=scale_cache["cur"][:,0]
            w=W1[:,i]; d=Hinv1[i,i]
            qv=torch.clamp(torch.round(w/s),-q-1,q)*s
            Q1[:,i]=qv
            err=(w-qv)/d
            W1[:,i:]-=err.unsqueeze(1)*Hinv1[i,i:].unsqueeze(0)
            Err1[:,i]=err
        Q[:,i1:i2]=Q1
        W[:,i2:]-=Err1@Hinv[i1:i2,i2:]
    if act_order: Q=Q[:,invperm]
    return Q

# ---------- calibration Hessians ----------
def collect_hessians(nsamples=32, seqlen=512):
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")
    text="\n\n".join(t for t in ds["text"] if t.strip())
    enc=tok(text, return_tensors="pt").input_ids[0]
    H={id(mod):torch.zeros(mod.weight.shape[1],mod.weight.shape[1]) for _,mod,_ in targets}
    hooks=[]
    def mk(mod):
        def hook(module, inp):
            x=inp[0].reshape(-1, inp[0].shape[-1]).float()
            H[id(mod)]+=x.t()@x
        return hook
    for _,mod,_ in targets: hooks.append(mod.register_forward_pre_hook(mk(mod)))
    n=0; pos=0
    with torch.no_grad():
        while n<nsamples and pos+seqlen<enc.numel():
            ii=enc[pos:pos+seqlen].unsqueeze(0); pos+=seqlen; n+=1
            m(ii)
            if n%8==0: print(f"  calib {n}/{nsamples}", flush=True)
    for h in hooks: h.remove()
    return H

# ---------- AWQ (activation-aware weight scaling), weight-only via reciprocal-scale input hook ----------
AWQ_HOOKS=[]
def clear_awq():
    global AWQ_HOOKS
    for h in AWQ_HOOKS: h.remove()
    AWQ_HOOKS=[]
def awq_scale(W, H, nbit=4, gs=128):
    d=torch.diag(H).clamp_min(1e-8)          # ~ sum_t x_j^2  (per-input-channel salience)
    sa=d.sqrt()
    best=None
    for a in [i/10 for i in range(11)]:
        s=sa**a; s=(s/s.mean().clamp_min(1e-8)).clamp_min(1e-4)   # unit-mean per-channel scale [I]
        Wq=fq(W*s[None,:],nbit,gs)
        E=Wq/s[None,:]-W                     # error seen by ORIGINAL activation
        loss=(d[None,:]*E*E).sum().item()    # Hessian-diagonal-weighted output error (AWQ objective)
        if best is None or loss<best[0]: best=(loss,s.clone(),Wq.clone())
    return best[1], best[2]                    # (scale, scaled-quant weight)

HESS=None
def ensure_hess():
    global HESS
    if HESS is None:
        print("collecting calibration Hessians ...", flush=True); t0=time.time()
        HESS=collect_hessians()
        print(f"Hessians done in {time.time()-t0:.0f}s", flush=True)

# ---------- apply a config in place ----------
MIXED_W8_LAYERS=set(range(8,16))   # per-layer sensitivity: keep the middle block at W8
def apply_cfg(cfg):
    clear_awq()                              # drop any AWQ input hooks from a previous config
    if cfg=="fp32":
        for _,mod,_ in targets: mod.weight.data=orig[id(mod)].clone(); return
    if cfg=="awq_g128":
        ensure_hess()
        print("running AWQ ...", flush=True); t0=time.time()
        for k,(nm,mod,_) in enumerate(targets):
            s,Wq=awq_scale(orig[id(mod)], HESS[id(mod)], 4, 128)
            mod.weight.data=Wq
            def mk(sc):
                def hook(module,args): return (args[0]/sc.to(args[0].dtype),)+tuple(args[1:])
                return hook
            AWQ_HOOKS.append(mod.register_forward_pre_hook(mk(s)))
            if (k+1)%80==0: print(f"  awq {k+1}/{len(targets)} ({time.time()-t0:.0f}s)",flush=True)
        print(f"AWQ done {time.time()-t0:.0f}s",flush=True); return
    if cfg=="gptq_g128":
        ensure_hess()
        print("running GPTQ ...", flush=True); t0=time.time()
        for k,(nm,mod,_) in enumerate(targets):
            mod.weight.data=gptq_layer(orig[id(mod)], HESS[id(mod)], 4, 128)
            if (k+1)%60==0: print(f"  gptq {k+1}/{len(targets)}  ({time.time()-t0:.0f}s)", flush=True)
        print(f"GPTQ done {time.time()-t0:.0f}s", flush=True); return
    for _,mod,li in targets:
        w0=orig[id(mod)]
        if   cfg=="rtn_g128": mod.weight.data=fq(w0,4,128)
        elif cfg=="rtn_g32":  mod.weight.data=fq(w0,4,32)
        elif cfg=="w8":       mod.weight.data=fq(w0,8,None)
        elif cfg=="mixed":    mod.weight.data=fq(w0,8,None) if li in MIXED_W8_LAYERS else fq(w0,4,128)

# ---------- sanity: top-1 agreement + KL vs fp32 ----------
def sanity(configs):
    PROMPTS=["Explain how photosynthesis works.","What is the capital of France? Name two landmarks.",
             "Why does quantization matter for phones?","A train travels 60 km in 1.5 h. Average speed?",
             "Three differences between ML and traditional programming."]
    ids=torch.cat([tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,
                   return_tensors="pt",return_dict=True)["input_ids"] for p in PROMPTS],dim=1)[:,:512]
    @torch.no_grad()
    def lg(): return m(ids).logits[0,:-1].float()
    apply_cfg("fp32");
    with torch.no_grad(): ref=lg(); rt1=ref.argmax(-1); rlp=torch.log_softmax(ref,-1)
    print(f"\n{'config':14s} {'top1-agree':>11s} {'meanKL':>9s}", flush=True)
    for cfg in configs:
        apply_cfg(cfg)
        with torch.no_grad():
            l=lg(); ag=(l.argmax(-1)==rt1).float().mean().item()
            kl=torch.nn.functional.kl_div(torch.log_softmax(l,-1),rlp,log_target=True,reduction="batchmean").item()
        print(f"{cfg:14s} {100*ag:9.1f}% {kl:9.4f}", flush=True)
    print("SANITY_DONE", flush=True)

# ---------- full lm-eval ----------
def full(mmlu_lim, gsm_lim, bs, configs):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    def L(x): return int(x) if x>1 else x
    TASKS=[("mmlu",L(mmlu_lim)),("gsm8k",L(gsm_lim))]
    res={}
    for cfg in configs:
        apply_cfg(cfg)
        lm=HFLM(pretrained=m, tokenizer=tok, batch_size=bs, add_bos_token=True)
        res[cfg]={}
        for task,lim in TASKS:
            gk={} if task=="mmlu" else {"apply_chat_template":True,"fewshot_as_multiturn":True}
            out=simple_evaluate(model=lm,tasks=[task],limit=lim,bootstrap_iters=0,**gk)
            r=out["results"].get(task,{})
            acc=next(((k,r[k]) for k in ("acc,none","exact_match,strict-match","exact_match,flexible-extract") if k in r),None)
            res[cfg][task]=acc; print(f"[{cfg}] {task}: {acc}",flush=True)
        del lm
    print("\n===== SUMMARY ====="); print(json.dumps(res,indent=2,default=str)); print("LMEVAL_DONE",flush=True)

if MODE=="sanity":
    sanity(["rtn_g128","awq_g128"])
else:
    mmlu=float(sys.argv[2]); gsm=float(sys.argv[3]); bs=sys.argv[4]
    cfgs=sys.argv[5].split(",") if len(sys.argv)>5 else ["rtn_g32","mixed","gptq_g128"]
    full(mmlu,gsm,bs,cfgs)
