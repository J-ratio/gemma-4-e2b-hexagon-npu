#!/usr/bin/env python3
"""Explore PURE uniform-W4 (NO mixed precision) to maximize fidelity for Gemma-4-E2B.
Composable weight-only transforms (all deploy-compatible on Hexagon/QAIRT):
  rotation (random-orthogonal, QuaRot-style) -> AWQ per-channel scale -> {RTN|GPTQ} at group gs with MSE clip.
Each is exact via an input transform hook: x -> x@M so fq(W')@ (x@M) == W x pre-quant.
Modes:
  python w4_explore.py sanity            -> top-1 agreement + KL vs fp32 for the whole ladder (Hessian collected once)
  python w4_explore.py full <mmlu> <gsm8k> <bs> <cfg names csv>
CPU, no NPU."""
import os, sys, json, time, torch
os.environ.setdefault("HF_HOME","/home/ubuntu/gq/hf_cache")
torch.set_num_threads(int(os.environ.get("NT",os.cpu_count())))
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODE=sys.argv[1] if len(sys.argv)>1 else "sanity"
MODEL="google/gemma-4-E2B-it"
tok=AutoTokenizer.from_pretrained(MODEL)
try: tok.add_bos_token=True
except Exception: pass
m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,device_map="cpu").eval()
tm=m.model.language_model
targets=[]
for li,layer in enumerate(tm.layers):
    for nm,mod in layer.named_modules():
        if isinstance(mod,torch.nn.Linear): targets.append((f"L{li}.{nm}",mod,li))
orig={id(mod):mod.weight.data.clone() for _,mod,_ in targets}
print(f"targets: {len(targets)}",flush=True)
CLIP_GRID=[1.0,0.95,0.9,0.85,0.8,0.75,0.7]

# ---- fq with optional per-(row,group) MSE-optimal clip ----
def fq(w,nbit,gs,clip=False):
    q=2**(nbit-1)-1;O,I=w.shape
    G = 1 if (gs is None or I%gs) else I//gs
    wr=w.reshape(O,G,I//G)
    base=wr.abs().amax(-1,keepdim=True).clamp_min(1e-12)
    if not clip:
        s=base/q; return (torch.clamp(torch.round(wr/s),-q-1,q)*s).reshape(O,I)
    best=None;berr=None
    for r in CLIP_GRID:
        s=base*r/q; qd=torch.clamp(torch.round(wr/s),-q-1,q)*s
        err=((wr-qd)**2).sum(-1,keepdim=True)
        if best is None: best,berr=qd,err
        else: mk=err<berr; best=torch.where(mk,qd,best); berr=torch.where(mk,err,berr)
    return best.reshape(O,I)

# ---- random-orthogonal rotation, cached per size ----
_ROT={}
def rot(n):
    if n not in _ROT:
        g=torch.Generator().manual_seed(1234+n); A=torch.randn(n,n,generator=g)
        Q,_=torch.linalg.qr(A); _ROT[n]=Q.float()
    return _ROT[n]

# ---- GPTQ (act_order) with optional MSE clip on group scale ----
def gscale(cols,q,clip):
    base=cols.abs().amax(1,keepdim=True).clamp_min(1e-12)/q
    if not clip: return base
    best=None;berr=None
    for r in CLIP_GRID:
        s=base*r; qd=torch.clamp(torch.round(cols/s),-q-1,q)*s
        err=((cols-qd)**2).sum(1,keepdim=True)
        if best is None: best,berr=s,err
        else: mk=err<berr; best=torch.where(mk,s,best); berr=torch.where(mk,err,berr)
    return best
def gptq_layer(W,H,gs=128,clip=False,nbit=4,percdamp=0.01,blocksize=128):
    W=W.clone().float();O,I=W.shape;q=2**(nbit-1)-1;H=H.clone().float()
    dead=torch.diag(H)==0;H[dead,dead]=1.0;W[:,dead]=0
    perm=torch.argsort(torch.diag(H),descending=True);W=W[:,perm];H=H[perm][:,perm];invperm=torch.argsort(perm)
    damp=percdamp*torch.mean(torch.diag(H)).clamp_min(1e-8);idx=torch.arange(I);H[idx,idx]+=damp
    H=torch.linalg.cholesky(H);H=torch.cholesky_inverse(H);H=torch.linalg.cholesky(H,upper=True);Hinv=H
    Q=torch.zeros_like(W);cur=None
    for i1 in range(0,I,blocksize):
        i2=min(i1+blocksize,I);cnt=i2-i1
        W1=W[:,i1:i2].clone();Q1=torch.zeros_like(W1);Err1=torch.zeros_like(W1);Hinv1=Hinv[i1:i2,i1:i2]
        for i in range(cnt):
            col=i1+i
            if col%gs==0: b=min(col+gs,I); cur=gscale(W[:,col:b],q,clip)[:,0]
            w=W1[:,i];d=Hinv1[i,i]
            qv=torch.clamp(torch.round(w/cur),-q-1,q)*cur;Q1[:,i]=qv
            err=(w-qv)/d;W1[:,i:]-=err.unsqueeze(1)*Hinv1[i,i:].unsqueeze(0);Err1[:,i]=err
        Q[:,i1:i2]=Q1;W[:,i2:]-=Err1@Hinv[i1:i2,i2:]
    return Q[:,invperm]

# ---- AWQ per-channel scale search (Hessian-diag weighted, with clip in trial quant) ----
def awq_scale(Wr,Hr,gs,clip,nbit=4):
    d=torch.diag(Hr).clamp_min(1e-8);sa=d.sqrt();best=None
    for a in [i/10 for i in range(11)]:
        s=sa**a;s=(s/s.mean().clamp_min(1e-8)).clamp_min(1e-4)
        Wq=fq(Wr*s[None,:],nbit,gs,clip);E=Wq/s[None,:]-Wr;loss=(d[None,:]*E*E).sum().item()
        if best is None or loss<best[0]: best=(loss,s.clone())
    return best[1]

# ---- calibration Hessians ----
HESS=None
def ensure_hess(ns=32,sl=512):
    global HESS
    if HESS is not None: return
    print("collecting Hessians ...",flush=True);t0=time.time()
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")
    text="\n\n".join(t for t in ds["text"] if t.strip());enc=tok(text,return_tensors="pt").input_ids[0]
    HESS={id(mod):torch.zeros(mod.weight.shape[1],mod.weight.shape[1]) for _,mod,_ in targets}
    hks=[]
    def mk(mod):
        def h(module,inp):
            x=inp[0].reshape(-1,inp[0].shape[-1]).float();HESS[id(mod)]+=x.t()@x
        return h
    for _,mod,_ in targets: hks.append(mod.register_forward_pre_hook(mk(mod)))
    n=0;pos=0
    with torch.no_grad():
        while n<ns and pos+sl<enc.numel(): m(enc[pos:pos+sl].unsqueeze(0));pos+=sl;n+=1
    for h in hks: h.remove()
    print(f"Hessians {time.time()-t0:.0f}s",flush=True)

# ---- config = dict(rot,awq,gptq,gs,clip); build weight + input-transform hook ----
HOOKS=[]
def clear_hooks():
    global HOOKS
    for h in HOOKS: h.remove()
    HOOKS=[]
def build(W0,H,o):
    O,I=W0.shape
    if o["rot"]: Q=rot(I);Wr=W0@Q;Hr=Q.t()@H@Q
    else: Q=None;Wr=W0;Hr=H
    s=awq_scale(Wr,Hr,o["gs"],o["clip"]) if o["awq"] else torch.ones(I)
    Wsc=Wr*s[None,:]
    if o["gptq"]:
        Hsc=Hr*(1.0/s)[:,None]*(1.0/s)[None,:];Wq=gptq_layer(Wsc,Hsc,o["gs"],o["clip"])
    else: Wq=fq(Wsc,4,o["gs"],o["clip"])
    if o["rot"]: kind,T="mat",(Q*(1.0/s)[None,:])
    elif o["awq"]: kind,T="vec",(1.0/s)
    else: kind,T=None,None
    return Wq,kind,T
def apply_cfg(o):
    clear_hooks()
    if o.get("fp32"):
        for _,mod,_ in targets: mod.weight.data=orig[id(mod)].clone()
        return
    need_h = o["awq"] or o["gptq"] or o["rot"]   # rot needs Hr only if awq/gptq; but Hr harmless
    if o["awq"] or o["gptq"]: ensure_hess()
    t0=time.time()
    for k,(nm,mod,_) in enumerate(targets):
        H=HESS[id(mod)] if HESS is not None else torch.eye(mod.weight.shape[1])
        Wq,kind,T=build(orig[id(mod)],H,o);mod.weight.data=Wq
        if kind=="mat":
            def mkm(M):
                def h(mm,a): return (a[0]@M.to(a[0].dtype),)+tuple(a[1:])
                return h
            HOOKS.append(mod.register_forward_pre_hook(mkm(T)))
        elif kind=="vec":
            def mkv(v):
                def h(mm,a): return (a[0]*v.to(a[0].dtype),)+tuple(a[1:])
                return h
            HOOKS.append(mod.register_forward_pre_hook(mkv(T)))
    print(f"  built {o['name']} ({time.time()-t0:.0f}s)",flush=True)

CFGS={
 "rtn_g128":      dict(rot=0,awq=0,gptq=0,gs=128,clip=0),
 "rtn_g64":       dict(rot=0,awq=0,gptq=0,gs=64, clip=0),
 "rtnclip_g64":   dict(rot=0,awq=0,gptq=0,gs=64, clip=1),
 "awq_g64c":      dict(rot=0,awq=1,gptq=0,gs=64, clip=1),
 "gptq_g64c":     dict(rot=0,awq=0,gptq=1,gs=64, clip=1),
 "awqgptq_g64c":  dict(rot=0,awq=1,gptq=1,gs=64, clip=1),
 "rot_rtn_g128":  dict(rot=1,awq=0,gptq=0,gs=128,clip=0),
 "rot_awq_g64c":  dict(rot=1,awq=1,gptq=0,gs=64, clip=1),
 "rot_gptq_g64c": dict(rot=1,awq=0,gptq=1,gs=64, clip=1),
 "rot_awqgptq_g64c":dict(rot=1,awq=1,gptq=1,gs=64,clip=1),
 # follow-up AWQ group-size / clip ablations (winner family)
 "awq_g32c":      dict(rot=0,awq=1,gptq=0,gs=32, clip=1),
 "awq_g128c":     dict(rot=0,awq=1,gptq=0,gs=128,clip=1),
 "awq_g64":       dict(rot=0,awq=1,gptq=0,gs=64, clip=0),
 "awq_g32":       dict(rot=0,awq=1,gptq=0,gs=32, clip=0),
}
for n,o in CFGS.items(): o["name"]=n

def sanity(names):
    PR=["Explain how photosynthesis works.","What is the capital of France? Name two landmarks.",
        "Why does quantization matter for phones?","A train travels 60 km in 1.5 h. Average speed?",
        "Three differences between ML and traditional programming."]
    ids=torch.cat([tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,
                   return_tensors="pt",return_dict=True)["input_ids"] for p in PR],dim=1)[:,:512]
    @torch.no_grad()
    def lg(): return m(ids).logits[0,:-1].float()
    apply_cfg(dict(name="fp32",fp32=1,rot=0,awq=0,gptq=0,gs=128,clip=0))
    with torch.no_grad(): ref=lg();rt1=ref.argmax(-1);rlp=torch.log_softmax(ref,-1)
    print(f"\n{'config':20s} {'top1':>7s} {'KL':>9s}",flush=True);rows=[]
    for nm in names:
        apply_cfg(CFGS[nm])
        with torch.no_grad():
            l=lg();ag=(l.argmax(-1)==rt1).float().mean().item()
            kl=torch.nn.functional.kl_div(torch.log_softmax(l,-1),rlp,log_target=True,reduction="batchmean").item()
        print(f"{nm:20s} {100*ag:6.1f}% {kl:9.4f}",flush=True);rows.append((nm,ag,kl))
    print("\nRANK:",", ".join(f"{n}:{100*a:.1f}%" for n,a,_ in sorted(rows,key=lambda x:-x[1])),flush=True)
    print("SANITY_DONE",flush=True)

def full(mmlu,gsm,bs,names):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    def L(x): return int(x) if x>1 else x
    res={}
    for nm in names:
        apply_cfg(CFGS[nm]);lm=HFLM(pretrained=m,tokenizer=tok,batch_size=bs,add_bos_token=True);res[nm]={}
        for task,lim in [("mmlu",L(mmlu)),("gsm8k",L(gsm))]:
            gk={} if task=="mmlu" else {"apply_chat_template":True,"fewshot_as_multiturn":True}
            out=simple_evaluate(model=lm,tasks=[task],limit=lim,bootstrap_iters=0,**gk);r=out["results"].get(task,{})
            acc=next(((k,r[k]) for k in ("acc,none","exact_match,strict-match","exact_match,flexible-extract") if k in r),None)
            res[nm][task]=acc;print(f"[{nm}] {task}: {acc}",flush=True)
        del lm
    print("\n===== SUMMARY =====");print(json.dumps(res,indent=2,default=str));print("LMEVAL_DONE",flush=True)

if MODE=="sanity":
    names=sys.argv[2].split(",") if len(sys.argv)>2 else list(CFGS.keys())
    sanity(names)
else:
    mmlu=float(sys.argv[2]);gsm=float(sys.argv[3]);bs=sys.argv[4];names=sys.argv[5].split(",")
    full(mmlu,gsm,bs,names)
