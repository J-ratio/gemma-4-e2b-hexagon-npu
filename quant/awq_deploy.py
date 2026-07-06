#!/usr/bin/env python3
"""Deploy AWQ at PER-CHANNEL int4 (the granularity HTP compiles) — the deployable form of awq_g64c.
AWQ per-input-channel scale s folds EXACTLY: W->W*s (quantized int4), x->x/s (int16, ~lossless), so output
is unchanged but salient channels survive int4 rounding. Zero new expensive ops (x/s is one elementwise Mul).
Builds TWO trunks: awq-scaled and plain-baseline. Re-exports each to ONNX for AIMET per-channel int4 -> QAIRT.
Stage 'build' = compute scales, fold, re-export both trunks (CPU). Then quantize+compile+profile separately.
"""
import os, sys, torch, numpy as np
os.environ.setdefault("HF_HOME","/home/ubuntu/gq/hf_cache")
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
SEQ=128; H=1536; PLD=256; NL=35
tok=AutoTokenizer.from_pretrained("google/gemma-4-E2B-it", token=os.environ["HF_TOKEN"])
m=AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it", token=os.environ["HF_TOKEN"],
                                       torch_dtype=torch.float32, device_map="cpu").eval()
cfg=m.config.get_text_config(); tm=m.model.language_model

# --- collect Hessian diag (activation salience) per Linear via calibration ---
def collect_salience(nseq=16, sl=512):
    lins=[(nm,mod) for nm,mod in tm.layers.named_modules() if isinstance(mod,torch.nn.Linear)]
    S={id(mod):torch.zeros(mod.weight.shape[1]) for _,mod in lins}
    hks=[]
    def mk(mod):
        def h(module,inp):
            x=inp[0].reshape(-1,inp[0].shape[-1]).float(); S[id(mod)]+=(x*x).sum(0)
        return h
    for _,mod in lins: hks.append(mod.register_forward_pre_hook(mk(mod)))
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")
    text="\n\n".join(t for t in ds["text"] if t.strip()); enc=tok(text,return_tensors="pt").input_ids[0]
    n=0;pos=0
    with torch.no_grad():
        while n<nseq and pos+sl<enc.numel(): m(enc[pos:pos+sl].unsqueeze(0),use_cache=False); pos+=sl; n+=1
    for h in hks: h.remove()
    return {k:v/max(n,1) for k,v in S.items()}, lins

def awq_scale(W, sal, nbit=4):
    """search alpha; per-channel int4 fake-quant of W*s, minimize salience-weighted error of W after unscale."""
    d=sal.clamp_min(1e-8); sa=d.sqrt(); best=None; q=2**(nbit-1)-1
    for a in [i/10 for i in range(11)]:
        s=(sa**a); s=(s/s.mean().clamp_min(1e-8)).clamp_min(1e-4)
        Ws=W*s[None,:]
        sc=(Ws.abs().amax(1,keepdim=True)/q).clamp_min(1e-12)         # PER-CHANNEL scale
        Wq=torch.clamp(torch.round(Ws/sc),-q-1,q)*sc
        E=Wq/s[None,:]-W; loss=(d[None,:]*E*E).sum().item()
        if best is None or loss<best[0]: best=(loss,s.clone())
    return best[1]

STAGE=sys.argv[1] if len(sys.argv)>1 else "build"
if STAGE=="build":
    print("collecting salience ...",flush=True)
    SAL,lins=collect_salience()
    print(f"{len(lins)} linears",flush=True)
    # compute + FOLD awq scales into weights, and register input-div pre-hooks (become Mul in ONNX)
    HOOKS=[]
    for nm,mod in lins:
        s=awq_scale(mod.weight.data, SAL[id(mod)])
        mod.weight.data=mod.weight.data*s[None,:]
        inv=(1.0/s)
        def mk(v):
            def h(module,args): return (args[0]*v.to(args[0].dtype),)+tuple(args[1:])
            return h
        HOOKS.append(mod.register_forward_pre_hook(mk(inv)))
    print("AWQ scales folded (W*=s, input*=1/s hooks live)",flush=True)

    # --- re-export the AWQ trunk (same dict-mask trunk as reexport_trunk.py) ---
    class Trunk(torch.nn.Module):
        def __init__(self,tm): super().__init__(); self.tm=tm
        def forward(self, inputs_embeds, per_layer_inputs, position_ids, full_mask, sliding_mask):
            out=self.tm(inputs_embeds=inputs_embeds, per_layer_inputs=per_layer_inputs, position_ids=position_ids,
                        attention_mask={"full_attention":full_mask,"sliding_attention":sliding_mask}, use_cache=False)
            return out.last_hidden_state
    w=Trunk(tm).eval()
    ie=torch.randn(1,SEQ,H); pli=torch.randn(1,SEQ,NL,PLD); pos=torch.arange(SEQ).unsqueeze(0)
    neg=torch.finfo(torch.float32).min; full=torch.triu(torch.full((SEQ,SEQ),neg),1).view(1,1,SEQ,SEQ)
    OUT="/home/ubuntu/gq/trunk_awq_onnx"; os.makedirs(OUT,exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(w,(ie,pli,pos,full,full.clone()),f"{OUT}/model.onnx",
            input_names=["inputs_embeds","per_layer_inputs","position_ids","full_mask","sliding_mask"],
            output_names=["hidden"], opset_version=18, dynamo=True)
    print("EXPORTED awq trunk:",os.listdir(OUT),flush=True)
    print("BUILD_DONE",flush=True)
