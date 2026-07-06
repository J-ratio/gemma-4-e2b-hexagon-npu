#!/usr/bin/env python3
"""Per-layer & per-projection QUANT SENSITIVITY for Gemma-4-E2B.
Quantize ONE decoder layer (or one projection type across all layers) at a time to W4/W8,
measure output-logit KL(fp32 || quant) + top-1 agreement drop. Identifies which layers/projections
to keep at higher precision (mixed-precision map). Weight-only fake-quant, CPU, no NPU.
Uses the CHAT TEMPLATE (this is an -it model) so inputs are in-distribution."""
import os, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/gq/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_num_threads(os.cpu_count())
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL="google/gemma-4-E2B-it"
tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cpu").eval()

# In-distribution eval text via chat template
PROMPTS=[
 "Explain how photosynthesis works in simple terms.",
 "What is the capital of France, and name two famous landmarks there.",
 "Write a short paragraph about why quantization matters for running models on phones.",
 "Solve: a train travels 60 km in 1.5 hours. What is its average speed?",
 "List three differences between machine learning and traditional programming.",
]
ids_list=[]
for p in PROMPTS:
    msg=[{"role":"user","content":p}]
    t=tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
    ids_list.append(t)
# concat into one stream capped at N positions
ids=torch.cat([t[:, :] for t in ids_list], dim=1)[:, :512]
print("eval positions:", ids.shape[1], flush=True)

@torch.no_grad()
def logits(): return m(ids).logits[0,:-1].float()   # [N,V]

def fq(w,nbit,mode):
    q=2**(nbit-1)-1
    if mode=="tensor": s=(w.abs().max()/q).clamp_min(1e-12); return torch.clamp(torch.round(w/s),-q-1,q)*s
    if mode=="channel": s=(w.abs().amax(1,keepdim=True)/q).clamp_min(1e-12); return torch.clamp(torch.round(w/s),-q-1,q)*s
    g=128;O,I=w.shape
    if I%g: return fq(w,nbit,"channel")
    wr=w.reshape(O,I//g,g); s=(wr.abs().amax(-1,keepdim=True)/q).clamp_min(1e-12)
    return (torch.clamp(torch.round(wr/s),-q-1,q)*s).reshape(O,I)

with torch.no_grad():
    ref=logits(); ref_top1=ref.argmax(-1); ref_lp=torch.log_softmax(ref,-1)

def kl_agree():
    with torch.no_grad():
        lg=logits()
        kl=torch.nn.functional.kl_div(torch.log_softmax(lg,-1), ref_lp, log_target=True, reduction="batchmean").item()
        agree=(lg.argmax(-1)==ref_top1).float().mean().item()
    return kl, agree

tm=m.model.language_model
NL=len(tm.layers)
# gather (module, original_weight) grouped by layer index and by projection suffix
by_layer=[[] for _ in range(NL)]
by_type={}
for li,layer in enumerate(tm.layers):
    for name,mod in layer.named_modules():
        if isinstance(mod, torch.nn.Linear):
            by_layer[li].append(mod)
            suf=name.split(".")[-1]          # q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj/...
            by_type.setdefault(suf,[]).append(mod)
orig={id(mod):mod.weight.data.clone() for grp in by_layer for mod in grp}
print("layers:", NL, "| projection types:", sorted(by_type), flush=True)

def set_q(mods, nbit, mode):
    for mod in mods: mod.weight.data=fq(orig[id(mod)], nbit, mode)
def restore(mods):
    for mod in mods: mod.weight.data=orig[id(mod)].clone()

# ---- full-model baselines ----
print("\n=== FULL-MODEL baselines (all layers quantized) ===", flush=True)
allm=[mod for grp in by_layer for mod in grp]
for nbit,mode,label in [(8,"channel","A16W8 per-channel"),(4,"group","A16W4 per-group-128")]:
    set_q(allm,nbit,mode); kl,ag=kl_agree(); restore(allm)
    print(f"  {label:22s}  KL={kl:8.4f}  top1={100*ag:5.1f}%", flush=True)

# ---- per-projection-type sensitivity (W4, all layers of that type) ----
print("\n=== PER-PROJECTION-TYPE (W4 per-group-128, that type across ALL layers) ===", flush=True)
res=[]
for suf,mods in by_type.items():
    set_q(mods,4,"group"); kl,ag=kl_agree(); restore(mods)
    res.append((suf,kl,ag))
for suf,kl,ag in sorted(res,key=lambda x:-x[1]):
    print(f"  {suf:24s}  KL={kl:8.4f}  top1={100*ag:5.1f}%", flush=True)

# ---- per-layer sensitivity (W4 per-group-128, one layer at a time) ----
print("\n=== PER-LAYER (W4 per-group-128, one decoder layer at a time) ===", flush=True)
lay=[]
for li in range(NL):
    set_q(by_layer[li],4,"group"); kl,ag=kl_agree(); restore(by_layer[li])
    lay.append((li,kl,ag))
    print(f"  layer {li:2d}  KL={kl:8.4f}  top1={100*ag:5.1f}%", flush=True)

print("\n=== MOST SENSITIVE LAYERS (W4), top 10 by KL ===", flush=True)
for li,kl,ag in sorted(lay,key=lambda x:-x[1])[:10]:
    print(f"  layer {li:2d}  KL={kl:8.4f}  top1={100*ag:5.1f}%", flush=True)
print("SENS_DONE", flush=True)
