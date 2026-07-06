#!/usr/bin/env python3
"""Robust quantization-accuracy metric: TOP-1 TOKEN AGREEMENT vs fp32 (behavioral fidelity), plus top-5
overlap and mean KL. Not confounded by pathological-token PPL. Weight-only fake-quant. CPU, no NPU."""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
tok=AutoTokenizer.from_pretrained("google/gemma-4-E2B-it", token=os.environ["HF_TOKEN"])
m=AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it", token=os.environ["HF_TOKEN"],
                                       torch_dtype=torch.float32, device_map="cpu").eval()
ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
text="\n\n".join(t for t in ds["text"] if t.strip())
ids=tok(text, return_tensors="pt").input_ids[:, :2048]     # one BOS-led window
print("eval positions:", ids.shape[1]-1, flush=True)

@torch.no_grad()
def logits(model): return model(ids).logits[0,:-1].float()   # [N,V]

def fq(w,nbit,mode):
    q=2**(nbit-1)-1
    if mode=="tensor": s=(w.abs().max()/q).clamp_min(1e-12); return torch.clamp(torch.round(w/s),-q-1,q)*s
    if mode=="channel": s=(w.abs().amax(1,keepdim=True)/q).clamp_min(1e-12); return torch.clamp(torch.round(w/s),-q-1,q)*s
    g=128;O,I=w.shape
    if I%g: return fq(w,nbit,"channel")
    wr=w.reshape(O,I//g,g); s=(wr.abs().amax(-1,keepdim=True)/q).clamp_min(1e-12)
    return (torch.clamp(torch.round(wr/s),-q-1,q)*s).reshape(O,I)

with torch.no_grad():
    ref=logits(m); ref_top1=ref.argmax(-1); ref_top5=ref.topk(5,-1).indices
    ref_lp=torch.log_softmax(ref,-1)
tm=m.model.language_model
lin=[mod for _,mod in tm.layers.named_modules() if isinstance(mod,torch.nn.Linear)]
orig=[mod.weight.data.clone() for mod in lin]
print(f"{'config':24s} {'top1-agree':>11s} {'top5-in':>9s} {'meanKL':>9s}", flush=True)
for nbit,mode,label in [(8,"channel","A16W8 per-channel"),(4,"group","A16W4 per-group-128"),
                        (4,"channel","A16W4 per-channel"),(4,"tensor","A16W4 per-tensor")]:
    for mod,w0 in zip(lin,orig): mod.weight.data=fq(w0,nbit,mode)
    with torch.no_grad():
        lg=logits(m); t1=lg.argmax(-1)
        agree=(t1==ref_top1).float().mean().item()
        top5in=(t1.unsqueeze(-1)==ref_top5).any(-1).float().mean().item()
        kl=torch.nn.functional.kl_div(torch.log_softmax(lg,-1), ref_lp, log_target=True, reduction="batchmean").item()
    print(f"{label:24s} {100*agree:9.1f}% {100*top5in:8.1f}% {kl:9.4f}", flush=True)
    for mod,w0 in zip(lin,orig): mod.weight.data=w0.clone()
print("AGREE_DONE", flush=True)
