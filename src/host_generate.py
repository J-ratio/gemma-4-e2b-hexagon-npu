#!/usr/bin/env python3
"""End-to-end generation via the float decode graph (ONNX Runtime CPU) + a host KV loop, compared to HF greedy.
This validates the full design (KV mgmt + mask + embeddings + lm_head) and IS the deployable runtime logic.
Runs token-by-token through the seq=1 decode graph (prefill = decode over the prompt)."""
import os, numpy as np, torch, onnxruntime as ort
GQ=os.environ.get("GQ_DIR",".")
from transformers import AutoModelForCausalLM, AutoTokenizer

CTX=4096; NL=35; NC=15; PLD=256
KVD=[(1,256)]*4+[(1,512)]+[(1,256)]*4+[(1,512)]+[(1,256)]*4+[(1,512)]
tok=AutoTokenizer.from_pretrained("google/gemma-4-E2B-it", token=os.environ["HF_TOKEN"])
m=AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it", token=os.environ["HF_TOKEN"],
                                       torch_dtype=torch.float32, device_map="cpu").eval()
tm=m.model.language_model
NEG=np.finfo(np.float32).min

sess=ort.InferenceSession(f"{GQ}/decode_fixed_onnx/model.onnx",
                          providers=["CPUExecutionProvider"])

@torch.no_grad()
def host_embeds(ids):  # ids: LongTensor [1,1]
    ie=tm.embed_tokens(ids)                                   # scaled word embedding
    ple=tm.embed_tokens_per_layer(ids).reshape(1,ids.shape[1],NL,PLD)  # raw per-layer gather
    return ie.numpy().astype(np.float32), ple.numpy().astype(np.float32)

def masks(pos):  # additive [1,1,1,CTX]; attend 0..pos (full) / pos-511..pos (sliding)
    j=np.arange(CTX)
    full=np.where(j<=pos,0.0,NEG).astype(np.float32).reshape(1,1,1,CTX)
    slide=np.where((j<=pos)&(j>pos-512),0.0,NEG).astype(np.float32).reshape(1,1,1,CTX)
    return full,slide

def gen_host(prompt, n=20):
    ids=tok(prompt, return_tensors="pt").input_ids
    kv={f"past_k_{i}":np.zeros((1,KVD[i][0],CTX,KVD[i][1]),np.float32) for i in range(NC)}
    kv.update({f"past_v_{i}":np.zeros((1,KVD[i][0],CTX,KVD[i][1]),np.float32) for i in range(NC)})
    out_ids=ids[0].tolist(); pos=0; last_hidden=None
    seq=ids[0].tolist()
    for step in range(len(seq)+n):
        t=seq[step] if step<len(seq) else nxt
        ie,ple=host_embeds(torch.tensor([[t]]))
        full,slide=masks(pos)
        feeds={"inputs_embeds":ie,"per_layer_inputs":ple,
               "position_ids":np.array([[pos]],np.int64),"cache_position":np.array([pos],np.int64),
               "full_mask":full,"sliding_mask":slide,**kv}
        res=sess.run(None,feeds)
        hidden=res[0]  # [1,1,H]
        for i in range(NC): kv[f"past_k_{i}"]=res[1+2*i]; kv[f"past_v_{i}"]=res[2+2*i]
        pos+=1
        if step>=len(seq)-1:
            h=torch.tensor(hidden)
            logits=m.lm_head(h)[0,-1]
            nxt=int(logits.argmax())
            if step>=len(seq)-1 and step>=len(seq)-1: out_ids.append(nxt) if step>=len(seq) else None
    # regenerate cleanly: collect n greedy tokens
    return out_ids

# simpler correctness check: compare first K greedy next-tokens vs HF
prompt="The capital of France is"
ids=tok(prompt,return_tensors="pt").input_ids
kv={f"past_k_{i}":np.zeros((1,KVD[i][0],CTX,KVD[i][1]),np.float32) for i in range(NC)}
kv.update({f"past_v_{i}":np.zeros((1,KVD[i][0],CTX,KVD[i][1]),np.float32) for i in range(NC)})
seq=ids[0].tolist(); pos=0; host_next=[]
gen=seq[:]
for step in range(len(seq)+15):
    t=gen[step]
    ie,ple=host_embeds(torch.tensor([[t]]))
    full,slide=masks(pos)
    feeds={"inputs_embeds":ie,"per_layer_inputs":ple,"position_ids":np.array([[pos]],np.int64),
           "cache_position":np.array([pos],np.int64),"full_mask":full,"sliding_mask":slide,**kv}
    res=sess.run(None,feeds)
    for i in range(NC): kv[f"past_k_{i}"]=res[1+2*i]; kv[f"past_v_{i}"]=res[2+2*i]
    pos+=1
    if step>=len(seq)-1:
        nxt=int(m.lm_head(torch.tensor(res[0]))[0,-1].argmax())
        host_next.append(nxt)
        if step>=len(seq)-1 and len(gen)<=step+1: gen.append(nxt)
print("HOST  greedy:", tok.decode(host_next[:15]), flush=True)
# HF reference
with torch.no_grad():
    hf=m.generate(ids, max_new_tokens=15, do_sample=False)
print("HF    greedy:", tok.decode(hf[0][len(seq):]), flush=True)
print("MATCH:", host_next[:10]==hf[0][len(seq):len(seq)+10].tolist(), flush=True)
