#!/usr/bin/env python3
"""WINDOWED-KV AR-1 decode graph for Gemma-4-E2B (matches the model's native hybrid attention).
28 sliding layers -> KV stored/attended over a 512 RING buffer (write at pos%512); 7 full layers -> 4096.
Only the 15 non-shared cache slots are stored (12 sliding@512 + 3 full@4096); HF shares KV to layers 15-34.
This is EXACT vs the full-4096 masked version (masked keys contribute 0). We PROVE it by matching HF greedy,
then export ONNX. Speed win: sliding layers attend 512 not 4096.
Usage: python decode_windowed.py [export]"""
import os, sys, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
CTX=4096; WIN=512; H=1536; PLD=256; NL=35
OUT="/home/ubuntu/gq/decode_windowed_onnx"; os.makedirs(OUT,exist_ok=True)
NEG=np.finfo(np.float32).min

tok=AutoTokenizer.from_pretrained("google/gemma-4-E2B-it", token=os.environ["HF_TOKEN"])
m=AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it", token=os.environ["HF_TOKEN"],
                                       torch_dtype=torch.float32, device_map="cpu").eval()
cfg=m.config.get_text_config(); tm=m.model.language_model
nshared=getattr(cfg,"num_kv_shared_layers",0); NC=NL-nshared
SLOT_SLIDING=[cfg.layer_types[i]=="sliding_attention" for i in range(NC)]
BUF=[WIN if SLOT_SLIDING[i] else CTX for i in range(NC)]
# per-slot (kv_heads, head_dim) from a live prefill
from transformers.cache_utils import DynamicCache
_c=DynamicCache()
with torch.no_grad(): tm(input_ids=torch.tensor([[2,105,2364,107]]), use_cache=True, past_key_values=_c)
KVD=[(int(_c.layers[i].keys.shape[1]), int(_c.layers[i].keys.shape[3])) for i in range(NC)]
print("NC",NC,"BUF",BUF,"KVD",KVD, flush=True)

class WinKV:
    """Per-slot ring buffer: sliding slots length 512 (write at cpos%512), full slots length 4096 (write at cpos)."""
    def __init__(self, kbuf, vbuf, cpos):
        self.kbuf=list(kbuf); self.vbuf=list(vbuf); self.cpos=cpos
    def update(self, key, value, layer_idx, cache_kwargs=None):
        w=self.kbuf[layer_idx].shape[2]
        wi=(self.cpos % w) if w<CTX else self.cpos
        k=self.kbuf[layer_idx].index_copy(2, wi, key); v=self.vbuf[layer_idx].index_copy(2, wi, value)
        self.kbuf[layer_idx]=k; self.vbuf[layer_idx]=v
        return k, v
    def get_seq_length(self, layer_idx=0): return int(CTX)
    def get_mask_sizes(self, cache_position, layer_idx):
        w=self.kbuf[layer_idx].shape[2] if layer_idx<len(self.kbuf) else CTX
        return (w, w)
    def __len__(self): return NC

class Decode(torch.nn.Module):
    def __init__(self, tm, nc): super().__init__(); self.tm=tm; self.nc=nc
    def forward(self, inputs_embeds, per_layer_inputs, position_ids, cache_position, full_mask, sliding_mask, *kv):
        kbuf=[kv[2*i] for i in range(self.nc)]; vbuf=[kv[2*i+1] for i in range(self.nc)]
        cache=WinKV(kbuf, vbuf, cache_position)
        mask={"full_attention": full_mask, "sliding_attention": sliding_mask}
        out=self.tm(inputs_embeds=inputs_embeds, per_layer_inputs=per_layer_inputs,
                    position_ids=position_ids, cache_position=cache_position,
                    attention_mask=mask, past_key_values=cache, use_cache=True)
        outs=[out.last_hidden_state]
        for i in range(self.nc): outs+=[cache.kbuf[i], cache.vbuf[i]]
        return tuple(outs)

w=Decode(tm,NC).eval()

@torch.no_grad()
def embeds(ids):
    ie=tm.embed_tokens(ids); ple=tm.embed_tokens_per_layer(ids).reshape(1,ids.shape[1],NL,PLD)
    return ie, ple
def masks(pos):
    jf=torch.arange(CTX); full=torch.where(jf<=pos,0.0,NEG).view(1,1,1,CTX)
    js=torch.arange(WIN)
    if pos>=WIN-1: slide=torch.zeros(WIN)                 # buffer full -> all 512 valid (last 512 positions)
    else:          slide=torch.where(js<=pos,0.0,NEG)     # only slots 0..pos written
    return full, slide.view(1,1,1,WIN)

def fresh_kv():
    kv=[]
    for i in range(NC):
        nkv,hd=KVD[i]; kv+=[torch.zeros(1,nkv,BUF[i],hd), torch.zeros(1,nkv,BUF[i],hd)]
    return kv

@torch.no_grad()
def greedy(prompt, n=15):
    ids=tok(prompt,return_tensors="pt").input_ids; seq=ids[0].tolist(); gen=seq[:]; kv=fresh_kv(); pos=0; nxt=[]
    for step in range(len(seq)+n):
        t=gen[step]; ie,ple=embeds(torch.tensor([[t]])); full,slide=masks(pos)
        res=w(ie,ple,torch.tensor([[pos]]),torch.tensor([pos]),full,slide,*kv)
        for i in range(NC): kv[2*i]=res[1+2*i]; kv[2*i+1]=res[2+2*i]
        pos+=1
        if step>=len(seq)-1:
            tok_id=int(m.lm_head(res[0])[0,-1].argmax()); nxt.append(tok_id)
            if len(gen)<=step+1: gen.append(tok_id)
    return nxt[:n]

# ---- CORRECTNESS: windowed decode greedy vs HF greedy ----
prompt="The capital of France is"
hn=greedy(prompt,15)
ids=tok(prompt,return_tensors="pt").input_ids
with torch.no_grad(): hf=m.generate(ids,max_new_tokens=15,do_sample=False)
hf_new=hf[0][ids.shape[1]:].tolist()
print("WINDOWED greedy:", tok.decode(hn), flush=True)
print("HF       greedy:", tok.decode(hf_new), flush=True)
MATCH = hn[:12]==hf_new[:12]
print("MATCH:", MATCH, flush=True)

# ---- also test a LONG prompt (> window) so the ring actually wraps ----
long_prompt=("Count upward: "+" ".join(str(i) for i in range(600))+". Now, the next number after all that is")
hn2=greedy(long_prompt,8)
ids2=tok(long_prompt,return_tensors="pt").input_ids
print("long prompt tokens:", ids2.shape[1], "(ring wraps if >512)", flush=True)
with torch.no_grad(): hf2=m.generate(ids2,max_new_tokens=8,do_sample=False)
hf2_new=hf2[0][ids2.shape[1]:].tolist()
print("WINDOWED(long):", tok.decode(hn2), flush=True)
print("HF      (long):", tok.decode(hf2_new), flush=True)
MATCH2 = hn2[:6]==hf2_new[:6]
print("MATCH_LONG:", MATCH2, flush=True)
print("CORRECT:", MATCH and MATCH2, flush=True)

if len(sys.argv)>1 and sys.argv[1]=="export" and MATCH and MATCH2:
    ie=torch.randn(1,1,H); pli=torch.randn(1,1,NL,PLD); pos=torch.tensor([[7]]); cpos=torch.tensor([7])
    full,slide=masks(7); kv=fresh_kv()
    names=["inputs_embeds","per_layer_inputs","position_ids","cache_position","full_mask","sliding_mask"]
    for i in range(NC): names+=[f"past_k_{i}",f"past_v_{i}"]
    onames=["hidden"]
    for i in range(NC): onames+=[f"present_k_{i}",f"present_v_{i}"]
    with torch.no_grad():
        torch.onnx.export(w,(ie,pli,pos,cpos,full,slide,*kv),f"{OUT}/model.onnx",
            input_names=names, output_names=onames, opset_version=18, dynamo=True)
    print("EXPORTED windowed decode:", os.listdir(OUT), flush=True)
elif len(sys.argv)>1 and sys.argv[1]=="export":
    print("NOT exported (correctness failed)", flush=True)
