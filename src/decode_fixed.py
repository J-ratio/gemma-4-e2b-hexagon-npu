#!/usr/bin/env python3
"""Reusable AR-1 decode graph: FIXED 4096 KV buffers, in-place write at cache_position (tensor input).
Custom cache reads cache_position as a TENSOR (not Python state) -> exportable + reusable every step.
All layers use full 4096 buffers; sliding-window enforced by the fed mask. Tests if QNN accepts the scatter."""
import os, sys, torch
GQ=os.environ.get("GQ_DIR",".")
from transformers import AutoModelForCausalLM
CTX=4096; H=1536; PLD=256; NL=35
OUT=f"{GQ}/decode_fixed_onnx"; os.makedirs(OUT,exist_ok=True)

m=AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it", token=os.environ["HF_TOKEN"],
                                       torch_dtype=torch.float32, device_map="cpu").eval()
cfg=m.config.get_text_config(); tm=m.model.language_model
nshared=getattr(cfg,"num_kv_shared_layers",0); NC=NL-nshared
# per-layer (kv_heads, head_dim) from a live prefill
from transformers.cache_utils import DynamicCache
_c=DynamicCache()
with torch.no_grad(): tm(input_ids=torch.tensor([[2,105,2364,107]]), use_cache=True, past_key_values=_c)
KVD=[(int(_c.layers[i].keys.shape[1]), int(_c.layers[i].keys.shape[3])) for i in range(NC)]
print("NC",NC,"KVD",KVD, flush=True)

class FixedKV:
    """Minimal cache: fixed [1,nkv,CTX,hd] buffers per non-shared layer; write at cache_position tensor."""
    def __init__(self, kbuf, vbuf, cpos):
        self.kbuf=list(kbuf); self.vbuf=list(vbuf); self.cpos=cpos  # cpos: LongTensor[1]
    def update(self, key, value, layer_idx, cache_kwargs=None):
        k=self.kbuf[layer_idx].index_copy(2, self.cpos, key)
        v=self.vbuf[layer_idx].index_copy(2, self.cpos, value)
        self.kbuf[layer_idx]=k; self.vbuf[layer_idx]=v
        return k, v
    def get_seq_length(self, layer_idx=0): return int(CTX)
    def get_mask_sizes(self, cache_position, layer_idx): return (CTX, CTX)
    def __len__(self): return NC

class Decode(torch.nn.Module):
    def __init__(self, tm, nc): super().__init__(); self.tm=tm; self.nc=nc
    def forward(self, inputs_embeds, per_layer_inputs, position_ids, cache_position, full_mask, sliding_mask, *kv):
        kbuf=[kv[2*i] for i in range(self.nc)]; vbuf=[kv[2*i+1] for i in range(self.nc)]
        cache=FixedKV(kbuf, vbuf, cache_position)
        mask={"full_attention": full_mask, "sliding_attention": sliding_mask}
        out=self.tm(inputs_embeds=inputs_embeds, per_layer_inputs=per_layer_inputs,
                    position_ids=position_ids, cache_position=cache_position,
                    attention_mask=mask, past_key_values=cache, use_cache=True)
        outs=[out.last_hidden_state]
        for i in range(self.nc): outs+=[cache.kbuf[i], cache.vbuf[i]]
        return tuple(outs)

w=Decode(tm,NC).eval()
ie=torch.randn(1,1,H); pli=torch.randn(1,1,NL,PLD)
pos=torch.tensor([[7]]); cpos=torch.tensor([7])
full=torch.zeros(1,1,1,CTX); slide=torch.zeros(1,1,1,CTX)
kv=[]
for i in range(NC):
    nkv,hd=KVD[i]; kv+=[torch.randn(1,nkv,CTX,hd), torch.randn(1,nkv,CTX,hd)]
print("dry-run fixed-buffer decode ...", flush=True)
with torch.no_grad():
    y=w(ie,pli,pos,cpos,full,slide,*kv)
print("forward OK. hidden:", tuple(y[0].shape), "present_k0:", tuple(y[1].shape), flush=True)
names=["inputs_embeds","per_layer_inputs","position_ids","cache_position","full_mask","sliding_mask"]
for i in range(NC): names+=[f"past_k_{i}",f"past_v_{i}"]
onames=["hidden"]
for i in range(NC): onames+=[f"present_k_{i}",f"present_v_{i}"]
with torch.no_grad():
    torch.onnx.export(w,(ie,pli,pos,cpos,full,slide,*kv),f"{OUT}/model.onnx",
        input_names=names, output_names=onames, opset_version=18, dynamo=True)
print("EXPORTED fixed decode:", os.listdir(OUT), flush=True)
