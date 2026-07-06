#!/usr/bin/env python3
"""WINDOWED-KV decode + BROADCAST-GQA attention: eliminates the materialized repeat_kv `expand`
(38% of our decode) by doing grouped attention with broadcasting matmuls (1 KV head vs 8 Q heads,
never copied to 8). Proven == HF greedy, then export. Usage: python decode_windowed_gqa.py [export]"""
import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
CTX=4096; WIN=512; H=1536; PLD=256; NL=35
OUT="/home/ubuntu/gq/decode_wgqa_onnx"; os.makedirs(OUT,exist_ok=True)
NEG=torch.finfo(torch.float32).min

tok=AutoTokenizer.from_pretrained("google/gemma-4-E2B-it", token=os.environ["HF_TOKEN"])
m=AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it", token=os.environ["HF_TOKEN"],
                                       torch_dtype=torch.float32, device_map="cpu").eval()
cfg=m.config.get_text_config(); tm=m.model.language_model
nshared=getattr(cfg,"num_kv_shared_layers",0); NC=NL-nshared
BUF=[WIN if cfg.layer_types[i]=="sliding_attention" else CTX for i in range(NC)]

# ---- CLEAN HF references FIRST (default attention, before any patching) ----
VAL_PROMPTS=[("The capital of France is",15),
             ("Count upward: "+" ".join(str(i) for i in range(600))+". The next number is",8)]
HF_REF={}
for p,nn in VAL_PROMPTS:
    _ids=tok(p,return_tensors="pt").input_ids
    with torch.no_grad(): _hf=m.generate(_ids,max_new_tokens=nn,do_sample=False)
    HF_REF[p]=_hf[0][_ids.shape[1]:].tolist()
    print(f"[ref len {_ids.shape[1]}] HF={tok.decode(HF_REF[p])!r}",flush=True)

# ---- broadcast-GQA attention interface (no repeat_kv / no expand) ----
def gqa_bcast(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kw):
    B,Hq,Lq,D=query.shape; Hkv=key.shape[1]; g=Hq//Hkv
    if scaling is None: scaling=D**-0.5
    q=query.reshape(B,Hkv,g,Lq,D)                 # group the query heads
    k=key.unsqueeze(2); v=value.unsqueeze(2)      # [B,Hkv,1,Lk,D] -> broadcasts over g (NO copy)
    scores=torch.matmul(q,k.transpose(-1,-2))*scaling            # [B,Hkv,g,Lq,Lk]
    if attention_mask is not None:
        scores=scores+attention_mask.unsqueeze(2)               # [B,1,1,Lq,Lk] broadcast
    attn=torch.softmax(scores,dim=-1)
    out=torch.matmul(attn,v)                                    # [B,Hkv,g,Lq,D]
    out=out.reshape(B,Hq,Lq,D).transpose(1,2).contiguous()      # [B,Lq,Hq,D]
    return out, None

try:
    from transformers import AttentionInterface
    AttentionInterface.register("gqa_bcast", gqa_bcast)
except Exception:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["gqa_bcast"]=gqa_bcast
m.config._attn_implementation="gqa_bcast"
cfg._attn_implementation="gqa_bcast"
for mod in m.modules():
    if hasattr(mod,"config") and hasattr(mod.config,"_attn_implementation"): mod.config._attn_implementation="gqa_bcast"
print("attn impl set to gqa_bcast", flush=True)

from transformers.cache_utils import DynamicCache
_c=DynamicCache()
with torch.no_grad(): tm(input_ids=torch.tensor([[2,105,2364,107]]), use_cache=True, past_key_values=_c)
KVD=[(int(_c.layers[i].keys.shape[1]), int(_c.layers[i].keys.shape[3])) for i in range(NC)]
print("NC",NC,"BUF",BUF,"KVD",KVD, flush=True)

class WinKV:
    def __init__(self,kbuf,vbuf,cpos): self.kbuf=list(kbuf);self.vbuf=list(vbuf);self.cpos=cpos
    def update(self,key,value,layer_idx,cache_kwargs=None):
        w=self.kbuf[layer_idx].shape[2]; wi=(self.cpos%w) if w<CTX else self.cpos
        self.kbuf[layer_idx]=self.kbuf[layer_idx].index_copy(2,wi,key)
        self.vbuf[layer_idx]=self.vbuf[layer_idx].index_copy(2,wi,value)
        return self.kbuf[layer_idx],self.vbuf[layer_idx]
    def get_seq_length(self,layer_idx=0): return int(CTX)
    def get_mask_sizes(self,cache_position,layer_idx):
        w=self.kbuf[layer_idx].shape[2] if layer_idx<len(self.kbuf) else CTX; return (w,w)
    def __len__(self): return NC

class Decode(torch.nn.Module):
    def __init__(self,tm,nc): super().__init__(); self.tm=tm; self.nc=nc
    def forward(self,inputs_embeds,per_layer_inputs,position_ids,cache_position,full_mask,sliding_mask,*kv):
        cache=WinKV([kv[2*i] for i in range(self.nc)],[kv[2*i+1] for i in range(self.nc)],cache_position)
        out=self.tm(inputs_embeds=inputs_embeds,per_layer_inputs=per_layer_inputs,position_ids=position_ids,
                    cache_position=cache_position,attention_mask={"full_attention":full_mask,"sliding_attention":sliding_mask},
                    past_key_values=cache,use_cache=True)
        outs=[out.last_hidden_state]
        for i in range(self.nc): outs+=[cache.kbuf[i],cache.vbuf[i]]
        return tuple(outs)

w=Decode(tm,NC).eval()
@torch.no_grad()
def embeds(ids):
    return tm.embed_tokens(ids), tm.embed_tokens_per_layer(ids).reshape(1,ids.shape[1],NL,PLD)
def masks(pos):
    jf=torch.arange(CTX); full=torch.where(jf<=pos,0.0,NEG).view(1,1,1,CTX)
    js=torch.arange(WIN); slide=(torch.zeros(WIN) if pos>=WIN-1 else torch.where(js<=pos,0.0,NEG)).view(1,1,1,WIN)
    return full,slide
def fresh_kv():
    kv=[]
    for i in range(NC):
        nkv,hd=KVD[i]; kv+=[torch.zeros(1,nkv,BUF[i],hd),torch.zeros(1,nkv,BUF[i],hd)]
    return kv
@torch.no_grad()
def greedy(prompt,n=15):
    ids=tok(prompt,return_tensors="pt").input_ids; seq=ids[0].tolist(); gen=seq[:]; kv=fresh_kv(); pos=0; nxt=[]
    for step in range(len(seq)+n):
        ie,ple=embeds(torch.tensor([[gen[step]]])); full,slide=masks(pos)
        res=w(ie,ple,torch.tensor([[pos]]),torch.tensor([pos]),full,slide,*kv)
        for i in range(NC): kv[2*i]=res[1+2*i]; kv[2*i+1]=res[2+2*i]
        pos+=1
        if step>=len(seq)-1:
            tid=int(m.lm_head(res[0])[0,-1].argmax()); nxt.append(tid)
            if len(gen)<=step+1: gen.append(tid)
    return nxt[:n]

# HF greedy stops on its eos; our argmax may emit a different special stop-token. Compare CONTENT only:
# truncate each sequence at its first special token, then require equality of the non-special prefix.
SPECIAL=set(tok.all_special_ids)
def content(x):
    out=[]
    for t in x:
        if t in SPECIAL: break
        out.append(t)
    return out
ALLOK=True
for p,nn in VAL_PROMPTS:
    hn=greedy(p,nn); ids=tok(p,return_tensors="pt").input_ids; hfn=HF_REF[p]
    ok=content(hn)==content(hfn)
    print(f"[len {ids.shape[1]}] WGQA={tok.decode(hn)!r}  HF={tok.decode(hfn)!r}  CONTENT_MATCH={ok}",flush=True)
    ALLOK=ALLOK and ok
print("CORRECT:",ALLOK,flush=True)

if len(sys.argv)>1 and sys.argv[1]=="export" and globals().get("ALLOK"):
    ie=torch.randn(1,1,H); pli=torch.randn(1,1,NL,PLD); pos=torch.tensor([[7]]); cpos=torch.tensor([7])
    full,slide=masks(7); kv=fresh_kv()
    names=["inputs_embeds","per_layer_inputs","position_ids","cache_position","full_mask","sliding_mask"]
    for i in range(NC): names+=[f"past_k_{i}",f"past_v_{i}"]
    onames=["hidden"]
    for i in range(NC): onames+=[f"present_k_{i}",f"present_v_{i}"]
    with torch.no_grad():
        torch.onnx.export(w,(ie,pli,pos,cpos,full,slide,*kv),f"{OUT}/model.onnx",
            input_names=names,output_names=onames,opset_version=18,dynamo=True)
    print("EXPORTED wgqa:",os.listdir(OUT),flush=True)
elif len(sys.argv)>1 and sys.argv[1]=="export":
    print("NOT exported (correctness failed)",flush=True)
