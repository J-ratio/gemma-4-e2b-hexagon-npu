#!/usr/bin/env python3
"""Clean trunk export: drive Gemma4TextModel.forward with inputs_embeds + per_layer_inputs + a precomputed
mask DICT (skips create_causal_mask) + position_ids. No embedding gathers, no mask/RoPE-construction subgraph,
no lm_head -> no >2GB tensors and none of the fatal Concat/Sub ops. Output = final hidden states."""
import os, torch
GQ=os.environ.get("GQ_DIR",".")
from transformers import AutoModelForCausalLM
REPO="google/gemma-4-E2B-it"; SEQ=128; OUT=f"{GQ}/trunk_onnx"; os.makedirs(OUT, exist_ok=True)

m=AutoModelForCausalLM.from_pretrained(REPO, token=os.environ["HF_TOKEN"], torch_dtype=torch.float32, device_map="cpu").eval()
tm=m
for a in ("model","language_model"):
    if hasattr(tm,a): tm=getattr(tm,a)
cfg=m.config.get_text_config()
H=cfg.hidden_size; PLI_TOTAL=cfg.num_hidden_layers*cfg.hidden_size_per_layer_input
print("trunk:", type(tm).__name__, "H",H,"per_layer_total",PLI_TOTAL, flush=True)

class Trunk(torch.nn.Module):
    def __init__(self, tm): super().__init__(); self.tm=tm
    def forward(self, inputs_embeds, per_layer_inputs, position_ids, full_mask, sliding_mask):
        mask={"full_attention": full_mask, "sliding_attention": sliding_mask}
        out=self.tm(inputs_embeds=inputs_embeds, per_layer_inputs=per_layer_inputs,
                    position_ids=position_ids, attention_mask=mask, use_cache=False)
        return out.last_hidden_state

w=Trunk(tm).eval()
ie=torch.randn(1,SEQ,H)
pli=torch.randn(1,SEQ,cfg.num_hidden_layers,cfg.hidden_size_per_layer_input)
pos=torch.arange(SEQ).unsqueeze(0)
neg=torch.finfo(torch.float32).min
full=torch.triu(torch.full((SEQ,SEQ),neg),1).view(1,1,SEQ,SEQ)
slide=full.clone()
with torch.no_grad():
    y=w(ie,pli,pos,full,slide)
print("forward OK, out:", tuple(y.shape), flush=True)
with torch.no_grad():
    torch.onnx.export(w,(ie,pli,pos,full,slide),f"{OUT}/model.onnx",
        input_names=["inputs_embeds","per_layer_inputs","position_ids","full_mask","sliding_mask"],
        output_names=["hidden"], opset_version=18, dynamo=True)
print("EXPORTED trunk:", os.listdir(OUT), flush=True)
sz=sum(os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT)); print("onnx MB:", round(sz/1e6,1), flush=True)
