#!/usr/bin/env python3
"""AIMET A16W8 quantize of the clean trunk (no exclusions needed - no giant vocab tensors)."""
import os, numpy as np, onnx
GQ=os.environ.get("GQ_DIR",".")
from aimet_onnx.quantsim import QuantizationSimModel
IN=f"{GQ}/trunk_onnx/model.onnx"; OUT=f"{GQ}/trunk_A16W8"; os.makedirs(OUT,exist_ok=True)
SEQ,H,NL,PLD=128,1536,35,256
def sample():
    neg=np.finfo(np.float32).min
    full=np.triu(np.full((SEQ,SEQ),neg,dtype=np.float32),1).reshape(1,1,SEQ,SEQ)
    return {"inputs_embeds":np.random.randn(1,SEQ,H).astype(np.float32),
            "per_layer_inputs":np.random.randn(1,SEQ,NL,PLD).astype(np.float32),
            "position_ids":np.arange(SEQ,dtype=np.int64).reshape(1,SEQ),
            "full_mask":full,"sliding_mask":full.copy()}
calib=[sample() for _ in range(2)]
print("loading trunk onnx ...", flush=True)
m=onnx.load(IN)
print("building QuantizationSimModel (int8/int16, min_max) ...", flush=True)
sim=QuantizationSimModel(m, param_type="int8", activation_type="int16", quant_scheme="min_max",
                         dummy_input=calib[0], providers=["CPUExecutionProvider"])
def fp(session,*_):
    for i,x in enumerate(calib): session.run(None,x); print(f"  calib {i+1}/{len(calib)}",flush=True)
print("compute_encodings ...", flush=True)
sim.compute_encodings(fp)
print("exporting ...", flush=True)
sim.export(OUT,"trunk_A16W8")
print("EXPORTED:", os.listdir(OUT), flush=True)
sz=sum(os.path.getsize(os.path.join(OUT,f)) for f in os.listdir(OUT)); print("MB:",round(sz/1e6,1),flush=True)
