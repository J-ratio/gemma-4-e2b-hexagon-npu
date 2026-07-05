#!/usr/bin/env python3
"""Quantize the reusable fixed-buffer decode graph A16W8 -> compile -> profile on S26 Ultra."""
import os, sys, numpy as np, onnx
GQ=os.environ.get("GQ_DIR",".")
from aimet_onnx.quantsim import QuantizationSimModel
import qai_hub as hub
IN=f"{GQ}/decode_fixed_onnx/model.onnx"; OUT=f"{GQ}/decode_A16W8"; os.makedirs(OUT,exist_ok=True)
CTX=4096; H=1536; PLD=256; NL=35; NC=15
KVD=[(1,256)]*4+[(1,512)]+[(1,256)]*4+[(1,512)]+[(1,256)]*4+[(1,512)]  # 12 sliding(256)+3 full(512)
DEV=hub.Device("Samsung Galaxy S26 Ultra")

def sample():
    d={"inputs_embeds":np.random.randn(1,1,H).astype(np.float32),
       "per_layer_inputs":np.random.randn(1,1,NL,PLD).astype(np.float32),
       "position_ids":np.array([[7]],dtype=np.int64),
       "cache_position":np.array([7],dtype=np.int64),
       "full_mask":np.zeros((1,1,1,CTX),np.float32),
       "sliding_mask":np.zeros((1,1,1,CTX),np.float32)}
    for i in range(NC):
        nkv,hd=KVD[i]
        d[f"past_k_{i}"]=np.random.randn(1,nkv,CTX,hd).astype(np.float32)
        d[f"past_v_{i}"]=np.random.randn(1,nkv,CTX,hd).astype(np.float32)
    return d
calib=[sample() for _ in range(2)]
print("loading decode onnx ...",flush=True)
m=onnx.load(IN)
print("QuantizationSimModel ...",flush=True)
sim=QuantizationSimModel(m,param_type="int8",activation_type="int16",quant_scheme="min_max",
                         dummy_input=calib[0],providers=["CPUExecutionProvider"])
def fp(session,*_):
    for i,x in enumerate(calib): session.run(None,x); print(f"  calib {i+1}",flush=True)
print("compute_encodings ...",flush=True); sim.compute_encodings(fp)
print("export ...",flush=True); sim.export(OUT,"decode_A16W8")
print("EXPORTED:",os.listdir(OUT),flush=True)
cj=hub.submit_compile_job(model=OUT,device=DEV,
    options="--target_runtime qnn_context_binary --truncate_64bit_io",name="gemma4-decode-A16W8-htp")
print(f"COMPILE {cj.job_id} {cj.url}",flush=True)
cj.wait(); cst=cj.get_status()
if not cst.success: print(f"compile FAILED: {getattr(cst,'message','')}",flush=True); sys.exit(1)
print(f"QNN binary: {cj.get_target_model().name}",flush=True)
pj=hub.submit_profile_job(model=cj.get_target_model(),device=DEV,name="gemma4-decode-A16W8-profile")
print(f"PROFILE {pj.job_id} {pj.url}",flush=True)
pj.wait(); pst=pj.get_status()
if pst.success:
    prof=pj.download_profile(); ex=prof.get("execution_summary",{}); units={}
    for l in prof.get("execution_detail",[]): u=l.get("compute_unit","?"); units[u]=units.get(u,0)+1
    print(f"ON-DEVICE infer_us={ex.get('estimated_inference_time')} units={units}",flush=True)
else: print(f"profile FAILED: {getattr(pst,'message','')}",flush=True)
