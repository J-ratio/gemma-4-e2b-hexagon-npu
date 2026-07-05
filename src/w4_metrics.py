#!/usr/bin/env python3
"""A16W4 (int4 weights, int16 activations): quantize BOTH graphs, compile, profile on S26 Ultra. Prints
infer latencies to compute TTFT / prefill tok-s / decode tok-s. (Latency is calibration-independent.)"""
import os, sys, numpy as np, onnx
GQ=os.environ.get("GQ_DIR",".")
from aimet_onnx.quantsim import QuantizationSimModel
import qai_hub as hub
DEV=hub.Device("Samsung Galaxy S26 Ultra")
SEQ=128; H=1536; PLD=256; NL=35; NC=15; CTX=4096
KVD=[(1,256)]*4+[(1,512)]+[(1,256)]*4+[(1,512)]+[(1,256)]*4+[(1,512)]
NEG=np.finfo(np.float32).min

def quant_compile_profile(src, out, mk_calib, name, extra_opts=""):
    m=onnx.load(src)
    sim=QuantizationSimModel(m, param_type="int4", activation_type="int16", quant_scheme="min_max",
                             dummy_input=mk_calib()[0], providers=["CPUExecutionProvider"])
    cal=mk_calib()
    def fp(session,*_):
        for x in cal: session.run(None,x)
    print(f"[{name}] compute_encodings (int4) ...",flush=True); sim.compute_encodings(fp)
    os.makedirs(out,exist_ok=True); sim.export(out, name)
    print(f"[{name}] exported",flush=True)
    cj=hub.submit_compile_job(model=out,device=DEV,
        options=f"--target_runtime qnn_context_binary --truncate_64bit_io {extra_opts}",name=f"{name}-htp")
    print(f"[{name}] COMPILE {cj.job_id} {cj.url}",flush=True); cj.wait()
    if not cj.get_status().success: print(f"[{name}] compile FAILED:",cj.get_status().message,flush=True); return
    pj=hub.submit_profile_job(model=cj.get_target_model(),device=DEV,name=f"{name}-prof")
    print(f"[{name}] PROFILE {pj.job_id} {pj.url}",flush=True); pj.wait()
    ex=pj.download_profile().get("execution_summary",{}); units={}
    for l in pj.download_profile().get("execution_detail",[]): u=l.get("compute_unit","?"); units[u]=units.get(u,0)+1
    print(f"[{name}] infer_us={ex.get('estimated_inference_time')} units={units}",flush=True)

def prefill_calib():
    def s():
        d={"inputs_embeds":np.random.randn(1,SEQ,H).astype(np.float32),
           "per_layer_inputs":np.random.randn(1,SEQ,NL,PLD).astype(np.float32),
           "position_ids":np.arange(SEQ,dtype=np.int64).reshape(1,SEQ)}
        j=np.arange(SEQ); f=np.where(j[None,:]<=j[:,None],0.0,NEG).astype(np.float32).reshape(1,1,SEQ,SEQ)
        d["full_mask"]=f; d["sliding_mask"]=f.copy(); return d
    return [s() for _ in range(2)]

def decode_calib():
    def s():
        d={"inputs_embeds":np.random.randn(1,1,H).astype(np.float32),
           "per_layer_inputs":np.random.randn(1,1,NL,PLD).astype(np.float32),
           "position_ids":np.array([[7]],np.int64),"cache_position":np.array([7],np.int64),
           "full_mask":np.zeros((1,1,1,CTX),np.float32),"sliding_mask":np.zeros((1,1,1,CTX),np.float32)}
        for i in range(NC):
            nkv,hd=KVD[i]; d[f"past_k_{i}"]=np.random.randn(1,nkv,CTX,hd).astype(np.float32); d[f"past_v_{i}"]=np.random.randn(1,nkv,CTX,hd).astype(np.float32)
        return d
    return [s() for _ in range(2)]

quant_compile_profile(f"{GQ}/trunk_onnx/model.onnx",f"{GQ}/trunk_A16W4",prefill_calib,"gemma4-prefill-A16W4")
quant_compile_profile(f"{GQ}/decode_fixed_onnx/model.onnx",f"{GQ}/decode_A16W4",decode_calib,"gemma4-decode-A16W4")
print("W4 DONE",flush=True)
