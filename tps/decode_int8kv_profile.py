#!/usr/bin/env python3
"""int8-KV on the WINDOWED decode graph: store the KV cache reads at int8 (half bandwidth) to speed the
full-attention layers (now ~80% of decode attention). A16W4 weights/acts unchanged. S26 Ultra Hexagon.
Variants: int8kv_full (only the 3 full-attn slots 4,9,14) and int8kv_all (all 15 slots). Latency is
calibration-independent, so this measures the bandwidth win directly. Baselines: full-KV 210177us, windowed 100979us."""
import os, numpy as np, onnx
from aimet_onnx.quantsim import QuantizationSimModel
import qai_hub as hub
DEV=hub.Device("Samsung Galaxy S26 Ultra")
H=1536; PLD=256; NL=35; NC=15; CTX=4096; WIN=512
KVD=[(1,256)]*4+[(1,512)]+[(1,256)]*4+[(1,512)]+[(1,256)]*4+[(1,512)]
BUF=[WIN]*4+[CTX]+[WIN]*4+[CTX]+[WIN]*4+[CTX]
FULL_SLOTS=[4,9,14]

def win_calib():
    def s():
        d={"inputs_embeds":np.random.randn(1,1,H).astype(np.float32),
           "per_layer_inputs":np.random.randn(1,1,NL,PLD).astype(np.float32),
           "position_ids":np.array([[7]],np.int64),"cache_position":np.array([7],np.int64),
           "full_mask":np.zeros((1,1,1,CTX),np.float32),"sliding_mask":np.zeros((1,1,1,WIN),np.float32)}
        for i in range(NC):
            nkv,hd=KVD[i];d[f"past_k_{i}"]=np.random.randn(1,nkv,BUF[i],hd).astype(np.float32);d[f"past_v_{i}"]=np.random.randn(1,nkv,BUF[i],hd).astype(np.float32)
        return d
    return [s() for _ in range(2)]

def qcp(out,name,int8_slots):
    m=onnx.load("/home/ubuntu/gq/decode_windowed_onnx/model.onnx")
    calib=win_calib()
    sim=QuantizationSimModel(m,param_type="int4",activation_type="int16",quant_scheme="min_max",
                             dummy_input=calib[0],providers=["CPUExecutionProvider"])
    qd=sim.qc_quantize_op_dict
    names=set()
    for s in int8_slots:
        for pfx in ("past_k_","past_v_","present_k_","present_v_"):
            t=f"{pfx}{s}"
            if t in qd: qd[t].set_bitwidth(8); names.add(t)
    print(f"[{name}] set int8 on {len(names)} KV tensors: {sorted(names)}",flush=True)
    def fp(sess,*_):
        for x in calib: sess.run(None,x)
    print(f"[{name}] compute_encodings ...",flush=True); sim.compute_encodings(fp)
    os.makedirs(out,exist_ok=True); sim.export(out,name); print(f"[{name}] exported",flush=True)
    cj=hub.submit_compile_job(model=out,device=DEV,
        options="--target_runtime qnn_context_binary --truncate_64bit_io",name=f"{name}-htp")
    print(f"[{name}] COMPILE {cj.job_id} {cj.url}",flush=True); cj.wait()
    if not cj.get_status().success: print(f"[{name}] COMPILE FAILED:",cj.get_status().message,flush=True); return None
    pj=hub.submit_profile_job(model=cj.get_target_model(),device=DEV,name=f"{name}-prof")
    print(f"[{name}] PROFILE {pj.job_id} {pj.url}",flush=True); pj.wait()
    prof=pj.download_profile(); ex=prof.get("execution_summary",{}); units={}
    for l in prof.get("execution_detail",[]): u=l.get("compute_unit","?"); units[u]=units.get(u,0)+1
    us=ex.get("estimated_inference_time"); print(f"[{name}] >>> infer_us={us} units={units}",flush=True); return us

r_full=qcp("/home/ubuntu/gq/decode_int8kv_full","decode-WIN-int8kvFULL-A16W4",FULL_SLOTS)
r_all =qcp("/home/ubuntu/gq/decode_int8kv_all","decode-WIN-int8kvALL-A16W4",list(range(NC)))
print("\n===== int8-KV RESULT (windowed base = 100979us, full-KV = 210177us) =====",flush=True)
print(f"windowed + int8-KV (full-attn slots only): {r_full} us",flush=True)
print(f"windowed + int8-KV (all slots)           : {r_all} us",flush=True)
if r_full: print(f"  vs windowed(int16 KV) 100979: {100979/r_full:.2f}x ; vs full-KV 210177: {210177/r_full:.2f}x",flush=True)
if r_all:  print(f"  all-int8 vs full-KV 210177: {210177/r_all:.2f}x",flush=True)
print("INT8KV_DONE",flush=True)
