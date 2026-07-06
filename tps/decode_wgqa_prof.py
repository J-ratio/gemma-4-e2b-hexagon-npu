#!/usr/bin/env python3
"""Profile the WINDOWED + BROADCAST-GQA decode graph (no expand op) on S26 Ultra Hexagon.
Two configs: A16W4 plain, and A16W4 + int8-KV on the 3 full-attn slots (best from prior sweep).
Baselines: full-KV 210177us | windowed 100979us | windowed+int8kv 92788us. Qualcomm NPU official ~18.2 tps (~55ms)."""
import os, numpy as np, onnx
from aimet_onnx.quantsim import QuantizationSimModel
import qai_hub as hub
DEV=hub.Device("Samsung Galaxy S26 Ultra")
SRC="/home/ubuntu/gq/decode_wgqa_onnx/model.onnx"
H=1536; PLD=256; NL=35; NC=15; CTX=4096; WIN=512
KVD=[(1,256)]*4+[(1,512)]+[(1,256)]*4+[(1,512)]+[(1,256)]*4+[(1,512)]
BUF=[WIN]*4+[CTX]+[WIN]*4+[CTX]+[WIN]*4+[CTX]
FULL_SLOTS=[4,9,14]

def calib():
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
    m=onnx.load(SRC); cal=calib()
    sim=QuantizationSimModel(m,param_type="int4",activation_type="int16",quant_scheme="min_max",
                             dummy_input=cal[0],providers=["CPUExecutionProvider"])
    qd=sim.qc_quantize_op_dict; nset=0
    for s in int8_slots:
        for pfx in ("past_k_","past_v_","present_k_","present_v_"):
            t=f"{pfx}{s}"
            if t in qd: qd[t].set_bitwidth(8); nset+=1
    print(f"[{name}] int8 KV tensors set: {nset}",flush=True)
    def fp(sess,*_):
        for x in cal: sess.run(None,x)
    print(f"[{name}] compute_encodings ...",flush=True); sim.compute_encodings(fp)
    os.makedirs(out,exist_ok=True); sim.export(out,name); print(f"[{name}] exported",flush=True)
    cj=hub.submit_compile_job(model=out,device=DEV,options="--target_runtime qnn_context_binary --truncate_64bit_io",name=f"{name}-htp")
    print(f"[{name}] COMPILE {cj.job_id} {cj.url}",flush=True); cj.wait()
    if not cj.get_status().success: print(f"[{name}] COMPILE FAILED:",cj.get_status().message,flush=True); return None
    pj=hub.submit_profile_job(model=cj.get_target_model(),device=DEV,name=f"{name}-prof")
    print(f"[{name}] PROFILE {pj.job_id} {pj.url}",flush=True); pj.wait()
    prof=pj.download_profile(); ex=prof.get("execution_summary",{}); units={}
    for l in prof.get("execution_detail",[]): u=l.get("compute_unit","?"); units[u]=units.get(u,0)+1
    us=ex.get("estimated_inference_time"); print(f"[{name}] >>> infer_us={us} units={units}",flush=True); return us

a=qcp("/home/ubuntu/gq/wgqa_a16w4","decode-WGQA-A16W4",[])
b=qcp("/home/ubuntu/gq/wgqa_a16w4_int8kv","decode-WGQA-int8kvFULL-A16W4",FULL_SLOTS)
print("\n===== WGQA RESULT (expand removed) =====",flush=True)
print(f"WGQA A16W4            : {a} us  ({1e6/a if a else 0:.1f} tps)",flush=True)
print(f"WGQA A16W4 + int8-KV  : {b} us  ({1e6/b if b else 0:.1f} tps)",flush=True)
print("baselines: full-KV 210177 (4.8) | windowed 100979 (9.9) | windowed+int8kv 92788 (10.8) | Qualcomm NPU ~18.2",flush=True)
if a: print(f"WGQA vs windowed 100979: {100979/a:.2f}x ; vs full-KV: {210177/a:.2f}x",flush=True)
print("WGQA_DONE",flush=True)
