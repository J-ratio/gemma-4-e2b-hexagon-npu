#!/usr/bin/env python3
"""DECISIVE: does LPBQ int4 (block-64, int8 outer scale) COMPILE on QAIRT/HTP where raw PER_BLOCK failed?
LPBQ = Low-Power Blockwise Quant, the QNN-supported block format (per-block int4 + per-channel int16 scale),
a DIFFERENT op than QNN_FullyConnected_w_blk_scale that failed. If this compiles+profiles 100% NPU, the
group-64 accuracy (awq_g64c 79.3%) is deployable. Tests the plain trunk (accuracy is separate; this is the
op-support gate). Usage: python lpbq_probe.py <block_size>"""
import os, sys, numpy as np, onnx
from aimet_onnx.quantsim import QuantizationSimModel, set_lpbq_for_params
import qai_hub as hub
DEV=hub.Device("Samsung Galaxy S26 Ultra")
BLK=int(sys.argv[1]) if len(sys.argv)>1 else 64
IN="/home/ubuntu/gq/trunk_onnx/model.onnx"; OUT=f"/home/ubuntu/gq/trunk_lpbq{BLK}"; os.makedirs(OUT,exist_ok=True)
SEQ,H,NL,PLD=128,1536,35,256
def sample():
    neg=np.finfo(np.float32).min; full=np.triu(np.full((SEQ,SEQ),neg,dtype=np.float32),1).reshape(1,1,SEQ,SEQ)
    return {"inputs_embeds":np.random.randn(1,SEQ,H).astype(np.float32),
            "per_layer_inputs":np.random.randn(1,SEQ,NL,PLD).astype(np.float32),
            "position_ids":np.arange(SEQ,dtype=np.int64).reshape(1,SEQ),
            "full_mask":full,"sliding_mask":full.copy()}
cal=[sample() for _ in range(2)]
m=onnx.load(IN)
sim=QuantizationSimModel(m,param_type="int4",activation_type="int16",quant_scheme="min_max",
                         dummy_input=cal[0],providers=["CPUExecutionProvider"])
# LPBQ: block-BLK int4 weights with int8 decompressed outer scale, on MatMul/Gemm
set_lpbq_for_params(sim, bitwidth=4, block_size=BLK, op_types={"MatMul","Gemm","Conv"})
print(f"LPBQ set: block={BLK}, bw=4, decomp=8",flush=True)
def fp(sess,*_):
    for x in cal: sess.run(None,x)
print("compute_encodings ...",flush=True); sim.compute_encodings(fp)
sim.export(OUT,f"trunk-lpbq{BLK}"); print("exported",flush=True)
cj=hub.submit_compile_job(model=OUT,device=DEV,options="--target_runtime qnn_context_binary --truncate_64bit_io",name=f"trunk-lpbq{BLK}-htp")
print(f"COMPILE {cj.job_id} {cj.url}",flush=True); cj.wait()
if not cj.get_status().success:
    print("COMPILE FAILED:",cj.get_status().message,flush=True); print(f"LPBQ{BLK}_RESULT: FAIL",flush=True)
else:
    pj=hub.submit_profile_job(model=cj.get_target_model(),device=DEV,name=f"trunk-lpbq{BLK}-prof")
    print(f"PROFILE {pj.job_id} {pj.url}",flush=True); pj.wait()
    prof=pj.download_profile(); ex=prof.get("execution_summary",{}); units={}
    for l in prof.get("execution_detail",[]): u=l.get("compute_unit","?"); units[u]=units.get(u,0)+1
    print(f">>> LPBQ{BLK} trunk infer_us={ex.get('estimated_inference_time')} units={units}",flush=True)
    print(f"LPBQ{BLK}_RESULT: OK",flush=True)
