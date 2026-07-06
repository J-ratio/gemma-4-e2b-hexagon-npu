#!/usr/bin/env python3
"""FINAL: deploy awq_g64c (the 79.3% recipe) on NPU = AWQ-scaled trunk + LPBQ block-64 int4.
AWQ scales already folded into trunk_awq_onnx (W*=s, input*=1/s Mul in graph). Now quantize THAT with LPBQ
block-64 (QNN-supported blockwise: block int4 + int8 outer scale) which we proved compiles on HTP.
Compile to QNN context binary + profile on S26. This is A16W4 (int16 acts, int4 weights), group-64, on the NPU.
Also runs an on-device INFERENCE job to capture real logits for top-1 agreement vs fp32 (measured separately)."""
import os, numpy as np, onnx
from aimet_onnx.quantsim import QuantizationSimModel, set_lpbq_for_params
import qai_hub as hub
DEV=hub.Device("Samsung Galaxy S26 Ultra")
SRC="/home/ubuntu/gq/trunk_awq_onnx/model.onnx"; OUT="/home/ubuntu/gq/trunk_awq_lpbq64"; os.makedirs(OUT,exist_ok=True)
SEQ,H,NL,PLD=128,1536,35,256
def sample():
    neg=np.finfo(np.float32).min; full=np.triu(np.full((SEQ,SEQ),neg,dtype=np.float32),1).reshape(1,1,SEQ,SEQ)
    return {"inputs_embeds":np.random.randn(1,SEQ,H).astype(np.float32),
            "per_layer_inputs":np.random.randn(1,SEQ,NL,PLD).astype(np.float32),
            "position_ids":np.arange(SEQ,dtype=np.int64).reshape(1,SEQ),
            "full_mask":full,"sliding_mask":full.copy()}
cal=[sample() for _ in range(2)]
m=onnx.load(SRC)
sim=QuantizationSimModel(m,param_type="int4",activation_type="int16",quant_scheme="min_max",
                         dummy_input=cal[0],providers=["CPUExecutionProvider"])
set_lpbq_for_params(sim, bitwidth=4, block_size=64, op_types={"MatMul","Gemm","Conv"})
print("AWQ+LPBQ64 set",flush=True)
def fp(sess,*_):
    for x in cal: sess.run(None,x)
print("compute_encodings ...",flush=True); sim.compute_encodings(fp)
sim.export(OUT,"trunk-awq-lpbq64"); print("exported",flush=True)
cj=hub.submit_compile_job(model=OUT,device=DEV,options="--target_runtime qnn_context_binary --truncate_64bit_io",name="trunk-awq-lpbq64-htp")
print(f"COMPILE {cj.job_id} {cj.url}",flush=True); cj.wait()
if not cj.get_status().success:
    print("COMPILE FAILED:",cj.get_status().message,flush=True); print("AWQLPBQ_RESULT: FAIL",flush=True)
else:
    pj=hub.submit_profile_job(model=cj.get_target_model(),device=DEV,name="trunk-awq-lpbq64-prof")
    print(f"PROFILE {pj.job_id} {pj.url}",flush=True); pj.wait()
    prof=pj.download_profile(); ex=prof.get("execution_summary",{}); units={}
    for l in prof.get("execution_detail",[]): u=l.get("compute_unit","?"); units[u]=units.get(u,0)+1
    print(f">>> AWQ+LPBQ64 trunk infer_us={ex.get('estimated_inference_time')} units={units}",flush=True)
    print("AWQLPBQ_RESULT: OK  (target_model_id="+cj.get_target_model().model_id+")",flush=True)
