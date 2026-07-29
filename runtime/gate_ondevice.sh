#!/system/bin/sh
# On-device single-step decode via qnn-net-run against the prebuilt A16W8 v79 context binary.
# Runs entirely on the device: KV present->past rotation is local file mv (no host transfer).
# Host pushes only the tiny per-step inputs (inputs_embeds, per_layer_inputs, position_ids,
# cache_position, full_mask, sliding_mask) and pulls only hidden.raw.
#
# Layout on device (BASE=/data/local/tmp/gemma):
#   $BASE/bin/qnn-net-run, $BASE/lib/*.so, $BASE/dsp/*.so
#   $BASE/artifacts/gemma4_decode_a16w8_v79.bin
#   $BASE/kv/past_{k,v}_<i>.raw     (persist across steps; seeded to zeros at step 0)
#   $BASE/step/<small per-step input .raw files>
#   $BASE/step/in.txt               (input_list: one line, space-separated name:=path)
#   $BASE/out/                      (net-run output dir -> hidden.raw + present_*.raw)
set -e
BASE=/data/local/tmp/gemma
export LD_LIBRARY_PATH=$BASE/lib:/system/lib64:/vendor/lib64
export ADSP_LIBRARY_PATH="$BASE/dsp;/vendor/dsp/cdsp;/vendor/lib/rfsa/cdsp;/vendor/dsp"
export LD_PRELOAD=/system/lib64/libbinder.so   # Android 16 linker fix (from qdc-session-notes)

STEP_DIR=$BASE/step
KV=$BASE/kv
OUT=$BASE/out
rm -rf "$OUT"; mkdir -p "$OUT"

# Build the input_list line. Order matches decode-io.tsv IN rows.
# net-run input_list format: "name1:=path1 name2:=path2 ..."
LINE="inputs_embeds:=$STEP_DIR/inputs_embeds.raw"
LINE="$LINE per_layer_inputs:=$STEP_DIR/per_layer_inputs.raw"
LINE="$LINE position_ids:=$STEP_DIR/position_ids.raw"
LINE="$LINE cache_position:=$STEP_DIR/cache_position.raw"
LINE="$LINE full_mask:=$STEP_DIR/full_mask.raw"
LINE="$LINE sliding_mask:=$STEP_DIR/sliding_mask.raw"
i=0
while [ $i -lt 15 ]; do
  LINE="$LINE past_k_$i:=$KV/past_k_$i.raw past_v_$i:=$KV/past_v_$i.raw"
  i=$((i+1))
done
echo "$LINE" > "$STEP_DIR/in.txt"

"$BASE/bin/qnn-net-run" \
  --backend "$BASE/lib/libQnnHtp.so" \
  --retrieve_context "$BASE/artifacts/gemma4_decode_baseline_a16w8_v79.bin" \
  --input_list "$STEP_DIR/in.txt" \
  --output_dir "$OUT" \
  --use_native_input_files \
  --use_native_output_files \
  --log_level error

# net-run writes outputs under $OUT/Result_0/
RES=$OUT/Result_0
# Rotate present_* -> past_* for the next step (on-device, no transfer).
i=0
while [ $i -lt 15 ]; do
  mv -f "$RES/present_k_$i.raw" "$KV/past_k_$i.raw"
  mv -f "$RES/present_v_$i.raw" "$KV/past_v_$i.raw"
  i=$((i+1))
done
echo "STEP_OK"
