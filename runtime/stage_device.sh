#!/usr/bin/env bash
# Stage this repo onto an adb-attached Snapdragon device.
#
# The QNN runtime (qnn-net-run + libQnnHtp*.so + the HTP skel/stub for your hexagon version)
# is NOT in this repo -- it ships with Qualcomm's AI Engine Direct (QAIRT) SDK and is not
# redistributable. Point QAIRT_DIR at your SDK install, or pre-stage those files yourself.
#
#   QAIRT_DIR=/path/to/qairt/2.45.0.xxxxxx ./stage_device.sh <serial> [v79|v81]
#
# Note the KV buffers are NOT pushed: run_gate.py creates them on device with dd, because
# pushing ~288 MB of zeros over a slow adb link was the flakiest part of the pipeline.
# (Baseline uses full-CTX buffers on every layer -- that is the cost windowed-KV removes.)
set -euo pipefail

SERIAL="${1:?usage: stage_device.sh <adb-serial> [v79|v81]}"
HTP="${2:-v79}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE=/data/local/tmp/gemma
ADB=(adb -s "$SERIAL")

case "$HTP" in
  v79) SOC_ARCH=v79 ;;
  v81) SOC_ARCH=v81 ;;
  *) echo "unknown HTP '$HTP' (expected v79 or v81)" >&2; exit 1 ;;
esac

echo ">> device: $("${ADB[@]}" shell getprop ro.soc.model | tr -d '\r') (staging for HTP $SOC_ARCH)"
"${ADB[@]}" shell "mkdir -p $BASE/bin $BASE/lib $BASE/dsp $BASE/artifacts $BASE/step $BASE/kv $BASE/out $BASE/tstep $BASE/tout"

# ---- QNN runtime (from your QAIRT SDK) ----------------------------------------------
if [ -n "${QAIRT_DIR:-}" ]; then
  echo ">> pushing QNN runtime from QAIRT_DIR=$QAIRT_DIR"
  AA="$QAIRT_DIR/lib/aarch64-android"
  HX="$QAIRT_DIR/lib/hexagon-$SOC_ARCH/unsigned"
  [ -d "$AA" ] || { echo "missing $AA" >&2; exit 1; }
  [ -d "$HX" ] || { echo "missing $HX -- your SDK may not include hexagon-$SOC_ARCH" >&2; exit 1; }
  "${ADB[@]}" push "$QAIRT_DIR/bin/aarch64-android/qnn-net-run" "$BASE/bin/" >/dev/null
  "${ADB[@]}" shell "chmod 755 $BASE/bin/qnn-net-run"
  for f in libQnnHtp.so libQnnSystem.so libQnnHtpPrepare.so libQnnHtpNetRunExtensions.so \
           "libQnnHtp${SOC_ARCH^^}Stub.so"; do
    [ -f "$AA/$f" ] && "${ADB[@]}" push "$AA/$f" "$BASE/lib/" >/dev/null
  done
  for f in "$HX"/libQnnHtp*.so; do "${ADB[@]}" push "$f" "$BASE/dsp/" >/dev/null; done
else
  echo ">> QAIRT_DIR not set -- skipping QNN runtime."
  echo "   You must stage these yourself under $BASE:"
  echo "     bin/qnn-net-run"
  echo "     lib/libQnnHtp.so libQnnSystem.so libQnnHtpPrepare.so libQnnHtpNetRunExtensions.so libQnnHtp${SOC_ARCH^^}Stub.so"
  echo "     dsp/libQnnHtp${SOC_ARCH^^}.so libQnnHtp${SOC_ARCH^^}Skel.so"
fi

# ---- context binaries ---------------------------------------------------------------
echo ">> pushing context binaries (~1.9 GB each, be patient)"
for b in "$REPO"/*baseline*"_$HTP.bin" "$REPO"/bins/*baseline*"_$HTP.bin"; do
  [ -f "$b" ] || continue
  echo "   $(basename "$b")"
  "${ADB[@]}" push "$b" "$BASE/artifacts/" >/dev/null
done

# ---- on-device step scripts ---------------------------------------------------------
for s in gate_ondevice.sh; do
  "${ADB[@]}" push "$REPO/runtime/$s" "$BASE/" >/dev/null
  "${ADB[@]}" shell "chmod 755 $BASE/$s"
done

echo ">> staged:"
"${ADB[@]}" shell "ls -la $BASE/artifacts $BASE/bin $BASE/lib $BASE/dsp"
echo ">> OK"
