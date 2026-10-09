#!/bin/bash
# Record the measurement environment. Read-only: changes nothing on the box.
# Run immediately before each measurement session; the output is the env of record.
OUT=latency_runs/env.txt
exec > "$OUT" 2>&1
echo "=== captured $(date -Is) ==="
echo
echo "--- uname -a ---";                 uname -a
echo; echo "--- L4T / JetPack (/etc/nv_tegra_release) ---"; cat /etc/nv_tegra_release
echo; echo "--- nvpmodel -q ---";        nvpmodel -q
echo "(/var/lib/nvpmodel/status: $(cat /var/lib/nvpmodel/status))"
echo; echo "--- jetson_clocks --show ---"; jetson_clocks --show || \
  echo "UNAVAILABLE: jetson_clocks requires root; this agent shell has no tty for sudo."
echo; echo "--- CPU/GPU clock state from sysfs (jetson_clocks substitute) ---"
for p in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
  echo "$p = $(cat $p 2>/dev/null)"; done
echo "cpu governor = $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
echo "gpu cur/max  = $(cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq 2>/dev/null)/$(cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/max_freq 2>/dev/null)"
echo; echo "--- tegrastats, 5 s idle @1Hz ---"
timeout 6 tegrastats --interval 1000 | head -5
echo; echo "--- free -m ---";            free -m
echo; echo "--- nproc ---";              nproc
echo; echo "--- df -h / ---";            df -h /
echo; echo "--- python / torch ---"
shubert_venv/bin/python3 -c "import sys,torch,cv2,numpy,transformers;\
print('python', sys.version.split()[0]);\
print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available());\
print('cv2', cv2.__version__, '| numpy', numpy.__version__, '| transformers', transformers.__version__)"
echo; echo "--- mediapipe ---"
shubert_venv/bin/python3 -c "import mediapipe; print('mediapipe', mediapipe.__version__)"
echo; echo "--- git ---"
git -C /home/sllu/asl-video-to-text rev-parse HEAD
git -C /home/sllu/asl-video-to-text status --short || true
echo "(PROJECT_CONTEXT.md and docs/ are gitignored; latency_runs/ is untracked)"
echo; echo "--- effective pipeline settings (code defaults unless an env var overrides) ---"
for v in BYT5_MAX_LENGTH BYT5_NUM_BEAMS BYT5_DEVICE BYT5_DTYPE BYT5_CKPT FRAME_STRIDE \
         PERCEPTION_WORKERS PERCEPTION_CHUNK MAX_LIVE_STREAMS MAX_RETAINED_FRAMES \
         MIN_AVAILABLE_MB RECORD_MODE MANUAL_TRIM MAX_CLIP_SECONDS DINOV2_BATCH_SIZE \
         GPU_SERIALIZE MEDIAPIPE_NUM_HANDS MEDIAPIPE_VIDEO_MODE USE_ONNX_PERCEPTION \
         PYTORCH_NO_CUDA_MEMORY_CACHING TRANSCRIPT; do
  echo "  env $v = ${!v-<unset>}"; done
shubert_venv/bin/python3 - <<'PY'
import os
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
import auto_segment_v5 as v5, streaming_perception as sp, kpe_mediapipe as kpe
import inference, dinov2_features as dino
print("  effective MAX_LIVE_STREAMS      =", v5.MAX_LIVE_STREAMS)
print("  effective MAX_RETAINED_FRAMES   =", v5.MAX_RETAINED_FRAMES)
print("  effective MIN_AVAILABLE_MB      =", v5.MIN_AVAILABLE_MB)
print("  effective MANUAL_RECORD         =", v5.MANUAL_RECORD, "(RECORD_MODE=manual -> True)")
print("  effective MAX_CLIP_SECONDS      =", v5.MAX_CLIP_SECONDS)
print("  effective STREAM_PERCEPTION     =", v5.STREAM_PERCEPTION)
print("  effective STREAM_DINOV2         =", v5.STREAM_DINOV2)
print("  effective PERCEPTION_WORKERS    =", sp.PERCEPTION_WORKERS)
print("  effective PERCEPTION_CHUNK      =", sp.PERCEPTION_CHUNK)
print("  effective FRAME_STRIDE          =", sp.stride_from_env())
print("  effective MEDIAPIPE num_hands   =", kpe.NUM_HANDS,
      "presence", kpe.HAND_PRESENCE_CONF, "tracking", kpe.HAND_TRACKING_CONF,
      "video_mode", kpe.VIDEO_MODE)
print("  effective DINOv2 dtypes         =", dino.HANDS_DTYPE, dino.FACE_DTYPE,
      "batch", dino.DEFAULT_BATCH_SIZE)
print("  ByT5 checkpoint (v5 config)     =", v5.config['slt_model_checkpoint'])
print("  ByT5 defaults                   = beams",
      max(1, int(os.environ.get("BYT5_NUM_BEAMS", "4"))), "max_length",
      max(64, int(os.environ.get("BYT5_MAX_LENGTH", "768"))),
      "device", os.environ.get("BYT5_DEVICE", "cuda"),
      "dtype", os.environ.get("BYT5_DTYPE", "bfloat16"))
PY
echo; echo "--- camera ---"
ls -l /dev/video* 2>&1 || echo "NO /dev/video* NODE"
lsusb | grep -i logitech || echo "(no Logitech device on USB)"
lsmod | grep -q uvcvideo && echo "uvcvideo: loaded" || echo "uvcvideo: NOT LOADED"
echo; echo "--- other load on the box (ambient state) ---"
echo "uptime: $(uptime)"
ps -eo pcpu,pmem,rss,etime,args --sort=-pcpu | head -12
echo; echo "witness (power-fault telemetry) is expected to be running at 1 Hz:"
ps -eo pid,etime,args | grep "[c]rash_witness" || echo "  witness NOT running"
