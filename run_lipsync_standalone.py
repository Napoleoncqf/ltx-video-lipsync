"""
Run LatentSync standalone - mimics ComfyUI node's loading method.
No ComfyUI overhead, 14GB VRAM available.
"""
import os, sys, gc, argparse

# Set expandable segments BEFORE any CUDA init
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

LATENTSYNC_DIR = r"E:\ComfyUI-aki-v2\ComfyUI\custom_nodes\ComfyUI-LatentSyncWrapper"
sys.path.insert(0, LATENTSYNC_DIR)

VIDEO_INPUT = r"E:\ComfyUI-aki-v2\ComfyUI\output\lipsync_input.mp4"
# Auto-detect today's podcast or use latest
from datetime import datetime
today = datetime.now().strftime("%Y-%m-%d")
AUDIO_INPUT = rf"D:\Projects\isotope\data\podcast\podcast_{today}.wav"
if not os.path.exists(AUDIO_INPUT):
    import glob
    podcasts = sorted(glob.glob(r"D:\Projects\isotope\data\podcast\podcast_*.wav"))
    AUDIO_INPUT = podcasts[-1] if podcasts else AUDIO_INPUT
VIDEO_OUTPUT = r"E:\ComfyUI-aki-v2\ComfyUI\output\lipsync_standalone_result.mp4"

# Get video duration and trim audio
import subprocess, json, tempfile
probe = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", VIDEO_INPUT],
    capture_output=True, text=True
)
video_duration = float(json.loads(probe.stdout)["format"]["duration"])
print(f"Video: {VIDEO_INPUT} ({video_duration:.1f}s)")

temp_dir = tempfile.mkdtemp(prefix="lipsync_")
trimmed_audio = os.path.join(temp_dir, "audio.wav")
subprocess.run([
    "ffmpeg", "-i", AUDIO_INPUT, "-t", str(video_duration),
    "-ar", "16000", "-ac", "1", trimmed_audio, "-y"
], capture_output=True)
print(f"Audio trimmed: {trimmed_audio}")

# GPU info
if torch.cuda.is_available():
    free = torch.cuda.mem_get_info()[0] / 1e9
    total = torch.cuda.mem_get_info()[1] / 1e9
    print(f"GPU: {free:.1f}GB free / {total:.1f}GB total")

# Find config
config_path = None
config_dir = os.path.join(LATENTSYNC_DIR, "configs", "unet")
for f in ["stage2_512.yaml", "stage2.yaml"]:
    p = os.path.join(config_dir, f)
    if os.path.exists(p):
        config_path = p
        break
if not config_path:
    configs = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
    config_path = os.path.join(config_dir, configs[-1])  # Last = highest stage
print(f"Config: {config_path}")

# Build args (matching ComfyUI node)
ckpt_path = os.path.join(LATENTSYNC_DIR, "checkpoints", "latentsync_unet.pt")
whisper_path = os.path.join(LATENTSYNC_DIR, "checkpoints", "whisper", "tiny.pt")
mask_path = os.path.join(LATENTSYNC_DIR, "latentsync", "utils", "mask.png")
scheduler_path = os.path.join(LATENTSYNC_DIR, "configs")

print(f"UNet: {ckpt_path} (exists: {os.path.exists(ckpt_path)})")
print(f"Whisper: {whisper_path} (exists: {os.path.exists(whisper_path)})")

from omegaconf import OmegaConf
config = OmegaConf.load(config_path)
if hasattr(config, "data") and hasattr(config.data, "mask_image_path"):
    config.data.mask_image_path = mask_path

args = argparse.Namespace(
    unet_config_path=config_path,
    inference_ckpt_path=ckpt_path,
    video_path=VIDEO_INPUT,
    audio_path=trimmed_audio,
    video_out_path=VIDEO_OUTPUT,
    seed=42,
    inference_steps=30,
    guidance_scale=2.0,
    scheduler_config_path=scheduler_path,
    whisper_ckpt_path=whisper_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=1,  # Minimal batch to save VRAM
    use_mixed_precision=True,
    temp_dir=temp_dir,
    mask_image_path=mask_path,
)

# Clean GPU
gc.collect()
torch.cuda.empty_cache()

# Import and run inference script
print("\nStarting LatentSync inference...")
inference_script = os.path.join(LATENTSYNC_DIR, "scripts", "inference.py")

import importlib.util
spec = importlib.util.spec_from_file_location("inference", inference_script)
inference_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_module)

try:
    inference_module.main(config, args)
    print(f"\nOutput: {VIDEO_OUTPUT}")
    if os.path.exists(VIDEO_OUTPUT):
        size = os.path.getsize(VIDEO_OUTPUT) / 1e6
        print(f"Size: {size:.1f}MB")
        os.startfile(VIDEO_OUTPUT)
    else:
        print("No output file - check for errors above")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
import shutil
shutil.rmtree(temp_dir, ignore_errors=True)
