"""
Configuration for LTX-Video podcast pipeline.
Copy this file to config_local.py and modify paths for your environment.
"""
import os

# === Paths (modify these for your environment) ===
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", "")
if not COMFYUI_DIR:
    # Auto-detect: look for ComfyUI in common locations
    for candidate in [r"C:\ComfyUI", r"E:\ComfyUI-aki-v2\ComfyUI", os.path.expanduser("~/ComfyUI")]:
        if os.path.exists(candidate):
            COMFYUI_DIR = candidate
            break
    if not COMFYUI_DIR:
        raise RuntimeError("Set COMFYUI_DIR env var or create config_local.py")

COMFYUI_PYTHON = os.environ.get("COMFYUI_PYTHON", os.path.join(os.path.dirname(COMFYUI_DIR), "python", "python.exe"))
if not os.path.exists(COMFYUI_PYTHON):
    COMFYUI_PYTHON = "python"  # Fallback to system python

# Input data
PODCAST_DIR = os.environ.get("PODCAST_DIR", os.path.join(COMFYUI_DIR, "input", "podcasts"))
FIRST_FRAME_SOURCE = os.environ.get("FIRST_FRAME", os.path.join(COMFYUI_DIR, "input", "first_frame.png"))

# Output
OUTPUT_DIR = os.path.join(COMFYUI_DIR, "output", "podcast_video")
FINAL_OUTPUT_DIR = os.environ.get("FINAL_OUTPUT_DIR", os.path.join(COMFYUI_DIR, "output", "final"))

# LatentSync
LATENTSYNC_DIR = os.path.join(COMFYUI_DIR, "custom_nodes", "ComfyUI-LatentSyncWrapper")

# === Model names (relative to ComfyUI models dir) ===
CHECKPOINT = "ltx-2.3-22b-dev-nvfp4.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
CHARACTER_LORA = "xixi_ltx23_2000.safetensors"
DISTILLED_LORA = "LTX\\ltx-2.3-22b-distilled-lora-resized_dynamic_rank_159_fro09_bf16.safetensors"

# === Generation parameters ===
CHARACTER_LORA_STRENGTH = 0.85
DISTILLED_LORA_STRENGTH = 0.3
GUIDE_STRENGTH = 0.7
WIDTH = 1024
HEIGHT = 576
FRAME_RATE = 24
SEGMENT_DURATION = 8.0
FRAMES_PER_SEGMENT = 193  # 8s * 24fps + 1
SAMPLING_STEPS = 15
SEED = 8042

# Prompts
POSITIVE_PROMPT = (
    "xixi, a young Chinese woman, 22 years old, bangs and long dark brown hair with hair clip, "
    "delicate slim oval face, large bright brown eyes, small nose, youthful appearance, fair skin, "
    "sitting at a cafe table, wearing a cozy cream cardigan over white top, speaking to camera, "
    "mouth moving while talking, warm cafe lighting with plants, close-up portrait, natural soft expression"
)

NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, noise, grainy, poor lighting, flickering, "
    "motion blur, distorted, unnatural skin, dark, grey, dull, watermark, text, logo, subtitle, "
    "fat face, chubby, wide face, puffy"
)

SIGMAS_15 = "1., 0.996, 0.991, 0.986, 0.98, 0.97, 0.95, 0.92, 0.87, 0.8, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0"

# === LatentSync parameters ===
LIPSYNC_INFERENCE_STEPS = 30
LIPSYNC_GUIDANCE_SCALE = 2.0

# === ComfyUI server ===
COMFYUI_PORT = 8288
COMFYUI_SERVER = f"127.0.0.1:{COMFYUI_PORT}"

# Override with local config if exists
try:
    from config_local import *  # noqa: F401,F403
except ImportError:
    pass
