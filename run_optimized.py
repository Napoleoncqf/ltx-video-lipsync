"""
Optimized xixi video gen - applying all findings from original workflow analysis.
Key changes vs previous tests:
1. First frame + LAST frame dual guide (frame_idx=0 and -1, strength=0.7)
2. Guide strength 0.7 (not 0.95)
3. LTXVPreprocess compression 25 (not 35)
4. 8 second duration (not 4s), length=193 frames (8s*24fps, rounded to 8n+1)
5. LoRA strength 0.65 (best from A test)
6. 2000-step LoRA
7. Negative prompt with watermark/text
8. 15-step sampling
"""
import json, urllib.request, urllib.error, websocket, uuid, time, sys, os, shutil
from PIL import Image

SERVER = "127.0.0.1:8288"
CLIENT_ID = str(uuid.uuid4())

SIGMAS_15 = "1., 0.996, 0.991, 0.986, 0.98, 0.97, 0.95, 0.92, 0.87, 0.8, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0"

def build_prompt(lora_name, lora_str, w, h, length, duration, seed, prefix):
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors"}},
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": lora_name, "strength_model": lora_str}},
        "3": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["2", 0],
                         "lora_name": "LTX\\ltx-2.3-22b-distilled-lora-resized_dynamic_rank_159_fro09_bf16.safetensors",
                         "strength_model": 0.3}},
        "4": {"class_type": "LTXAVTextEncoderLoader",
              "inputs": {"text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
                         "ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors", "device": "default"}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["4", 0],
                         "text": "xixi, a young Chinese woman with bangs and long dark brown hair, round face, big brown eyes, fair skin, speaking to camera, close-up portrait, wearing a white blouse, bright natural lighting, warm tones, natural skin texture, mouth moving while talking, professional news anchor pose"}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["4", 0],
                         "text": "blurry, out of focus, overexposed, underexposed, low contrast, noise, grainy, poor lighting, flickering, motion blur, distorted, unnatural skin, dark, grey, dull, watermark, text, logo, subtitle"}},
        # Audio
        "10": {"class_type": "LoadAudio", "inputs": {"audio": "podcast_audio.wav"}},
        "11": {"class_type": "MelBandRoFormerModelLoader",
               "inputs": {"model_name": "MelBandRoformer_fp16.safetensors"}},
        "12": {"class_type": "MelBandRoFormerSampler",
               "inputs": {"audio": ["10", 0], "model": ["11", 0]}},
        "13": {"class_type": "TrimAudioDuration",
               "inputs": {"audio": ["12", 0], "start_index": 0.0, "duration": duration}},
        "14": {"class_type": "LTXVAudioVAELoader",
               "inputs": {"ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors"}},
        "15": {"class_type": "LTXVAudioVAEEncode",
               "inputs": {"audio": ["13", 0], "audio_vae": ["14", 0]}},
        # Image
        "20": {"class_type": "LoadImage", "inputs": {"image": "xixi_first_frame.png"}},
        "21": {"class_type": "LTXVPreprocess", "inputs": {"image": ["20", 0], "img_compression": 25}},
        # Latent
        "30": {"class_type": "EmptyLTXVLatentVideo",
               "inputs": {"width": w, "height": h, "length": length, "batch_size": 1}},
        # Conditioning
        "31": {"class_type": "LTXVConditioning",
               "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": 24}},
        # First frame guide (strength 0.7, matching original workflow)
        "32": {"class_type": "LTXVAddGuide",
               "inputs": {"positive": ["31", 0], "negative": ["31", 1], "vae": ["1", 2],
                          "latent": ["30", 0], "image": ["21", 0],
                          "frame_idx": 0, "strength": 0.7}},
        # LAST frame guide (NEW! frame_idx=-1, strength 0.7)
        "33": {"class_type": "LTXVAddGuide",
               "inputs": {"positive": ["32", 0], "negative": ["32", 1], "vae": ["1", 2],
                          "latent": ["32", 2], "image": ["21", 0],
                          "frame_idx": -1, "strength": 0.7}},
        # AV Concat + Noise Mask + CropGuides (use node 33 output now)
        "35": {"class_type": "LTXVConcatAVLatent",
               "inputs": {"video_latent": ["33", 2], "audio_latent": ["15", 0]}},
        "36": {"class_type": "SolidMask",
               "inputs": {"value": 1.0, "width": w // 8, "height": h // 8}},
        "37": {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["35", 0], "mask": ["36", 0]}},
        "38": {"class_type": "LTXVCropGuides",
               "inputs": {"positive": ["33", 0], "negative": ["33", 1], "latent": ["37", 0]}},
        # Sampling
        "40": {"class_type": "CFGGuider",
               "inputs": {"model": ["3", 0], "positive": ["38", 0], "negative": ["38", 1], "cfg": 1.0}},
        "41": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "42": {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS_15}},
        "43": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "44": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["43", 0], "guider": ["40", 0],
                          "sampler": ["41", 0], "sigmas": ["42", 0], "latent_image": ["38", 2]}},
        # Decode
        "48": {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 1, "height": 1}},
        "49": {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["44", 0], "mask": ["48", 0]}},
        "51": {"class_type": "VAEDecode", "inputs": {"samples": ["49", 0], "vae": ["1", 2]}},
        # Output
        "60": {"class_type": "CreateVideo",
               "inputs": {"images": ["51", 0], "fps": 24.0, "audio": ["13", 0]}},
        "61": {"class_type": "SaveVideo",
               "inputs": {"video": ["60", 0], "filename_prefix": prefix,
                          "format": "mp4", "codec": "h264"}},
    }


def run_test(prompt_data, name):
    data = json.dumps({"prompt": prompt_data, "client_id": CLIENT_ID}).encode("utf-8")
    try:
        req = urllib.request.Request(f"http://{SERVER}/prompt", data=data)
        resp = json.loads(urllib.request.urlopen(req).read())
        prompt_id = resp.get("prompt_id")
    except urllib.error.HTTPError as e:
        print(f"  ERROR: {e.read().decode('utf-8', errors='replace')[:500]}")
        return False
    ws = websocket.create_connection(f"ws://{SERVER}/ws?clientId={CLIENT_ID}")
    t0 = time.time()
    try:
        while True:
            msg = ws.recv()
            if isinstance(msg, str):
                d = json.loads(msg)
                if d.get("type") == "progress":
                    v, m = d["data"].get("value", 0), d["data"].get("max", 1)
                    sys.stdout.write(f"\r  Step {v}/{m} ({time.time()-t0:.0f}s)")
                    sys.stdout.flush()
                elif d.get("type") == "executing":
                    nid = d["data"].get("node")
                    if nid is None:
                        print(f" -> Done ({time.time()-t0:.0f}s)")
                        break
                elif d.get("type") == "execution_error":
                    print(f"\n  ERROR: {d.get('data',{}).get('exception_message','')[:300]}")
                    return False
    finally:
        ws.close()
    return True


# Setup first frame
print("Setting up...")
orig = Image.open(r"C:\Users\admin\Pictures\曦曦\lora\xixi_09.png")

# Test configs: (name, lora, w, h, length, duration, seed)
# length = frames, must be 8n+1. 8s * 24fps = 192 -> 193 (8*24+1)
TESTS = [
    # Main optimized test: portrait 8s with dual guide
    ("opt_portrait_8s", "xixi_ltx23_2000.safetensors", 0.65, 768, 1024, 193, 8.0, 5001),
    # Landscape 8s for comparison
    ("opt_landscape_8s", "xixi_ltx23_2000.safetensors", 0.65, 1024, 576, 193, 8.0, 5002),
    # Compare with 1500 step
    ("opt_1500_portrait", "xixi_ltx23_1500.safetensors", 0.65, 768, 1024, 193, 8.0, 5001),
]

# Wait for server
urllib.request.urlopen(f"http://{SERVER}/", timeout=5)
print("ComfyUI ready!\n")

for i, (name, lora, lora_str, w, h, length, dur, seed) in enumerate(TESTS):
    # Resize first frame
    img = orig.copy()
    ratio = max(w / img.size[0], h / img.size[1])
    nw, nh = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    img = img.crop((left, top, left + w, top + h))
    img.save(r"E:\ComfyUI-aki-v2\ComfyUI\input\xixi_first_frame.png")

    print(f"[{i+1}/{len(TESTS)}] {name} (LoRA={lora_str}, {w}x{h}, {dur}s, {length}frames)")
    prompt = build_prompt(lora, lora_str, w, h, length, dur, seed, name)
    run_test(prompt, name)

print("\nAll optimized tests complete!")
