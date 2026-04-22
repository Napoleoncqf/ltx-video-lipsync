"""
Batch test v2 - comprehensive parameter sweep for xixi LoRA 2000 step.
Key optimizations:
1. Match aspect ratio to training data (portrait 3:4)
2. Test LoRA strength range 0.55-1.0
3. Test with/without distilled LoRA
4. 15-step sampling for all tests
5. First frame properly resized
"""
import json, urllib.request, urllib.error, websocket, uuid, time, sys, os, shutil
from PIL import Image

SERVER = "127.0.0.1:8288"
CLIENT_ID = str(uuid.uuid4())

SIGMAS_15 = "1., 0.996, 0.991, 0.986, 0.98, 0.97, 0.95, 0.92, 0.87, 0.8, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0"

# Detect which LoRA files are available
def get_lora_name():
    resp = urllib.request.urlopen(f"http://{SERVER}/object_info")
    info = json.loads(resp.read())
    loras = info['LoraLoaderModelOnly']['input']['required']['lora_name'][0]
    for l in loras:
        if 'xixi_ltx23' in l and '2000' in l:
            return l
    for l in loras:
        if 'xixi_ltx23' in l and '1500' in l:
            return l
    for l in loras:
        if 'xixi_ltx23' in l:
            return l
    return "xixi_ltx23_1500.safetensors"


def build_prompt(lora_name, lora_strength, width, height, seed, prefix,
                 use_distilled=True, guide_strength=0.95):
    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors"}},
    }

    # LoRA chain
    prev_model = ["1", 0]
    next_id = 2

    # Xixi LoRA
    prompt[str(next_id)] = {"class_type": "LoraLoaderModelOnly",
        "inputs": {"model": prev_model, "lora_name": lora_name, "strength_model": lora_strength}}
    prev_model = [str(next_id), 0]
    next_id += 1

    # Distilled LoRA (optional)
    if use_distilled:
        prompt[str(next_id)] = {"class_type": "LoraLoaderModelOnly",
            "inputs": {"model": prev_model,
                       "lora_name": "LTX\\ltx-2.3-22b-distilled-lora-resized_dynamic_rank_159_fro09_bf16.safetensors",
                       "strength_model": 0.3}}
        prev_model = [str(next_id), 0]
        next_id += 1

    final_model = prev_model

    prompt.update({
        "4": {"class_type": "LTXAVTextEncoderLoader",
              "inputs": {"text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
                         "ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors", "device": "default"}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["4", 0],
                         "text": "xixi, a young Chinese woman with bangs and long dark brown hair, round face, big brown eyes, speaking to camera, close-up portrait, wearing a white blouse, bright natural lighting, warm tones, natural skin texture, mouth moving while talking"}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["4", 0],
                         "text": "blurry, out of focus, overexposed, underexposed, low contrast, noise, grainy, poor lighting, flickering, motion blur, distorted, unnatural skin, dark, grey, dull, watermark, text, logo"}},
        "10": {"class_type": "LoadAudio", "inputs": {"audio": "podcast_audio.wav"}},
        "13": {"class_type": "TrimAudioDuration", "inputs": {"audio": ["10", 0], "start_index": 0.0, "duration": 4.0}},
        "14": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors"}},
        "15": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["13", 0], "audio_vae": ["14", 0]}},
        "20": {"class_type": "LoadImage", "inputs": {"image": "xixi_first_frame.png"}},
        "21": {"class_type": "LTXVPreprocess", "inputs": {"image": ["20", 0], "img_compression": 35}},
        "30": {"class_type": "EmptyLTXVLatentVideo",
               "inputs": {"width": width, "height": height, "length": 97, "batch_size": 1}},
        "31": {"class_type": "LTXVConditioning",
               "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": 24}},
        "32": {"class_type": "LTXVAddGuide",
               "inputs": {"positive": ["31", 0], "negative": ["31", 1], "vae": ["1", 2],
                          "latent": ["30", 0], "image": ["21", 0],
                          "frame_idx": 0, "strength": guide_strength}},
        "35": {"class_type": "LTXVConcatAVLatent",
               "inputs": {"video_latent": ["32", 2], "audio_latent": ["15", 0]}},
        "36": {"class_type": "SolidMask",
               "inputs": {"value": 1.0, "width": width // 8, "height": height // 8}},
        "37": {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["35", 0], "mask": ["36", 0]}},
        "38": {"class_type": "LTXVCropGuides",
               "inputs": {"positive": ["32", 0], "negative": ["32", 1], "latent": ["37", 0]}},
        "40": {"class_type": "CFGGuider",
               "inputs": {"model": final_model, "positive": ["38", 0], "negative": ["38", 1], "cfg": 1.0}},
        "41": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "42": {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS_15}},
        "43": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "44": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["43", 0], "guider": ["40", 0],
                          "sampler": ["41", 0], "sigmas": ["42", 0], "latent_image": ["38", 2]}},
        "48": {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 1, "height": 1}},
        "49": {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["44", 0], "mask": ["48", 0]}},
        "51": {"class_type": "VAEDecode", "inputs": {"samples": ["49", 0], "vae": ["1", 2]}},
        "60": {"class_type": "CreateVideo",
               "inputs": {"images": ["51", 0], "fps": 24.0, "audio": ["13", 0]}},
        "61": {"class_type": "SaveVideo",
               "inputs": {"video": ["60", 0], "filename_prefix": prefix,
                          "format": "mp4", "codec": "h264"}},
    })
    return prompt


def run_test(prompt_data, name):
    data = json.dumps({"prompt": prompt_data, "client_id": CLIENT_ID}).encode("utf-8")
    try:
        req = urllib.request.Request(f"http://{SERVER}/prompt", data=data)
        resp = json.loads(urllib.request.urlopen(req).read())
        prompt_id = resp.get("prompt_id")
    except urllib.error.HTTPError as e:
        print(f"  ERROR: {e.read().decode('utf-8', errors='replace')[:300]}")
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
                if d.get("type") == "executing" and d["data"].get("node") is None:
                    print(f" -> Done ({time.time()-t0:.0f}s)")
                    break
                if d.get("type") == "execution_error":
                    print(f"\n  ERROR: {d.get('data',{}).get('exception_message','')[:200]}")
                    return False
    finally:
        ws.close()
    return True


def set_first_frame(width, height):
    orig = Image.open(r"C:\Users\admin\Pictures\曦曦\lora\xixi_09.png")  # cafe photo, nice lighting
    ratio = max(width / orig.size[0], height / orig.size[1])
    nw, nh = int(orig.size[0] * ratio), int(orig.size[1] * ratio)
    img = orig.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - width) // 2, (nh - height) // 2
    img = img.crop((left, top, left + width, top + height))
    img.save(r"E:\ComfyUI-aki-v2\ComfyUI\input\xixi_first_frame.png")


# Wait for server
print("Checking ComfyUI...")
urllib.request.urlopen(f"http://{SERVER}/", timeout=5)
lora_name = get_lora_name()
print(f"Using LoRA: {lora_name}\n")

# Tests: (name, lora_str, w, h, seed, use_distilled, guide_str)
TESTS = [
    # Portrait mode (match training aspect ratio)
    ("v2_portrait_s0.65", 0.65, 768, 1024, 4001, True, 0.95),
    ("v2_portrait_s0.85", 0.85, 768, 1024, 4001, True, 0.95),
    ("v2_portrait_s1.0",  1.0,  768, 1024, 4001, True, 0.95),
    # Landscape 16:9
    ("v2_landscape_s0.85", 0.85, 1024, 576, 4002, True, 0.95),
    # Without distilled LoRA (test if it interferes)
    ("v2_nodistill_s0.85", 0.85, 768, 1024, 4001, False, 0.95),
]

for i, (name, lora_str, w, h, seed, use_dist, guide_str) in enumerate(TESTS):
    set_first_frame(w, h)
    print(f"[{i+1}/{len(TESTS)}] {name} (LoRA={lora_str}, {w}x{h}, distill={use_dist})")
    prompt = build_prompt(lora_name, lora_str, w, h, seed, name, use_dist, guide_str)
    run_test(prompt, name)

print("\nAll tests done! Check output folder.")
