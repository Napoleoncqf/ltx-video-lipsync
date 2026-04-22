"""
Test with xixi_32 (cafe photo) as first+last frame.
Two variants: with and without MelBandRoFormer vocal separation.
"""
import json, urllib.request, urllib.error, websocket, uuid, time, sys, os
from PIL import Image

SERVER = "127.0.0.1:8288"
CLIENT_ID = str(uuid.uuid4())
SIGMAS_15 = "1., 0.996, 0.991, 0.986, 0.98, 0.97, 0.95, 0.92, 0.87, 0.8, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0"

def build_prompt(w, h, length, duration, seed, prefix, use_vocal_sep=True):
    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors"}},
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": "xixi_ltx23_2000.safetensors", "strength_model": 0.65}},
        "3": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["2", 0],
                         "lora_name": "LTX\\ltx-2.3-22b-distilled-lora-resized_dynamic_rank_159_fro09_bf16.safetensors",
                         "strength_model": 0.3}},
        "4": {"class_type": "LTXAVTextEncoderLoader",
              "inputs": {"text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
                         "ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors", "device": "default"}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["4", 0],
                         "text": "xixi, a young Chinese woman with bangs and long dark brown hair, round face, big brown eyes, fair skin, sitting at a cafe, wearing a cream cardigan over white top, speaking to camera, mouth moving while talking, warm indoor lighting, close-up portrait, natural expression"}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["4", 0],
                         "text": "blurry, out of focus, overexposed, underexposed, noise, grainy, poor lighting, flickering, motion blur, distorted, unnatural skin, dark, grey, dull, watermark, text, logo, subtitle"}},
        "10": {"class_type": "LoadAudio", "inputs": {"audio": "podcast_audio.wav"}},
    }

    # Audio pipeline
    if use_vocal_sep:
        prompt["11"] = {"class_type": "MelBandRoFormerModelLoader",
                        "inputs": {"model_name": "MelBandRoformer_fp16.safetensors"}}
        prompt["12"] = {"class_type": "MelBandRoFormerSampler",
                        "inputs": {"audio": ["10", 0], "model": ["11", 0]}}
        audio_src = ["12", 0]
    else:
        audio_src = ["10", 0]

    prompt.update({
        "13": {"class_type": "TrimAudioDuration",
               "inputs": {"audio": audio_src, "start_index": 0.0, "duration": duration}},
        "14": {"class_type": "LTXVAudioVAELoader",
               "inputs": {"ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors"}},
        "15": {"class_type": "LTXVAudioVAEEncode",
               "inputs": {"audio": ["13", 0], "audio_vae": ["14", 0]}},
        "20": {"class_type": "LoadImage", "inputs": {"image": "xixi_first_frame.png"}},
        "21": {"class_type": "LTXVPreprocess", "inputs": {"image": ["20", 0], "img_compression": 25}},
        "30": {"class_type": "EmptyLTXVLatentVideo",
               "inputs": {"width": w, "height": h, "length": length, "batch_size": 1}},
        "31": {"class_type": "LTXVConditioning",
               "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": 24}},
        # First frame guide
        "32": {"class_type": "LTXVAddGuide",
               "inputs": {"positive": ["31", 0], "negative": ["31", 1], "vae": ["1", 2],
                          "latent": ["30", 0], "image": ["21", 0],
                          "frame_idx": 0, "strength": 0.7}},
        # Last frame guide (same image)
        "33": {"class_type": "LTXVAddGuide",
               "inputs": {"positive": ["32", 0], "negative": ["32", 1], "vae": ["1", 2],
                          "latent": ["32", 2], "image": ["21", 0],
                          "frame_idx": -1, "strength": 0.7}},
        "35": {"class_type": "LTXVConcatAVLatent",
               "inputs": {"video_latent": ["33", 2], "audio_latent": ["15", 0]}},
        "36": {"class_type": "SolidMask",
               "inputs": {"value": 1.0, "width": w // 8, "height": h // 8}},
        "37": {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["35", 0], "mask": ["36", 0]}},
        "38": {"class_type": "LTXVCropGuides",
               "inputs": {"positive": ["33", 0], "negative": ["33", 1], "latent": ["37", 0]}},
        "40": {"class_type": "CFGGuider",
               "inputs": {"model": ["3", 0], "positive": ["38", 0], "negative": ["38", 1], "cfg": 1.0}},
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
                elif d.get("type") == "executing" and d["data"].get("node") is None:
                    print(f" -> Done ({time.time()-t0:.0f}s)")
                    break
                elif d.get("type") == "execution_error":
                    print(f"\n  ERROR: {d.get('data',{}).get('exception_message','')[:300]}")
                    return False
    finally:
        ws.close()
    return True


# Resize first frame
orig = Image.open(r"E:\ComfyUI-aki-v2\ComfyUI\input\xixi_first_frame.png")

urllib.request.urlopen(f"http://{SERVER}/", timeout=5)
print("ComfyUI ready!\n")

# 8s = 193 frames (8*24+1)
TESTS = [
    # With vocal separation
    ("cafe_vocal_sep", True, 768, 1024, 193, 8.0, 6001),
    # Without vocal separation (raw audio)
    ("cafe_raw_audio", False, 768, 1024, 193, 8.0, 6001),
    # Landscape with vocal sep
    ("cafe_landscape", True, 1024, 576, 193, 8.0, 6002),
]

for i, (name, vocal_sep, w, h, length, dur, seed) in enumerate(TESTS):
    # Resize first frame to match
    img = orig.copy()
    ratio = max(w / img.size[0], h / img.size[1])
    nw, nh = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    img = img.crop((left, top, left + w, top + h))
    img.save(r"E:\ComfyUI-aki-v2\ComfyUI\input\xixi_first_frame.png")

    sep_str = "vocal_sep" if vocal_sep else "raw_audio"
    print(f"[{i+1}/{len(TESTS)}] {name} ({w}x{h}, {sep_str}, {dur}s)")
    prompt = build_prompt(w, h, length, dur, seed, name, vocal_sep)
    run_test(prompt, name)

# Open all videos
for t in ['cafe_vocal_sep', 'cafe_raw_audio', 'cafe_landscape']:
    os.startfile(rf'E:\ComfyUI-aki-v2\ComfyUI\output\{t}_00001_.mp4')

print("\nAll done! Videos opened.")
