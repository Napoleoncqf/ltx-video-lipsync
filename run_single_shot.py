"""
LTX2.3 single-shot: full AV pipeline with patched SeparateAVLatent + CropGuides.
"""
import json, urllib.request, urllib.error, websocket, uuid, time, sys

SERVER = "127.0.0.1:8288"
CLIENT_ID = str(uuid.uuid4())

prompt = {
    # === Model ===
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors"}},
    "2": {"class_type": "LoraLoaderModelOnly",
          "inputs": {"model": ["1", 0], "lora_name": "xixi_ltx23_1000.safetensors", "strength_model": 1.0}},
    "3": {"class_type": "LoraLoaderModelOnly",
          "inputs": {"model": ["2", 0], "lora_name": "LTX\\ltx-2.3-22b-distilled-lora-resized_dynamic_rank_159_fro09_bf16.safetensors", "strength_model": 0.3}},

    # === Text Encoder ===
    "4": {"class_type": "LTXAVTextEncoderLoader",
          "inputs": {"text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors", "ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors", "device": "default"}},
    "5": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["4", 0], "text": "xixi, a young Chinese woman with bangs and long dark brown hair, speaking to camera, close-up portrait, wearing a white blouse, bright natural lighting, warm tones, expressive face, mouth moving while talking, professional news anchor pose"}},
    "6": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["4", 0], "text": "blurry, out of focus, overexposed, underexposed, low contrast, noise, grainy, poor lighting, flickering, motion blur, distorted, unnatural skin, dark, grey, dull"}},

    # === Audio Pipeline (raw, no vocal separation) ===
    "10": {"class_type": "LoadAudio", "inputs": {"audio": "podcast_audio.wav"}},
    "13": {"class_type": "TrimAudioDuration", "inputs": {"audio": ["10", 0], "start_index": 0.0, "duration": 4.0}},
    "14": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors"}},
    "15": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["13", 0], "audio_vae": ["14", 0]}},

    # === Image ===
    "20": {"class_type": "LoadImage", "inputs": {"image": "xixi_first_frame.png"}},
    "21": {"class_type": "LTXVPreprocess", "inputs": {"image": ["20", 0], "img_compression": 35}},

    # === Latent (768x1024 portrait, matching first frame) ===
    "30": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 768, "height": 1024, "length": 97, "batch_size": 1}},

    # === Conditioning ===
    "31": {"class_type": "LTXVConditioning", "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": 24}},

    # === First Frame Guide ===
    "32": {"class_type": "LTXVAddGuide",
           "inputs": {"positive": ["31", 0], "negative": ["31", 1], "vae": ["1", 2],
                      "latent": ["30", 0], "image": ["21", 0], "frame_idx": 0, "strength": 0.95}},

    # === AV Concat + Noise Mask + CropGuides ===
    "35": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["32", 2], "audio_latent": ["15", 0]}},
    "36": {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": 96, "height": 128}},
    "37": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["35", 0], "mask": ["36", 0]}},
    "38": {"class_type": "LTXVCropGuides",
           "inputs": {"positive": ["32", 0], "negative": ["32", 1], "latent": ["37", 0]}},

    # === Sampling ===
    "40": {"class_type": "CFGGuider",
           "inputs": {"model": ["3", 0], "positive": ["38", 0], "negative": ["38", 1], "cfg": 1.0}},
    "41": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "42": {"class_type": "ManualSigmas", "inputs": {"sigmas": "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"}},
    "43": {"class_type": "RandomNoise", "inputs": {"noise_seed": 2024}},
    "44": {"class_type": "SamplerCustomAdvanced",
           "inputs": {"noise": ["43", 0], "guider": ["40", 0], "sampler": ["41", 0], "sigmas": ["42", 0], "latent_image": ["38", 2]}},

    # === Decode (patched SeparateAVLatent + VAEDecodeTiled + AudioVAEDecode) ===
    "50": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["44", 0]}},
    "51": {"class_type": "VAEDecodeTiled",
           "inputs": {"samples": ["50", 0], "vae": ["1", 2], "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 64}},
    "52": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["50", 1], "audio_vae": ["14", 0]}},

    # === Output ===
    "60": {"class_type": "CreateVideo", "inputs": {"images": ["51", 0], "fps": 24.0, "audio": ["52", 0]}},
    "61": {"class_type": "SaveVideo",
           "inputs": {"video": ["60", 0], "filename_prefix": "ltx23_xixi_s1.0", "format": "mp4", "codec": "h264"}},
}

# --- Submit ---
print(f"Submitting {len(prompt)} nodes...")
data = json.dumps({"prompt": prompt, "client_id": CLIENT_ID}).encode("utf-8")
try:
    req = urllib.request.Request(f"http://{SERVER}/prompt", data=data)
    resp = json.loads(urllib.request.urlopen(req).read())
    prompt_id = resp.get("prompt_id")
    print(f"Queued! prompt_id={prompt_id}")
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8", errors="replace")
    print(f"HTTP {e.code}:")
    try:
        err = json.loads(body)
        for nid, nerr in err.get("node_errors", {}).items():
            ct = prompt.get(nid, {}).get("class_type", "?")
            print(f"  Node {nid} ({ct}):")
            for ei in nerr.get("errors", []):
                print(f"    {ei.get('message','')}: {ei.get('details','')}")
    except: print(body[:2000])
    sys.exit(1)

# --- Monitor ---
print("Executing... (full AV pipeline, may take 3-5 minutes)")
ws = websocket.create_connection(f"ws://{SERVER}/ws?clientId={CLIENT_ID}")
t0 = time.time()
try:
    while True:
        msg = ws.recv()
        if isinstance(msg, str):
            d = json.loads(msg)
            dt, dd = d.get("type"), d.get("data", {})
            if dt == "progress":
                sys.stdout.write(f"\r  Sampling: {dd.get('value',0)}/{dd.get('max',1)} ({time.time()-t0:.0f}s)   ")
                sys.stdout.flush()
            elif dt == "executing":
                nid = dd.get("node")
                if nid is None:
                    print(f"\n\nDone! ({time.time()-t0:.0f}s)")
                    break
                print(f"  [{time.time()-t0:.0f}s] {prompt.get(nid,{}).get('class_type','?')}")
            elif dt == "execution_error":
                print(f"\nERROR: {json.dumps(dd, ensure_ascii=False)[:1500]}")
                break
finally:
    ws.close()

# --- Output ---
time.sleep(1)
resp = urllib.request.urlopen(f"http://{SERVER}/history/{prompt_id}")
history = json.loads(resp.read())
if prompt_id in history:
    for nid, out in history[prompt_id].get("outputs", {}).items():
        for vid in out.get("videos", []):
            sf = vid.get("subfolder", "")
            fn = vid["filename"]
            p = f"E:\\ComfyUI-aki-v2\\ComfyUI\\output\\{sf}\\{fn}" if sf else f"E:\\ComfyUI-aki-v2\\ComfyUI\\output\\{fn}"
            print(f"\n  OUTPUT VIDEO: {p}")
