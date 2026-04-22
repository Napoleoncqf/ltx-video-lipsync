"""
Best config: 1280x720, full AV decode for lip sync, MelBandRoFormer, seed2.
"""
import json, urllib.request, urllib.error, websocket, uuid, time, sys, os
from PIL import Image

SERVER = "127.0.0.1:8288"
CLIENT_ID = str(uuid.uuid4())
SIGMAS_15 = "1., 0.996, 0.991, 0.986, 0.98, 0.97, 0.95, 0.92, 0.87, 0.8, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0"

SLIM_PROMPT = "xixi, a young Chinese woman, 22 years old, bangs and long dark brown hair with hair clip, delicate slim oval face, large bright brown eyes, small nose, youthful appearance, fair skin, sitting at a cafe table, wearing a cozy cream cardigan over white top, speaking to camera, mouth moving while talking, warm cafe lighting with plants, close-up portrait, natural soft expression"
NEG_PROMPT = "blurry, out of focus, overexposed, underexposed, noise, grainy, poor lighting, flickering, motion blur, distorted, unnatural skin, dark, grey, dull, watermark, text, logo, subtitle, fat face, chubby, wide face, puffy"

def build_prompt(w, h, seed, prefix, use_full_av_decode=True, use_vocal_sep=True):
    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors"}},
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": "xixi_ltx23_2000.safetensors", "strength_model": 0.85}},
        "3": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["2", 0],
                         "lora_name": "LTX\\ltx-2.3-22b-distilled-lora-resized_dynamic_rank_159_fro09_bf16.safetensors",
                         "strength_model": 0.3}},
        "4": {"class_type": "LTXAVTextEncoderLoader",
              "inputs": {"text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
                         "ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors", "device": "default"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": SLIM_PROMPT}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": NEG_PROMPT}},
        # Audio with vocal separation
        "10": {"class_type": "LoadAudio", "inputs": {"audio": "podcast_audio.wav"}},
    }

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
               "inputs": {"audio": audio_src, "start_index": 0.0, "duration": 8.0}},
        "14": {"class_type": "LTXVAudioVAELoader",
               "inputs": {"ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors"}},
        "15": {"class_type": "LTXVAudioVAEEncode",
               "inputs": {"audio": ["13", 0], "audio_vae": ["14", 0]}},
        "20": {"class_type": "LoadImage", "inputs": {"image": "xixi_first_frame.png"}},
        "21": {"class_type": "LTXVPreprocess", "inputs": {"image": ["20", 0], "img_compression": 25}},
        "30": {"class_type": "EmptyLTXVLatentVideo",
               "inputs": {"width": w, "height": h, "length": 193, "batch_size": 1}},
        "31": {"class_type": "LTXVConditioning",
               "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": 24}},
        "32": {"class_type": "LTXVAddGuide",
               "inputs": {"positive": ["31", 0], "negative": ["31", 1], "vae": ["1", 2],
                          "latent": ["30", 0], "image": ["21", 0],
                          "frame_idx": 0, "strength": 0.7}},
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
    })

    if use_full_av_decode:
        # Full AV decode: separate video+audio latent, decode each properly
        prompt.update({
            "50": {"class_type": "LTXVSeparateAVLatent",
                   "inputs": {"av_latent": ["44", 0]}},
            "51": {"class_type": "VAEDecodeTiled",
                   "inputs": {"samples": ["50", 0], "vae": ["1", 2],
                              "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 64}},
            "52": {"class_type": "LTXVAudioVAEDecode",
                   "inputs": {"samples": ["50", 1], "audio_vae": ["14", 0]}},
            "60": {"class_type": "CreateVideo",
                   "inputs": {"images": ["51", 0], "fps": 24.0, "audio": ["52", 0]}},
        })
    else:
        # Fallback: strip noise mask + VAEDecode + raw audio
        prompt.update({
            "48": {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 1, "height": 1}},
            "49": {"class_type": "SetLatentNoiseMask",
                   "inputs": {"samples": ["44", 0], "mask": ["48", 0]}},
            "51": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["49", 0], "vae": ["1", 2]}},
            "60": {"class_type": "CreateVideo",
                   "inputs": {"images": ["51", 0], "fps": 24.0, "audio": ["13", 0]}},
        })

    prompt["61"] = {"class_type": "SaveVideo",
                    "inputs": {"video": ["60", 0], "filename_prefix": prefix,
                               "format": "mp4", "codec": "h264"}}
    return prompt


def run_test(prompt_data):
    data = json.dumps({"prompt": prompt_data, "client_id": CLIENT_ID}).encode("utf-8")
    try:
        req = urllib.request.Request(f"http://{SERVER}/prompt", data=data)
        resp = json.loads(urllib.request.urlopen(req).read())
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
                    # Show key node names
                    ct = prompt_data.get(nid, {}).get("class_type", "")
                    if ct in ["SamplerCustomAdvanced", "VAEDecodeTiled", "LTXVSeparateAVLatent",
                              "LTXVAudioVAEDecode", "MelBandRoFormerSampler"]:
                        print(f"  [{time.time()-t0:.0f}s] {ct}")
                elif d.get("type") == "execution_error":
                    print(f"\n  ERROR: {d.get('data',{}).get('exception_message','')[:300]}")
                    return False
    finally:
        ws.close()
    return True


# Setup first frame for 1280x720
orig = Image.open(r"C:\Users\admin\Pictures\曦曦\lora\xixi_32.png")
for w, h in [(1280, 720), (1024, 576)]:
    img = orig.copy()
    ratio = max(w / img.size[0], h / img.size[1])
    nw, nh = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    img.crop((left, top, left + w, top + h)).save(
        rf"E:\ComfyUI-aki-v2\ComfyUI\input\xixi_frame_{w}x{h}.png")

urllib.request.urlopen(f"http://{SERVER}/", timeout=5)
print("Ready!\n")

import shutil
TESTS = [
    # 1280x720 + full AV decode + vocal sep (best lip sync attempt)
    ("best_720p_avsync", 1280, 720, 8042, True, True),
    # 1280x720 + fallback decode + vocal sep
    ("best_720p_fallback", 1280, 720, 8042, False, True),
    # 1024x576 + full AV decode + vocal sep (compare resolution)
    ("best_576p_avsync", 1024, 576, 8042, True, True),
    # 1024x576 seed1 + full AV decode (best face)
    ("best_576p_seed1", 1024, 576, 8001, True, True),
]

for i, (name, w, h, seed, full_av, vocal_sep) in enumerate(TESTS):
    shutil.copy2(rf"E:\ComfyUI-aki-v2\ComfyUI\input\xixi_frame_{w}x{h}.png",
                 r"E:\ComfyUI-aki-v2\ComfyUI\input\xixi_first_frame.png")
    av_str = "full_AV" if full_av else "fallback"
    print(f"[{i+1}/{len(TESTS)}] {name} ({w}x{h}, {av_str}, seed={seed})")
    prompt = build_prompt(w, h, seed, name, full_av, vocal_sep)
    success = run_test(prompt)
    if not success and full_av:
        print("  Full AV failed, trying fallback...")
        prompt = build_prompt(w, h, seed, name + "_fb", False, vocal_sep)
        run_test(prompt)

# Open all
import subprocess, glob
for mp4 in glob.glob(r"E:\ComfyUI-aki-v2\ComfyUI\output\best_*_00001_.mp4"):
    subprocess.run(['ffmpeg','-i', mp4,'-vf',f'select=eq(n\\,72)','-vframes','1',
        mp4.replace('.mp4','_mid.png'),'-y'], capture_output=True)
    os.startfile(mp4)

print("\nAll done!")
