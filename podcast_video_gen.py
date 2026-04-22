"""
Auto-generate podcast video from isotope daily podcast.
Splits audio into 8s segments, generates video for each, concatenates.
Usage: python podcast_video_gen.py [date] [max_segments]
  date: YYYY-MM-DD (default: today)
  max_segments: limit segments to generate (default: all)
"""
import json, urllib.request, urllib.error, websocket, uuid, time, sys, os, subprocess, shutil, wave
from datetime import datetime
from PIL import Image

SERVER = "127.0.0.1:8288"
CLIENT_ID = str(uuid.uuid4())
SIGMAS_15 = "1., 0.996, 0.991, 0.986, 0.98, 0.97, 0.95, 0.92, 0.87, 0.8, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0"
SEGMENT_DURATION = 8.0
OUTPUT_DIR = r"E:\ComfyUI-aki-v2\ComfyUI\output\podcast_video"

SLIM_PROMPT = "xixi, a young Chinese woman, 22 years old, bangs and long dark brown hair with hair clip, delicate slim oval face, large bright brown eyes, small nose, youthful appearance, fair skin, sitting at a cafe table, wearing a cozy cream cardigan over white top, speaking to camera, mouth moving while talking, warm cafe lighting with plants, close-up portrait, natural soft expression"
NEG_PROMPT = "blurry, out of focus, overexposed, underexposed, noise, grainy, poor lighting, flickering, motion blur, distorted, unnatural skin, dark, grey, dull, watermark, text, logo, subtitle, fat face, chubby, wide face, puffy"


def build_prompt(start_time, duration, seed, prefix):
    """Build API prompt for one segment."""
    return {
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
        "10": {"class_type": "LoadAudio", "inputs": {"audio": "podcast_today.wav"}},
        "11": {"class_type": "MelBandRoFormerModelLoader",
               "inputs": {"model_name": "MelBandRoformer_fp16.safetensors"}},
        "12": {"class_type": "MelBandRoFormerSampler",
               "inputs": {"audio": ["10", 0], "model": ["11", 0]}},
        "13": {"class_type": "TrimAudioDuration",
               "inputs": {"audio": ["12", 0], "start_index": start_time, "duration": duration}},
        "14": {"class_type": "LTXVAudioVAELoader",
               "inputs": {"ckpt_name": "ltx-2.3-22b-dev-nvfp4.safetensors"}},
        "15": {"class_type": "LTXVAudioVAEEncode",
               "inputs": {"audio": ["13", 0], "audio_vae": ["14", 0]}},
        "20": {"class_type": "LoadImage", "inputs": {"image": "xixi_first_frame.png"}},
        "21": {"class_type": "LTXVPreprocess", "inputs": {"image": ["20", 0], "img_compression": 25}},
        "30": {"class_type": "EmptyLTXVLatentVideo",
               "inputs": {"width": 1024, "height": 576, "length": 193, "batch_size": 1}},
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
               "inputs": {"value": 1.0, "width": 128, "height": 72}},
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
        "50": {"class_type": "LTXVSeparateAVLatent",
               "inputs": {"av_latent": ["44", 0]}},
        "51": {"class_type": "VAEDecodeTiled",
               "inputs": {"samples": ["50", 0], "vae": ["1", 2],
                          "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 64}},
        "52": {"class_type": "LTXVAudioVAEDecode",
               "inputs": {"samples": ["50", 1], "audio_vae": ["14", 0]}},
        "60": {"class_type": "CreateVideo",
               "inputs": {"images": ["51", 0], "fps": 24.0, "audio": ["52", 0]}},
        "61": {"class_type": "SaveVideo",
               "inputs": {"video": ["60", 0], "filename_prefix": prefix,
                          "format": "mp4", "codec": "h264"}},
    }


def run_segment(prompt_data):
    data = json.dumps({"prompt": prompt_data, "client_id": CLIENT_ID}).encode("utf-8")
    try:
        req = urllib.request.Request(f"http://{SERVER}/prompt", data=data)
        resp = json.loads(urllib.request.urlopen(req).read())
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
                elif d.get("type") == "executing" and d["data"].get("node") is None:
                    print(f" -> Done ({time.time()-t0:.0f}s)")
                    break
                elif d.get("type") == "execution_error":
                    print(f"\n  ERROR: {d.get('data',{}).get('exception_message','')[:300]}")
                    return False
    finally:
        ws.close()
    return True


def main():
    # Parse args
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    max_segments = int(sys.argv[2]) if len(sys.argv) > 2 else 999
    global GLOBAL_SEED
    GLOBAL_SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 8001

    podcast_path = rf"D:\Projects\isotope\data\podcast\podcast_{date_str}.wav"
    if not os.path.exists(podcast_path):
        print(f"Podcast not found: {podcast_path}")
        sys.exit(1)

    # Get duration
    with wave.open(podcast_path) as w:
        total_duration = w.getnframes() / w.getframerate()
    num_segments = min(int(total_duration / SEGMENT_DURATION) + 1, max_segments)
    print(f"Podcast: {date_str} ({total_duration:.0f}s)")
    print(f"Segments: {num_segments} x {SEGMENT_DURATION}s")
    print(f"Estimated time: {num_segments * 6:.0f} minutes\n")

    # Copy podcast to ComfyUI input
    shutil.copy2(podcast_path, r"E:\ComfyUI-aki-v2\ComfyUI\input\podcast_today.wav")

    # Setup first frame
    orig = Image.open(r"C:\Users\admin\Pictures\曦曦\lora\xixi_32.png")
    img = orig.copy()
    ratio = max(1024 / img.size[0], 576 / img.size[1])
    nw, nh = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - 1024) // 2, (nh - 576) // 2
    img.crop((left, top, left + 1024, top + 576)).save(
        r"E:\ComfyUI-aki-v2\ComfyUI\input\xixi_first_frame.png")

    # Wait for server
    urllib.request.urlopen(f"http://{SERVER}/", timeout=5)

    # Create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate segments
    segment_files = []
    for i in range(num_segments):
        start = i * SEGMENT_DURATION
        remaining = total_duration - start
        dur = min(SEGMENT_DURATION, remaining)
        if dur < 2:
            break

        seed = GLOBAL_SEED  # Same seed for all segments to keep face consistent
        prefix = f"podcast_video/seg_{i:03d}"
        print(f"[{i+1}/{num_segments}] Segment {i}: {start:.0f}s-{start+dur:.0f}s (seed={seed})")

        prompt = build_prompt(start, dur, seed, prefix)
        success = run_segment(prompt)

        if success:
            seg_file = os.path.join(r"E:\ComfyUI-aki-v2\ComfyUI\output\podcast_video",
                                    f"seg_{i:03d}_00001_.mp4")
            if os.path.exists(seg_file):
                segment_files.append(seg_file)
                print(f"  Saved: {seg_file}")
        else:
            print(f"  Segment {i} failed, skipping")

    if not segment_files:
        print("No segments generated!")
        return

    # Concatenate all segments using ffmpeg
    concat_list = os.path.join(OUTPUT_DIR, "concat_list.txt")
    with open(concat_list, "w") as f:
        for sf in segment_files:
            f.write(f"file '{sf}'\n")

    final_output = os.path.join(OUTPUT_DIR, f"podcast_{date_str}_seed{GLOBAL_SEED}.mp4")
    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list,
        "-c", "copy", final_output, "-y"
    ], capture_output=True)

    if os.path.exists(final_output):
        size_mb = os.path.getsize(final_output) / 1e6
        print(f"\n=== FINAL VIDEO ===")
        print(f"  {final_output}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Segments: {len(segment_files)}")
        os.startfile(final_output)
    else:
        print("Failed to concatenate segments")


if __name__ == "__main__":
    main()
