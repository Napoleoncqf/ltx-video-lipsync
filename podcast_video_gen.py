"""
Auto-generate podcast video from audio file.
Splits audio into segments, generates video for each, concatenates.
Usage: python podcast_video_gen.py [date] [max_segments] [seed]
"""
import json, urllib.request, urllib.error, websocket, uuid, time, sys, os, subprocess, shutil, wave
from datetime import datetime
from PIL import Image
from config import *

CLIENT_ID = str(uuid.uuid4())


def build_prompt(start_time, duration, seed, prefix):
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": CHECKPOINT}},
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": CHARACTER_LORA, "strength_model": CHARACTER_LORA_STRENGTH}},
        "3": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["2", 0], "lora_name": DISTILLED_LORA, "strength_model": DISTILLED_LORA_STRENGTH}},
        "4": {"class_type": "LTXAVTextEncoderLoader",
              "inputs": {"text_encoder": TEXT_ENCODER, "ckpt_name": CHECKPOINT, "device": "default"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": POSITIVE_PROMPT}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": NEGATIVE_PROMPT}},
        "10": {"class_type": "LoadAudio", "inputs": {"audio": "podcast_today.wav"}},
        "13": {"class_type": "TrimAudioDuration",
               "inputs": {"audio": ["10", 0], "start_index": start_time, "duration": duration}},
        "14": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CHECKPOINT}},
        "15": {"class_type": "LTXVAudioVAEEncode",
               "inputs": {"audio": ["13", 0], "audio_vae": ["14", 0]}},
        "20": {"class_type": "LoadImage", "inputs": {"image": "first_frame.png"}},
        "21": {"class_type": "LTXVPreprocess", "inputs": {"image": ["20", 0], "img_compression": 25}},
        "30": {"class_type": "EmptyLTXVLatentVideo",
               "inputs": {"width": WIDTH, "height": HEIGHT, "length": FRAMES_PER_SEGMENT, "batch_size": 1}},
        "31": {"class_type": "LTXVConditioning",
               "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": FRAME_RATE}},
        "32": {"class_type": "LTXVAddGuide",
               "inputs": {"positive": ["31", 0], "negative": ["31", 1], "vae": ["1", 2],
                          "latent": ["30", 0], "image": ["21", 0],
                          "frame_idx": 0, "strength": GUIDE_STRENGTH}},
        "33": {"class_type": "LTXVAddGuide",
               "inputs": {"positive": ["32", 0], "negative": ["32", 1], "vae": ["1", 2],
                          "latent": ["32", 2], "image": ["21", 0],
                          "frame_idx": -1, "strength": GUIDE_STRENGTH}},
        "35": {"class_type": "LTXVConcatAVLatent",
               "inputs": {"video_latent": ["33", 2], "audio_latent": ["15", 0]}},
        "36": {"class_type": "SolidMask",
               "inputs": {"value": 1.0, "width": WIDTH // 8, "height": HEIGHT // 8}},
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
        "50": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["44", 0]}},
        "51": {"class_type": "VAEDecodeTiled",
               "inputs": {"samples": ["50", 0], "vae": ["1", 2],
                          "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 64}},
        "52": {"class_type": "LTXVAudioVAEDecode",
               "inputs": {"samples": ["50", 1], "audio_vae": ["14", 0]}},
        "60": {"class_type": "CreateVideo",
               "inputs": {"images": ["51", 0], "fps": float(FRAME_RATE), "audio": ["52", 0]}},
        "61": {"class_type": "SaveVideo",
               "inputs": {"video": ["60", 0], "filename_prefix": prefix,
                          "format": "mp4", "codec": "h264"}},
    }


def run_segment(prompt_data):
    data = json.dumps({"prompt": prompt_data, "client_id": CLIENT_ID}).encode("utf-8")
    try:
        req = urllib.request.Request(f"http://{COMFYUI_SERVER}/prompt", data=data)
        resp = json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        print(f"  ERROR: {e.read().decode('utf-8', errors='replace')[:300]}")
        return False
    ws = websocket.create_connection(f"ws://{COMFYUI_SERVER}/ws?clientId={CLIENT_ID}")
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
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    max_segments = int(sys.argv[2]) if len(sys.argv) > 2 else 999
    global_seed = int(sys.argv[3]) if len(sys.argv) > 3 else SEED

    podcast_path = os.path.join(PODCAST_DIR, f"podcast_{date_str}.wav")
    if not os.path.exists(podcast_path):
        print(f"Podcast not found: {podcast_path}")
        sys.exit(1)

    with wave.open(podcast_path) as w:
        total_duration = w.getnframes() / w.getframerate()
    num_segments = min(int(total_duration / SEGMENT_DURATION) + 1, max_segments)
    print(f"Podcast: {date_str} ({total_duration:.0f}s)")
    print(f"Segments: {num_segments} x {SEGMENT_DURATION}s\n")

    # Copy podcast to ComfyUI input
    shutil.copy2(podcast_path, os.path.join(COMFYUI_DIR, "input", "podcast_today.wav"))

    # Setup first frame
    if os.path.exists(FIRST_FRAME_SOURCE):
        orig = Image.open(FIRST_FRAME_SOURCE)
    else:
        print(f"WARNING: First frame not found at {FIRST_FRAME_SOURCE}, using default")
        orig = Image.new("RGB", (WIDTH, HEIGHT), (128, 128, 128))
    ratio = max(WIDTH / orig.size[0], HEIGHT / orig.size[1])
    nw, nh = int(orig.size[0] * ratio), int(orig.size[1] * ratio)
    img = orig.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - WIDTH) // 2, (nh - HEIGHT) // 2
    img.crop((left, top, left + WIDTH, top + HEIGHT)).save(
        os.path.join(COMFYUI_DIR, "input", "first_frame.png"))

    # Wait for server
    urllib.request.urlopen(f"http://{COMFYUI_SERVER}/", timeout=5)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate segments
    segment_files = []
    for i in range(num_segments):
        start = i * SEGMENT_DURATION
        remaining = total_duration - start
        dur = min(SEGMENT_DURATION, remaining)
        if dur < 2:
            break

        prefix = f"podcast_video/seg_{i:03d}"
        print(f"[{i+1}/{num_segments}] Segment {i}: {start:.0f}s-{start+dur:.0f}s (seed={global_seed})")

        prompt = build_prompt(start, dur, global_seed, prefix)
        success = run_segment(prompt)

        if success:
            seg_file = os.path.join(OUTPUT_DIR, f"seg_{i:03d}_00001_.mp4")
            if os.path.exists(seg_file):
                segment_files.append(seg_file)
                print(f"  Saved: {seg_file}")
        else:
            print(f"  Segment {i} failed, skipping")

    if not segment_files:
        print("No segments generated!")
        return

    # Concatenate
    concat_list = os.path.join(OUTPUT_DIR, "concat_list.txt")
    with open(concat_list, "w") as f:
        for sf in segment_files:
            f.write(f"file '{sf}'\n")

    final_output = os.path.join(OUTPUT_DIR, f"podcast_{date_str}_seed{global_seed}.mp4")
    subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list,
                    "-c", "copy", final_output, "-y"], capture_output=True)

    if os.path.exists(final_output):
        size_mb = os.path.getsize(final_output) / 1e6
        print(f"\n=== FINAL VIDEO ===")
        print(f"  {final_output}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Segments: {len(segment_files)}")
        os.startfile(final_output)


if __name__ == "__main__":
    main()
