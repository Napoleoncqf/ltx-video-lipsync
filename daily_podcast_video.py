"""
Daily Podcast Video Generator for isotope (曦曦)
Complete pipeline: LTX-Video generation → LatentSync lip sync
Usage: python daily_podcast_video.py [date]
  date: YYYY-MM-DD (default: today)
"""
import os, sys, subprocess, time, shutil, json, glob
from datetime import datetime

# === Configuration ===
COMFYUI_DIR = r"E:\ComfyUI-aki-v2\ComfyUI"
COMFYUI_PYTHON = r"E:\ComfyUI-aki-v2\python\python.exe"
COMFYUI_MAIN = os.path.join(COMFYUI_DIR, "main.py")
SCRIPT_DIR = r"E:\夸克下载\comfy\LTX2.3\一键口播"
PODCAST_DIR = r"D:\Projects\isotope\data\podcast"
OUTPUT_DIR = os.path.join(COMFYUI_DIR, "output", "podcast_video")
FINAL_OUTPUT_DIR = r"D:\Projects\isotope\data\podcast_video"

SEED = 8042  # Best seed for lip sync
SEGMENT_DURATION = 8.0
MAX_SEGMENTS = 999  # All segments

def get_date():
    return sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")

def kill_process_on_port(port):
    result = subprocess.run(f'netstat -ano | findstr {port} | findstr LISTENING',
                           shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        pid = result.stdout.strip().split()[-1]
        os.system(f'taskkill /PID {pid} /F')
        print(f"  Killed process on port {port} (PID {pid})")
        time.sleep(2)

def wait_for_comfyui(timeout=120):
    import urllib.request
    for _ in range(timeout // 3):
        try:
            urllib.request.urlopen("http://127.0.0.1:8288/", timeout=2)
            return True
        except:
            time.sleep(3)
    return False

def main():
    date_str = get_date()
    podcast_path = os.path.join(PODCAST_DIR, f"podcast_{date_str}.wav")

    if not os.path.exists(podcast_path):
        print(f"ERROR: Podcast not found: {podcast_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

    print(f"=" * 60)
    print(f"  Daily Podcast Video Generator")
    print(f"  Date: {date_str}")
    print(f"  Podcast: {podcast_path}")
    print(f"=" * 60)

    # === Phase 1: Generate video segments with LTX-Video ===
    print(f"\n{'='*60}")
    print(f"  Phase 1: LTX-Video Generation")
    print(f"{'='*60}")

    # Clean old segments
    for f in glob.glob(os.path.join(OUTPUT_DIR, "seg_*.mp4")):
        os.remove(f)

    # Start ComfyUI
    print("\n[1/6] Starting ComfyUI...")
    kill_process_on_port(8288)
    comfyui_proc = subprocess.Popen(
        [COMFYUI_PYTHON, COMFYUI_MAIN, "--port", "8288", "--preview-method", "auto", "--disable-cuda-malloc"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    if not wait_for_comfyui():
        print("ERROR: ComfyUI failed to start")
        sys.exit(1)
    print("  ComfyUI ready!")

    # Run podcast video generation
    print("\n[2/6] Generating video segments...")
    result = subprocess.run(
        [COMFYUI_PYTHON, os.path.join(SCRIPT_DIR, "podcast_video_gen.py"),
         date_str, str(MAX_SEGMENTS), str(SEED)],
        capture_output=False, timeout=7200  # 2 hour timeout
    )

    # Stop ComfyUI to free VRAM
    print("\n[3/6] Stopping ComfyUI...")
    kill_process_on_port(8288)
    if comfyui_proc.poll() is None:
        comfyui_proc.terminate()
    time.sleep(3)

    # Check generated video
    raw_video = os.path.join(OUTPUT_DIR, f"podcast_{date_str}_seed{SEED}.mp4")
    if not os.path.exists(raw_video):
        print(f"ERROR: Video generation failed, no output at {raw_video}")
        sys.exit(1)

    raw_size = os.path.getsize(raw_video) / 1e6
    print(f"  Raw video: {raw_video} ({raw_size:.1f}MB)")

    # === Phase 2: LatentSync lip sync ===
    print(f"\n{'='*60}")
    print(f"  Phase 2: LatentSync Lip Sync")
    print(f"{'='*60}")

    # Copy video as lipsync input
    lipsync_input = os.path.join(COMFYUI_DIR, "output", "lipsync_input.mp4")
    shutil.copy2(raw_video, lipsync_input)

    # Run LatentSync standalone
    print("\n[4/6] Running LatentSync 1.6 (512px)...")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [COMFYUI_PYTHON, os.path.join(SCRIPT_DIR, "run_lipsync_standalone.py")],
        env=env, capture_output=False, timeout=3600  # 1 hour timeout
    )

    lipsync_output = os.path.join(COMFYUI_DIR, "output", "lipsync_standalone_result.mp4")
    if not os.path.exists(lipsync_output):
        print("WARNING: LatentSync failed, using raw video instead")
        lipsync_output = raw_video

    # === Phase 3: Final output ===
    print(f"\n{'='*60}")
    print(f"  Phase 3: Final Output")
    print(f"{'='*60}")

    # Copy to final location
    final_path = os.path.join(FINAL_OUTPUT_DIR, f"podcast_video_{date_str}.mp4")
    shutil.copy2(lipsync_output, final_path)
    final_size = os.path.getsize(final_path) / 1e6

    print(f"\n[5/6] Final video: {final_path}")
    print(f"  Size: {final_size:.1f}MB")

    # Open video
    print("\n[6/6] Opening video...")
    os.startfile(final_path)

    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Date: {date_str}")
    print(f"  Output: {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
