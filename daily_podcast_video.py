"""
Daily Podcast Video Generator
Complete pipeline: LTX-Video generation -> LatentSync lip sync
Usage: python daily_podcast_video.py [date]
"""
import os, sys, subprocess, time, shutil, glob
from datetime import datetime
from config import *

def kill_process_on_port(port):
    result = subprocess.run(f'netstat -ano | findstr {port} | findstr LISTENING',
                           shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        pid = result.stdout.strip().split()[-1]
        os.system(f'taskkill /PID {pid} /F')
        time.sleep(2)

def wait_for_comfyui(timeout=120):
    import urllib.request
    for _ in range(timeout // 3):
        try:
            urllib.request.urlopen(f"http://{COMFYUI_SERVER}/", timeout=2)
            return True
        except:
            time.sleep(3)
    return False

def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    podcast_path = os.path.join(PODCAST_DIR, f"podcast_{date_str}.wav")

    if not os.path.exists(podcast_path):
        print(f"ERROR: Podcast not found: {podcast_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Daily Podcast Video Generator")
    print(f"  Date: {date_str}")
    print(f"{'='*60}")

    # === Phase 1: Generate video segments ===
    print(f"\n[Phase 1] LTX-Video Generation")
    for f in glob.glob(os.path.join(OUTPUT_DIR, "seg_*.mp4")):
        os.remove(f)

    kill_process_on_port(COMFYUI_PORT)
    comfyui_proc = subprocess.Popen(
        [COMFYUI_PYTHON, os.path.join(COMFYUI_DIR, "main.py"),
         "--port", str(COMFYUI_PORT), "--preview-method", "auto", "--disable-cuda-malloc"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    if not wait_for_comfyui():
        print("ERROR: ComfyUI failed to start")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(
        [COMFYUI_PYTHON, os.path.join(script_dir, "podcast_video_gen.py"),
         date_str, "999", str(SEED)],
        timeout=7200
    )

    kill_process_on_port(COMFYUI_PORT)
    if comfyui_proc.poll() is None:
        comfyui_proc.terminate()
    time.sleep(3)

    raw_video = os.path.join(OUTPUT_DIR, f"podcast_{date_str}_seed{SEED}.mp4")
    if not os.path.exists(raw_video):
        print(f"ERROR: No video at {raw_video}")
        sys.exit(1)

    # === Phase 2: LatentSync lip sync ===
    print(f"\n[Phase 2] LatentSync Lip Sync")
    lipsync_input = os.path.join(COMFYUI_DIR, "output", "lipsync_input.mp4")
    shutil.copy2(raw_video, lipsync_input)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    subprocess.run(
        [COMFYUI_PYTHON, os.path.join(script_dir, "run_lipsync_standalone.py")],
        env=env, timeout=3600
    )

    lipsync_output = os.path.join(COMFYUI_DIR, "output", "lipsync_standalone_result.mp4")
    if not os.path.exists(lipsync_output):
        print("WARNING: LatentSync failed, using raw video")
        lipsync_output = raw_video

    # === Phase 3: Output ===
    final_path = os.path.join(FINAL_OUTPUT_DIR, f"podcast_video_{date_str}.mp4")
    shutil.copy2(lipsync_output, final_path)

    print(f"\n{'='*60}")
    print(f"  DONE! {final_path}")
    print(f"  Size: {os.path.getsize(final_path)/1e6:.1f}MB")
    print(f"{'='*60}")
    os.startfile(final_path)


if __name__ == "__main__":
    main()
