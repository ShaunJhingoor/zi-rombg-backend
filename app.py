# app.py — FastAPI + Robust Video Matting (RVM, resnet50), minimal processing

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ===================== CONFIG =====================
DEFAULT_FPS = 20          # for extraction / re-encode
DOWNSAMPLE_RATIO = 0.10   # lower = better quality, slower
WARMUP_STEPS     = 6      # extra passes on first frame to stabilize

# No aggressive cleanup anymore:
CLAMP_BELOW = 0           # 0 = do NOT kill weak alpha
SHRINK_PX   = 0           # 0 = do NOT erode matte
BLUR_SIGMA  = 0.0         # 0 = no blur; raw edges from model


# ===================== SIMPLE FFMPEG WRAPPERS =====================

def ffmpeg_run(args):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"] + args
    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_frames(in_path: Path, frames_dir: Path, fps: int = DEFAULT_FPS):
    """
    Extract RGBA PNG frames from input video using ffmpeg.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_run([
        "-i", str(in_path),
        "-vf", f"fps={fps},format=rgba",
        "-pix_fmt", "rgba",
        str(frames_dir / "frame_%05d.png"),
    ])


def encode_rgba_to_mov(frames_dir: Path, out_path: Path, fps: int = DEFAULT_FPS):
    """
    Encode RGBA PNG frames into a .mov with alpha (ProRes 4444).
    """
    ffmpeg_run([
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "prores_ks",
        "-profile:v", "4444",
        "-pix_fmt", "yuva444p10le",
        str(out_path),
    ])


# ===================== LOAD RVM MODEL (RESNET50) =====================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("➡ Using device:", device)

try:
    print("🔄 Loading RVM (resnet50)...")
    rvm_model = torch.hub.load(
        "PeterL1n/RobustVideoMatting",
        "resnet50",        # higher quality than mobilenetv3
        trust_repo=True,
    ).to(device).eval()
    print("✅ RVM loaded.")
except Exception as e:
    print("❌ Failed to load RVM:", e)
    rvm_model = None


def ensure_rvm_ready():
    if rvm_model is None:
        raise HTTPException(status_code=500, detail="RVM model failed to load on server startup")


# ===================== RVM FRAME PIPELINE (NO FANCY CLEANUP) =====================

def run_rvm_on_frames(
    frames_dir: Path,
    out_dir: Path,
    downsample_ratio: float = DOWNSAMPLE_RATIO,
    clamp_below: int = CLAMP_BELOW,
    shrink_px: int = SHRINK_PX,
    blur_sigma: float = BLUR_SIGMA,
    warmup_steps: int = WARMUP_STEPS,
):
    """
    Run RVM on frames with minimal modifications:
      - warmup on first frame (stabilize recurrence)
      - optional *gentle* alpha tweaks (currently all disabled)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frames_dir.glob("frame_*.png"))

    if not frame_paths:
        raise RuntimeError("No frames extracted")

    rec = [None, None, None, None]

    with torch.no_grad():
        for idx, fpath in enumerate(frame_paths):
            img = Image.open(fpath).convert("RGB")
            rgb = np.array(img).astype(np.float32) / 255.0

            # RVM input tensor
            src = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

            # warmup on first frame
            if idx == 0:
                for _ in range(warmup_steps):
                    fgr, pha, *rec = rvm_model(src, *rec, downsample_ratio=downsample_ratio)
            else:
                fgr, pha, *rec = rvm_model(src, *rec, downsample_ratio=downsample_ratio)

            # back to numpy
            fgr_np = (fgr[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            alpha = (pha[0, 0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            # ---------- MINIMAL ALPHA TWEAKS (ALL VERY GENTLE / OFF) ----------

            # 1) optional noise clamp
            if clamp_below > 0:
                alpha[alpha < clamp_below] = 0

            # 2) optional tiny shrink
            if shrink_px > 0:
                fg = (alpha > 0).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fg_shrunk = cv2.erode(fg, kernel, iterations=shrink_px)
                alpha[(fg == 1) & (fg_shrunk == 0)] = 0

            # 3) optional smoothing
            if blur_sigma > 0:
                alpha = cv2.GaussianBlur(alpha, (0, 0), blur_sigma)

            # Compose RGBA with *raw* alpha
            rgba = np.dstack([fgr_np, alpha])
            Image.fromarray(rgba, "RGBA").save(out_dir / fpath.name)


# ===================== FASTAPI APP =====================

app = FastAPI(title="AI Video BG Remover – RVM (minimal)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-rembg.vercel.app/"],    # tighten for production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok" if rvm_model else "error",
        "device": device,
    }


# ===================== /process ENDPOINT =====================
@app.get("/health")
def health():
    return {
        "rvm_status": "ok" if rvm_model else "error",
        "device": device,
    }


@app.post("/process")
async def process(video: UploadFile = File(...)):
    ensure_rvm_ready()

    if not video.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    print("📥 Uploaded:", video.filename)

    with tempfile.TemporaryDirectory() as tdir_str:
        tdir = Path(tdir_str)
        input_path = tdir / video.filename
        frames_dir = tdir / "frames"
        out_frames_dir = tdir / "nobg"
        out_video = tdir / f"{input_path.stem}_nobg.mov"

        # Save upload
        with open(input_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        print("➡ Saved to:", input_path)

        # 1) Extract frames
        print("🎞 Extracting frames...")
        extract_frames(input_path, frames_dir, fps=DEFAULT_FPS)

        # 2) Run RVM with minimal tweaks
        print("🧠 Running RVM (minimal cleanup)...")
        run_rvm_on_frames(
            frames_dir,
            out_frames_dir,
            downsample_ratio=DOWNSAMPLE_RATIO,
            clamp_below=CLAMP_BELOW,
            shrink_px=SHRINK_PX,
            blur_sigma=BLUR_SIGMA,
            warmup_steps=WARMUP_STEPS,
        )

        # 3) Encode to MOV with alpha
        print("🎬 Encoding output video...")
        encode_rgba_to_mov(out_frames_dir, out_video, fps=DEFAULT_FPS)

        print("✅ DONE:", out_video)

        # 4) Return file
        final_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mov")
        final_tmp.close()
        shutil.copyfile(out_video, final_tmp.name)

        return FileResponse(
            path=final_tmp.name,
            media_type="video/quicktime",
            filename="bg_removed.mov",
            headers={"Cache-Control": "no-store"},
        )
