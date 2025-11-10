import os, shutil, subprocess, tempfile
from pathlib import Path
from typing import Deque, Optional, Tuple, List
from collections import deque

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from rembg import remove, new_session

# =============================================================
# CONFIG
# =============================================================
DEFAULT_FPS = 20
EDGE_SHRINK_PX = 10
EDGE_BAND_PX = 4
INPAINT_RADIUS = 5
CHROMA_GREEN = (16, 255, 16)

app = FastAPI(title="AI Video BG Remover – Stable Tight Matte")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5173", "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================
# UTILITIES
# =============================================================
def which(cmd: str) -> bool:
    from shutil import which as _which
    return _which(cmd) is not None

def ffmpeg(args: List[str]):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"] + args
    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=True)

def safe_extract_rgba(in_path: Path, frames_dir: Path, fps: int = DEFAULT_FPS):
    ffmpeg([
        "-i", str(in_path),
        "-vf", f"fps={fps},format=rgba",
        "-pix_fmt", "rgba",
        str(frames_dir / "frame_%05d.png"),
    ])

def mask_from(img: Image.Image, session):
    m = remove(img, session=session, only_mask=True,
               alpha_matting=True,
               alpha_matting_foreground_threshold=240,
               alpha_matting_background_threshold=10,
               alpha_matting_erode_size=6)
    return np.array(m, dtype=np.uint8)

# =============================================================
# PYMATTING COMPATIBILITY SHIM
# =============================================================
import inspect
HAS_PYMATTING = False
try:
    from pymatting import estimate_alpha_cf as _estimate_alpha_cf
    HAS_PYMATTING = True

    def estimate_alpha_cf_compat(I, T, reg=1e-5, wr=1):
        """
        Compatibility wrapper for all PyMatting API variants.
        """
        sig = inspect.signature(_estimate_alpha_cf)
        params = sig.parameters

        # new API
        if "regularization" in params or "window_radius" in params:
            try:
                return _estimate_alpha_cf(I, T,
                                          regularization=reg,
                                          window_radius=wr)
            except TypeError:
                pass

        # mid API
        if "lambda_" in params or "lmbda" in params:
            try:
                kw = {}
                if "lambda_" in params: kw["lambda_"] = reg
                if "lmbda" in params: kw["lmbda"] = reg
                if "window_radius" in params: kw["window_radius"] = wr
                return _estimate_alpha_cf(I, T, **kw)
            except TypeError:
                pass

        # legacy API (preconditioner/laplacian_kwargs)
        if "laplacian_kwargs" in params:
            lap_kwargs = {"radius": wr, "epsilon": 1e-7}
            try:
                return _estimate_alpha_cf(I, T,
                                          preconditioner=None,
                                          laplacian_kwargs=lap_kwargs,
                                          cg_kwargs=None)
            except TypeError:
                pass

        # oldest fallback
        try:
            return _estimate_alpha_cf(I, T, reg, wr)
        except Exception:
            return _estimate_alpha_cf(I, T)
except Exception:
    def estimate_alpha_cf_compat(I, T, reg=1e-5, wr=1):
        return None

# =============================================================
# REFINEMENT FUNCTIONS
# =============================================================
def inward_offset_mask(alpha_u8: np.ndarray, px: int) -> np.ndarray:
    fg = (alpha_u8 > 0).astype(np.uint8)
    if fg.max() == 0: return alpha_u8
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    return (dist > float(px)).astype(np.uint8) * 255

def edge_band(alpha_u8: np.ndarray, inner_px: int, band_px: int) -> np.ndarray:
    fg = (alpha_u8 > 0).astype(np.uint8)
    if fg.max() == 0: return np.zeros_like(alpha_u8, np.uint8)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    return ((dist > float(inner_px)) & (dist <= float(inner_px + band_px))).astype(np.uint8) * 255

def edge_inpaint_inward(rgb_u8: np.ndarray, ring_mask_u8: np.ndarray, radius: int) -> np.ndarray:
    if ring_mask_u8.max() == 0: return rgb_u8
    return cv2.inpaint(rgb_u8, ring_mask_u8, radius, cv2.INPAINT_TELEA)

def closed_form_refine(rgb_u8: np.ndarray, alpha_u8: np.ndarray,
                       inner_px: int, band_px: int) -> np.ndarray:
    """
    Closed-form matting refinement with compatibility across PyMatting versions.
    """
    if not HAS_PYMATTING:
        return alpha_u8

    h, w = alpha_u8.shape
    trimap = np.zeros_like(alpha_u8, dtype=np.uint8)
    trimap[alpha_u8 >= 240] = 255
    trimap[(alpha_u8 > 0) & (alpha_u8 < 240)] = 128

    # early exit if no ambiguous pixels
    if (trimap == 128).sum() < 20:
        return alpha_u8

    I = rgb_u8.astype(np.float64) / 255.0
    T = trimap.astype(np.float64) / 255.0

    try:
        alpha_ref = estimate_alpha_cf_compat(I, T, reg=1e-5, wr=1)
        if alpha_ref is None:
            return alpha_u8
        return np.clip(alpha_ref * 255.0, 0, 255).astype(np.uint8)
    except Exception as e:
        print("Closed-form refine failed:", e)
        return alpha_u8

# =============================================================
# MAIN ROUTE
# =============================================================
SESSION_HUMAN = new_session("u2net_human_seg")

@app.post("/process")
async def process(video: UploadFile = File(...)):
    if not which("ffmpeg"):
        raise HTTPException(500, "ffmpeg not found")

    with tempfile.TemporaryDirectory() as tdir_str:
        tdir = Path(tdir_str)
        in_path = tdir / video.filename
        frames_dir = tdir / "frames"
        out_dir = tdir / "nobg"
        frames_dir.mkdir(); out_dir.mkdir()

        with open(in_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        safe_extract_rgba(in_path, frames_dir, DEFAULT_FPS)
        frame_paths = sorted(frames_dir.glob("frame_*.png"))
        if not frame_paths:
            raise RuntimeError("no frames extracted")

        prev_rgb = None
        for fpath in frame_paths:
            img = Image.open(fpath).convert("RGBA")
            np_rgba = np.array(img)
            rgb = np_rgba[:, :, :3].astype(np.uint8)
            mask = mask_from(img, SESSION_HUMAN)

            # shrink + refine
            mask_in = inward_offset_mask(mask, EDGE_SHRINK_PX)
            ring = edge_band(mask_in, EDGE_SHRINK_PX, EDGE_BAND_PX)
            rgb_fixed = edge_inpaint_inward(rgb, ring, INPAINT_RADIUS)

            # refine matte edges
            refined = closed_form_refine(rgb_fixed, mask_in, EDGE_SHRINK_PX, EDGE_BAND_PX)

            out_rgba = np.dstack([rgb_fixed, refined])
            Image.fromarray(out_rgba, "RGBA").save(out_dir / fpath.name)
            prev_rgb = rgb.copy()

        out_mov = tdir / f"{in_path.stem}_nobg.mov"
        ffmpeg([
            "-framerate", str(DEFAULT_FPS),
            "-i", str(out_dir / "frame_%05d.png"),
            "-c:v", "prores_ks", "-profile:v", "4444",
            "-pix_fmt", "yuva444p10le",
            str(out_mov),
        ])

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mov"); tmp.close()
        shutil.copyfile(out_mov, tmp.name)
        return FileResponse(
            path=tmp.name, media_type="video/quicktime",
            filename=Path(tmp.name).name,
            background=BackgroundTask(lambda p: os.path.exists(p) and os.remove(p), tmp.name)
        )
