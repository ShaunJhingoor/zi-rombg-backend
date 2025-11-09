import os, json, shutil, subprocess, tempfile, traceback
from pathlib import Path
from typing import Deque, Tuple
from collections import deque

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from rembg import remove, new_session

# ===================== MAX-TIGHT SETTINGS =====================
DEFAULT_FPS       = 20

# temporal
MEDIAN_WINDOW     = 5

# hard inward clamp (aggressive)
EDGE_SHRINK_PX    = 20        # bump up if fringe remains
EDGE_BAND_PX      = 6
INPAINT_RADIUS    = 7

# alpha thresholds
TH_STRONG         = 235
TH_WEAK           = 80
CLAMP_BELOW       = 20

# tiny, edge-aware feather (keep min to avoid halo)
BILAT_DIAM        = 5
BILAT_SIGMA       = 18

# rembg matting
AM_FG_T           = 246
AM_BG_T           = 6
AM_ERODE          = 2

# morphology
CLOSE_ITERS       = 1
DILATE_ITERS      = 0            # do NOT dilate (causes halo growth)
ERODE_ITERS       = 1            # nibble once

# bg gate (Lab) – stricter
LAB_L_DIV         = 10.0
LAB_A_DIV         = 7.0
LAB_B_DIV         = 7.0
LAB_THRESH2       = 1.0

CHROMA_GREEN      = (16, 255, 16)

# sessions
SESSION_HUMAN = new_session("u2net_human_seg")
SESSION_U2NET = new_session("u2net")
SESSION_ISNET = new_session("isnet-general-use")

app = FastAPI(title="AI Video BG Remover – MAX Tight + Decontam")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def which(cmd: str) -> bool:
    from shutil import which as _which
    return _which(cmd) is not None

def ffmpeg(args: list[str]):
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
    m = remove(
        img, session=session, only_mask=True,
        alpha_matting=True,
        alpha_matting_foreground_threshold=AM_FG_T,
        alpha_matting_background_threshold=AM_BG_T,
        alpha_matting_erode_size=AM_ERODE,
    )
    return np.array(m, dtype=np.uint8)

def hysteresis_stabilize(alpha_u8: np.ndarray) -> np.ndarray:
    strong = (alpha_u8 >= TH_STRONG).astype(np.uint8)
    weak   = (alpha_u8 >= TH_WEAK).astype(np.uint8)
    num, labels = cv2.connectedComponents(weak, connectivity=4)
    if num <= 1:
        return (strong * 255).astype(np.uint8)
    keep = np.zeros(num, dtype=np.uint8)
    for lab in range(1, num):
        if np.any((labels == lab) & (strong == 1)):
            keep[lab] = 1
    kept = keep[labels]
    return (kept * 255).astype(np.uint8)

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

def estimate_bg_gate(rgb_u8: np.ndarray) -> np.ndarray:
    h, w, _ = rgb_u8.shape
    sw = max(32, min(h, w) // 6)
    patches = [rgb_u8[0:sw,0:sw], rgb_u8[0:sw,w-sw:w], rgb_u8[h-sw:h,0:sw], rgb_u8[h-sw:h,w-sw:w]]
    samples = np.concatenate([p.reshape(-1,3) for p in patches], axis=0).astype(np.uint8)
    lab_samples = cv2.cvtColor(samples.reshape(-1,1,3), cv2.COLOR_RGB2LAB).reshape(-1,3).astype(np.float32)
    center = np.median(lab_samples, axis=0)

    img_lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    d = img_lab - center[None,None,:]
    dist2 = (d[...,0]/LAB_L_DIV)**2 + (d[...,1]/LAB_A_DIV)**2 + (d[...,2]/LAB_B_DIV)**2
    gate = (dist2 <= LAB_THRESH2).astype(np.uint8) * 255
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    gate = cv2.morphologyEx(gate, cv2.MORPH_OPEN, ker, iterations=1)
    gate = cv2.morphologyEx(gate, cv2.MORPH_CLOSE, ker, iterations=1)
    return gate

def decontaminate_edge(rgb_u8: np.ndarray, alpha_u8: np.ndarray, bg_gate_u8: np.ndarray,
                       inner_px: int, band_px: int) -> np.ndarray:
    ring = edge_band(alpha_u8, inner_px, band_px)
    ring_bg = cv2.bitwise_and(bg_gate_u8, ring)

    a = np.maximum(alpha_u8.astype(np.float32)/255.0, 1e-3)
    rgb = rgb_u8.astype(np.float32)
    unp = rgb / a[..., None]

    interior = (alpha_u8 >= 240)
    fg_mean = (rgb[interior].mean(axis=0) if np.any(interior) else rgb.mean(axis=0))

    k = np.zeros_like(alpha_u8, dtype=np.float32)
    k[ring_bg > 0] = 0.75  # stronger pull to FG color

    for c in range(3):
        unp[..., c] = (1.0 - k) * unp[..., c] + k * fg_mean[c]

    return (unp * a[..., None]).clip(0,255).astype(np.uint8)

def subtract_bg_in_ring(alpha_u8: np.ndarray, bg_gate_u8: np.ndarray,
                        inner_px: int, band_px: int) -> np.ndarray:
    ring = edge_band(alpha_u8, inner_px, band_px)
    ring_bg = cv2.bitwise_and(bg_gate_u8, ring)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    ring_bg = cv2.erode(ring_bg, ker, iterations=1)
    out = alpha_u8.copy()
    out[ring_bg > 0] = 0
    return out

def composite_on_color(rgb_u8: np.ndarray, alpha_u8: np.ndarray, color_rgb: Tuple[int,int,int]) -> np.ndarray:
    bg = np.zeros_like(rgb_u8, dtype=np.uint8); bg[:] = color_rgb
    a = (alpha_u8.astype(np.float32)/255.0)[...,None]
    return (rgb_u8.astype(np.float32)*a + bg.astype(np.float32)*(1.0-a)).clip(0,255).astype(np.uint8)

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/process")
async def process(video: UploadFile = File(...)):
    if not which("ffmpeg"):
        raise HTTPException(500, "ffmpeg not found")

    with tempfile.TemporaryDirectory() as tdir_str:
        tdir = Path(tdir_str)
        in_path    = tdir / video.filename
        frames_dir = tdir / "frames"
        out_dir    = tdir / "nobg"
        dbg_dir    = tdir / "greenscreen"
        frames_dir.mkdir(); out_dir.mkdir(); dbg_dir.mkdir()

        with open(in_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        safe_extract_rgba(in_path, frames_dir, DEFAULT_FPS)
        frame_paths = sorted(frames_dir.glob("frame_*.png"))
        if not frame_paths:
            raise RuntimeError("no frames extracted")

        alpha_buf: Deque[np.ndarray] = deque(maxlen=MEDIAN_WINDOW)
        ker3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        for idx, fpath in enumerate(frame_paths, start=1):
            img = Image.open(fpath).convert("RGBA")
            np_rgba = np.array(img)
            rgb = np_rgba[:, :, :3].astype(np.uint8)

            # ensemble masks
            try:    m_h = mask_from(img, SESSION_HUMAN)
            except: m_h = mask_from(img, SESSION_U2NET)
            try:    m_i = mask_from(img, SESSION_ISNET)
            except: m_i = mask_from(img, SESSION_U2NET)
            mask = np.maximum(m_h, m_i)

            # light reconnect + tiny erode (NO dilate)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker3, iterations=CLOSE_ITERS)
            if ERODE_ITERS:
                mask = cv2.erode(mask, ker3, iterations=ERODE_ITERS)

            # temporal
            alpha_buf.append(mask)
            median_alpha = np.median(np.stack(list(alpha_buf), axis=0), axis=0).astype(np.uint8)
            stable = hysteresis_stabilize(median_alpha)

            # **aggressive** inward clamp
            refined = inward_offset_mask(stable, EDGE_SHRINK_PX)

            # ring + bg model
            ring = edge_band(stable, EDGE_SHRINK_PX, EDGE_BAND_PX)
            bg_gate = estimate_bg_gate(rgb)

            # inpaint ring inward, then zero any bg-like pixels in ring
            rgb_fixed = edge_inpaint_inward(rgb, ring, INPAINT_RADIUS)
            refined   = subtract_bg_in_ring(refined, bg_gate, EDGE_SHRINK_PX, EDGE_BAND_PX)

            # decontaminate colors in ring
            rgb_decont = decontaminate_edge(rgb_fixed, refined, bg_gate, EDGE_SHRINK_PX, EDGE_BAND_PX)

            # clamp tiny alphas, edge-aware micro-feather
            a8 = refined.copy()
            a8[a8 < CLAMP_BELOW] = 0
            a8 = cv2.bilateralFilter(a8, BILAT_DIAM, BILAT_SIGMA, BILAT_SIGMA)

            out_rgba = np.dstack([rgb_decont, a8])
            Image.fromarray(out_rgba, "RGBA").save(out_dir / fpath.name)
            Image.fromarray(composite_on_color(rgb_decont, a8, CHROMA_GREEN), "RGB").save(dbg_dir / fpath.name)

        # outputs: ProRes4444 (alpha) + greenscreen mp4 for QA
        out_mov = tdir / f"{in_path.stem}_tight_nobg.mov"
        out_dbg = tdir / f"{in_path.stem}_greenscreen.mp4"

        ffmpeg([
            "-framerate", str(DEFAULT_FPS),
            "-i", str(out_dir / "frame_%05d.png"),
            "-c:v", "prores_ks", "-profile:v", "4444",
            "-pix_fmt", "yuva444p10le",
            "-color_primaries","bt709","-color_trc","bt709","-colorspace","bt709",
            str(out_mov),
        ])
        ffmpeg([
            "-framerate", str(DEFAULT_FPS),
            "-i", str(dbg_dir / "frame_%05d.png"),
            "-c:v","libx264","-pix_fmt","yuv420p","-crf","18",
            str(out_dbg),
        ])

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mov"); tmp.close()
        shutil.copyfile(out_mov, tmp.name)
        headers = {"Cache-Control":"no-store", "X-Green-Debug": out_dbg.name}
        return FileResponse(
            path=tmp.name, media_type="video/quicktime",
            filename=Path(tmp.name).name, headers=headers,
            background=BackgroundTask(lambda p: os.path.exists(p) and os.remove(p), tmp.name),
        )
