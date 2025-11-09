import os, shutil, subprocess, tempfile
from pathlib import Path
from typing import Deque, Tuple, List, Optional
from collections import deque

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from rembg import remove, new_session

# ===================== DEFAULTS (tight but not destructive) =====================
DEFAULT_FPS       = 20
MEDIAN_WINDOW     = 5

# inward clamp & ring
EDGE_SHRINK_PX    = 10
EDGE_BAND_PX      = 4
INPAINT_RADIUS    = 5

# mask thresholds
TH_STRONG         = 245
TH_WEAK           = 80
CLAMP_BELOW       = 30

# feather
BILAT_DIAM        = 5
BILAT_SIGMA       = 18

# rembg alpha matting
AM_FG_T           = 248
AM_BG_T           = 6
AM_ERODE          = 6

# morphology
CLOSE_ITERS       = 1
DILATE_ITERS      = 0
ERODE_ITERS       = 0

# Lab background gate
LAB_L_DIV         = 10.0
LAB_A_DIV         = 7.0
LAB_B_DIV         = 7.0
LAB_THRESH2       = 1.0

CHROMA_GREEN      = (16, 255, 16)

# sessions (load once)
SESSION_HUMAN = new_session("u2net_human_seg")
SESSION_U2NET = new_session("u2net")
SESSION_ISNET = new_session("isnet-general-use")

app = FastAPI(title="AI Video BG Remover – Tight Matte + Edge Snap + Flow Stabilize")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000","http://127.0.0.1:3000",
        "http://localhost:5173","http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------ utils ---------------------------------
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
    weak_alpha = (out < 200).astype(np.uint8) * 255
    kill = cv2.bitwise_and(ring_bg, weak_alpha)
    out[kill > 0] = 0
    return out

def composite_on_color(rgb_u8: np.ndarray, alpha_u8: np.ndarray, color_rgb: Tuple[int,int,int]) -> np.ndarray:
    bg = np.zeros_like(rgb_u8, dtype=np.uint8); bg[:] = color_rgb
    a = (alpha_u8.astype(np.float32)/255.0)[...,None]
    return (rgb_u8.astype(np.float32)*a + bg.astype(np.float32)*(1.0-a)).clip(0,255).astype(np.uint8)

def snap_to_image_edges(alpha_u8: np.ndarray, rgb_u8: np.ndarray, max_snap_px: int = 6) -> np.ndarray:
    """
    Inward-snap the alpha boundary using image gradients so we hug true edges.
    Works best after a small inward offset.
    """
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    med = np.median(gray)
    low = int(max(0, 0.66 * med))
    high = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, low, high)

    inv_edges = (edges == 0).astype(np.uint8)
    dist_to_edge = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 3)

    fg = (alpha_u8 > 0).astype(np.uint8)
    if fg.max() == 0:
        return alpha_u8
    dist_inside = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 3)

    snapped = alpha_u8.copy()
    collar = ((dist_inside > 0) & (dist_inside <= float(max_snap_px))).astype(np.uint8)
    keep = ((dist_to_edge <= dist_inside) & (collar == 1))
    to_zero = ((collar == 1) & (~keep))
    snapped[to_zero] = 0

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    snapped = cv2.morphologyEx(snapped, cv2.MORPH_CLOSE, ker, iterations=1)
    return snapped

# -------- Optical-flow temporal stabilization + motion hardening --------
def temporal_flow_refine(prev_mask: Optional[np.ndarray],
                         curr_mask: np.ndarray,
                         prev_frame: Optional[np.ndarray],
                         curr_frame: np.ndarray) -> np.ndarray:
    """Warp previous mask into current frame via optical flow and merge."""
    if prev_mask is None or prev_frame is None:
        return curr_mask

    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    h, w = curr_mask.shape
    # build remap grid
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[...,0]).astype(np.float32)
    map_y = (grid_y + flow[...,1]).astype(np.float32)

    warped_prev = cv2.remap(prev_mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    # combine: favor current, but fill holes with warped previous
    out = np.maximum(curr_mask, (warped_prev * 0.85).astype(np.uint8))
    return out

def motion_edge_harden(mask_u8: np.ndarray,
                       prev_rgb: Optional[np.ndarray],
                       curr_rgb: np.ndarray,
                       boost: int = 30) -> np.ndarray:
    """Increase alpha on areas with motion to avoid semi-transparent fingers."""
    if prev_rgb is None:
        return mask_u8
    diff = cv2.absdiff(curr_rgb, prev_rgb)
    motion_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    # suppress camera noise with blur and threshold
    motion_gray = cv2.GaussianBlur(motion_gray, (5,5), 0)
    _, motion_mask = cv2.threshold(motion_gray, 20, 255, cv2.THRESH_BINARY)
    # expand slightly to cover edges
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    motion_mask = cv2.dilate(motion_mask, ker, iterations=1)

    hardened = mask_u8.copy()
    idx = motion_mask > 0
    hardened[idx] = np.clip(hardened[idx].astype(np.int32) + boost, 0, 255).astype(np.uint8)
    return hardened

# ------------------------------ API ---------------------------------
@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/process")
async def process(
    video: UploadFile = File(...),
    tight: int = Query(10, ge=0,  le=40),   # maps to EDGE_SHRINK_PX
    snap:  int = Query(6,  ge=0,  le=20),   # max_snap_px for edge snap
    clamp: int = Query(30, ge=0,  le=80),   # maps to CLAMP_BELOW
    boost: int = Query(30, ge=0,  le=80)    # motion hardening strength
):
    if not which("ffmpeg"):
        raise HTTPException(500, "ffmpeg not found")

    # apply tunables
    global EDGE_SHRINK_PX, CLAMP_BELOW
    EDGE_SHRINK_PX = int(tight)
    CLAMP_BELOW    = int(clamp)

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

        alpha_buf: Deque[np.ndarray]   = deque(maxlen=MEDIAN_WINDOW)
        refined_buf: Deque[np.ndarray] = deque(maxlen=MEDIAN_WINDOW)
        ker3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        prev_refined: Optional[np.ndarray] = None
        prev_rgb: Optional[np.ndarray]     = None

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

            # light reconnect + avoid dilate
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker3, iterations=CLOSE_ITERS)
            if ERODE_ITERS:
                mask = cv2.erode(mask, ker3, iterations=ERODE_ITERS)

            # temporal stabilize the raw mask
            alpha_buf.append(mask)
            median_alpha = np.median(np.stack(list(alpha_buf), axis=0), axis=0).astype(np.uint8)
            stable = hysteresis_stabilize(median_alpha)

            # inward clamp then edge-aware snap
            refined = inward_offset_mask(stable, EDGE_SHRINK_PX)
            refined = snap_to_image_edges(refined, rgb, max_snap_px=int(snap))

            # optical-flow temporal refine (for motion like waving)
            refined = temporal_flow_refine(prev_refined, refined, prev_rgb, rgb)

            # motion-edge hardening (keeps moving fingers opaque)
            refined = motion_edge_harden(refined, prev_rgb, rgb, boost=int(boost))

            # temporal stabilize the *refined* mask to kill shimmer
            refined_buf.append(refined)
            refined = np.median(np.stack(list(refined_buf), axis=0), axis=0).astype(np.uint8)

            # ring & background model
            ring = edge_band(stable, EDGE_SHRINK_PX, EDGE_BAND_PX)
            bg_gate = estimate_bg_gate(rgb)

            # inpaint ring inward, subtract bg-like ring, decontaminate colors
            rgb_fixed  = edge_inpaint_inward(rgb, ring, INPAINT_RADIUS)
            refined    = subtract_bg_in_ring(refined, bg_gate, EDGE_SHRINK_PX, EDGE_BAND_PX)
            rgb_decont = decontaminate_edge(rgb_fixed, refined, bg_gate, EDGE_SHRINK_PX, EDGE_BAND_PX)

            # clamp tiny alphas, micro-feather
            a8 = refined.copy()
            a8[a8 < CLAMP_BELOW] = 0
            a8 = cv2.bilateralFilter(a8, BILAT_DIAM, BILAT_SIGMA, BILAT_SIGMA)

            out_rgba = np.dstack([rgb_decont, a8])
            Image.fromarray(out_rgba, "RGBA").save(out_dir / fpath.name)
            Image.fromarray(composite_on_color(rgb_decont, a8, CHROMA_GREEN), "RGB").save(dbg_dir / fpath.name)

            # carry over for next frame
            prev_refined = refined.copy()
            prev_rgb = rgb.copy()

        # outputs: ProRes4444 w/ alpha (.mov) + greenscreen mp4 for QA
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
