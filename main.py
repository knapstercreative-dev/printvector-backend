"""
PrintVector.io — Vectorization Backend
FastAPI + vtracer + Pillow

Deploy to Railway:
  1. Create new project → Deploy from GitHub repo
  2. Add this file + requirements.txt
  3. Set start command: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import io
import base64
import traceback
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import vtracer

app = FastAPI(title="PrintVector API", version="1.0.0")

# ── CORS — allow your frontend domain ────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Lock this down to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# MODE CONFIGS
# Each mode tunes vtracer differently for the target use case
# ─────────────────────────────────────────────────────────────────────────────

MODE_CONFIGS = {

    # Logo — clean flat shapes, minimal paths, print-ready
    "logo": {
        "colormode":         "color",
        "hierarchical":      "stacked",
        "mode":              "polygon",
        "filter_speckle":    8,
        "color_precision":   6,
        "layer_difference":  16,
        "corner_threshold":  60,
        "length_threshold":  4.0,
        "max_iterations":    10,
        "splice_threshold":  45,
        "path_precision":    3,
        "preprocess":        "logo",   # custom preprocessing flag
    },

    # Line Art — fine lines, high detail, minimal color reduction
    "lineart": {
        "colormode":         "binary",
        "hierarchical":      "cutout",
        "mode":              "spline",
        "filter_speckle":    4,
        "color_precision":   8,
        "layer_difference":  8,
        "corner_threshold":  90,
        "length_threshold":  2.0,
        "max_iterations":    10,
        "splice_threshold":  45,
        "path_precision":    4,
        "preprocess":        "lineart",
    },

    # Photo — color fidelity, smooth gradients, detailed paths
    "photo": {
        "colormode":         "color",
        "hierarchical":      "stacked",
        "mode":              "spline",
        "filter_speckle":    2,
        "color_precision":   8,
        "layer_difference":  6,
        "corner_threshold":  60,
        "length_threshold":  2.0,
        "max_iterations":    10,
        "splice_threshold":  45,
        "path_precision":    5,
        "preprocess":        "photo",
    },

    # Halftone — prepare for screen print / spot color output
    "halftone": {
        "colormode":         "color",
        "hierarchical":      "stacked",
        "mode":              "polygon",
        "filter_speckle":    6,
        "color_precision":   4,      # reduce to spot colors
        "layer_difference":  20,
        "corner_threshold":  60,
        "length_threshold":  4.0,
        "max_iterations":    10,
        "splice_threshold":  45,
        "path_precision":    3,
        "preprocess":        "halftone",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(img: Image.Image, mode: str, simplify: int, color_target: int) -> Image.Image:
    """
    Mode-specific image preprocessing before vectorization.
    simplify: 0-100, higher = more simplified output
    color_target: target number of colors (2-16)
    """

    # Convert to RGBA for consistent handling
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")

    # Flatten transparency to white background
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert("RGB")

    # Resize — vtracer works best at 1000-2000px on longest side
    max_dim = 1800
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    if mode == "logo":
        # Boost contrast, reduce noise, posterize to target colors
        img = ImageEnhance.Contrast(img).enhance(1.4)
        img = ImageEnhance.Sharpness(img).enhance(1.2)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        # Posterize based on color_target (maps 2-16 colors to 1-4 bits)
        bits = max(1, min(4, round(color_target / 4)))
        img = ImageOps.posterize(img, bits)

    elif mode == "lineart":
        # Convert to grayscale, high contrast, threshold
        img = img.convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.0)
        # Adaptive threshold via point
        threshold = 128 - (simplify - 50)
        img = img.point(lambda p: 255 if p > threshold else 0)
        img = img.convert("RGB")

    elif mode == "photo":
        # Light smoothing, preserve color
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        img = ImageEnhance.Color(img).enhance(1.1)

    elif mode == "halftone":
        # Reduce to spot colors — quantize to color_target palette
        img = img.quantize(colors=min(color_target, 8), method=Image.Quantize.MEDIANCUT)
        img = img.convert("RGB")

    return img


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# Rate the vectorization quality 0-100
# ─────────────────────────────────────────────────────────────────────────────

def score_svg(svg: str) -> int:
    """Heuristic quality score based on SVG output characteristics."""
    path_count = svg.count("<path")
    svg_size   = len(svg)

    # Penalize for too many paths (over-complex) or too few (under-traced)
    if path_count == 0:
        return 20
    elif path_count < 5:
        score = 70
    elif path_count < 50:
        score = 92
    elif path_count < 200:
        score = 85
    elif path_count < 500:
        score = 75
    else:
        score = 60  # too complex for most print uses

    # Bonus for reasonable file size
    if 2000 < svg_size < 80000:
        score = min(100, score + 5)

    return score


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "PrintVector API", "status": "ok", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/vectorize")
async def vectorize(
    file:         UploadFile = File(...),
    mode:         str        = Form("logo"),
    simplify:     int        = Form(65),
    color_target: int        = Form(4),
):
    """
    Main vectorization endpoint.

    Parameters:
        file         — uploaded image (PNG, JPG, WEBP, BMP)
        mode         — logo | lineart | photo | halftone
        simplify     — 0-100, path simplification level
        color_target — target color count (2-16)

    Returns JSON:
        svg          — SVG string
        score        — quality score 0-100
        path_count   — number of paths in output
        file_size_kb — SVG file size in KB
        mode         — mode used
    """

    # Validate mode
    if mode not in MODE_CONFIGS:
        raise HTTPException(400, f"Unknown mode '{mode}'. Use: {list(MODE_CONFIGS.keys())}")

    # Validate simplify / color_target
    simplify     = max(0, min(100, simplify))
    color_target = max(2, min(16, color_target))

    # Read and validate image
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(413, "File too large. Max 20MB.")

    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Could not open image. Upload PNG, JPG, WEBP, or BMP.")

    # Preprocess
    try:
        img = preprocess(img, mode, simplify, color_target)
    except Exception as e:
        raise HTTPException(500, f"Preprocessing failed: {str(e)}")

    # Convert preprocessed image to bytes for vtracer
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # Get vtracer config for this mode
    cfg = MODE_CONFIGS[mode].copy()
    cfg.pop("preprocess", None)  # internal key, not a vtracer param

    # Apply simplify to filter_speckle and length_threshold
    cfg["filter_speckle"]    = max(1, int(cfg["filter_speckle"] * (simplify / 65)))
    cfg["length_threshold"]  = round(cfg["length_threshold"] * (simplify / 65), 1)

    # Run vtracer
    try:
        svg = vtracer.convert_raw_image_to_svg(
            img_bytes,
            img_format = "PNG",
            **cfg
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Vectorization failed: {str(e)}")

    # Score and stats
    path_count   = svg.count("<path")
    file_size_kb = round(len(svg.encode()) / 1024, 1)
    quality      = score_svg(svg)

    return {
        "svg":          svg,
        "score":        quality,
        "path_count":   path_count,
        "file_size_kb": file_size_kb,
        "mode":         mode,
    }


@app.post("/vectorize/base64")
async def vectorize_b64(payload: dict):
    """
    Alternative endpoint accepting base64 image input.
    Useful for browser clients that can't send multipart.

    Body: { "image": "data:image/png;base64,...", "mode": "logo", "simplify": 65, "color_target": 4 }
    """
    try:
        b64 = payload.get("image", "")
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image data.")

    mode         = payload.get("mode", "logo")
    simplify     = int(payload.get("simplify", 65))
    color_target = int(payload.get("color_target", 4))

    if mode not in MODE_CONFIGS:
        raise HTTPException(400, f"Unknown mode '{mode}'.")

    simplify     = max(0, min(100, simplify))
    color_target = max(2, min(16, color_target))

    try:
        img = Image.open(io.BytesIO(img_bytes))
    except Exception:
        raise HTTPException(400, "Could not decode image.")

    img = preprocess(img, mode, simplify, color_target)

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    cfg = MODE_CONFIGS[mode].copy()
    cfg.pop("preprocess", None)
    cfg["filter_speckle"]   = max(1, int(cfg["filter_speckle"] * (simplify / 65)))
    cfg["length_threshold"] = round(cfg["length_threshold"] * (simplify / 65), 1)

    try:
        svg = vtracer.convert_raw_image_to_svg(
            buf.getvalue(),
            img_format="PNG",
            **cfg
        )
    except Exception as e:
        raise HTTPException(500, f"Vectorization failed: {str(e)}")

    return {
        "svg":          svg,
        "score":        score_svg(svg),
        "path_count":   svg.count("<path"),
        "file_size_kb": round(len(svg.encode()) / 1024, 1),
        "mode":         mode,
    }
