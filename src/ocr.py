"""
ocr.py
------
Photo -> text conversion, built in.

Two backends:

1. `tesseract` (default, fully offline, free)
   Uses OpenCV for basic preprocessing (grayscale, denoise, adaptive
   threshold, deskew) then pytesseract for text extraction. This is the
   backend used out of the box. It works reasonably well for clearly
   printed/typed question papers and neat handwriting, but — like any
   OCR engine — accuracy drops on messy handwriting.

2. `claude_vision` (optional, needs an ANTHROPIC_API_KEY, best accuracy)
   Sends the image to the Claude API (vision-capable model) and asks it to
   transcribe the handwritten/printed text only, ignoring diagrams/figures.
   This is far more robust to messy handwriting than tesseract, at the cost
   of needing an API key and internet access. Recommended if OCR accuracy
   on real student handwriting matters for your use case.

Both backends return plain text. Neither backend interprets or grades
anything — that is grader.py's job.

--- Robustness fixes for real phone-camera photo uploads -------------------
Photos of paper taken on a phone are messier than clean scans, and a few
issues used to make PNG/JPG uploads fail or silently OCR badly:
  * EXIF orientation was ignored, so portrait photos stored "sideways" by
    the camera were read sideways/upside-down, tanking OCR accuracy.
    -> now corrected via Pillow's exif_transpose() before anything else.
  * Transparent PNGs (RGBA) or palette PNGs with a transparency channel
    were converted in a way that could turn transparent areas black,
    confusing thresholding. -> now flattened onto a white background.
  * Very large photos (12MP+ phone cameras) made denoising slow and could
    time out the app for no accuracy benefit. -> now downscaled to a
    reasonable max dimension first.
  * A corrupt/unreadable file, or a missing tesseract binary, used to
    raise an uncaught exception straight out of this module and could
    crash the whole grading run. -> now raises a clear, catchable
    RuntimeError/ValueError with a helpful message instead.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError

# On Windows, the tesseract.exe binary usually isn't on PATH even after
# installing it, so pytesseract can't find it by default. Auto-detect the
# standard install location so users don't have to edit this file by hand.
if os.name == "nt":
    _default_win_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for _p in _default_win_paths:
        if os.path.exists(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break

# Cap the longest side of any uploaded photo before processing. Phone
# cameras often produce 3000-4000px images; that resolution buys nothing
# for OCR accuracy but makes denoising/deskewing noticeably slow.
MAX_IMAGE_SIDE = 2200


# ---------------------------------------------------------------------------
# Preprocessing (shared by the tesseract path; also used to sanity-check
# images before sending them to the vision API)
# ---------------------------------------------------------------------------
def preprocess_image(image_path: str) -> np.ndarray:
    """
    Light-touch preprocessing tuned for phone photos of paper (uneven lighting,
    slight skew, sideways EXIF orientation, transparency). Deliberately
    conservative: aggressive thresholding/deskewing can do more harm than
    good on already-legible images, so each step only kicks in when there's
    clear evidence it's needed.
    """
    try:
        pil_img = Image.open(image_path)
        pil_img.load()  # force-read now so truncated files fail here, clearly
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(
            f"Could not read image at {image_path}: the file may be corrupted, "
            f"in an unsupported format, or not actually an image. ({e})"
        ) from e

    # Respect the camera's EXIF orientation tag. Phone photos are very often
    # stored "sideways" with an orientation flag that a naive image read
    # ignores, which used to make OCR fail on perfectly good photos.
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass  # missing/corrupt EXIF metadata shouldn't block OCR

    # Flatten any transparency onto a white background (paper is white, so
    # this matches what a scanned/printed page actually looks like) instead
    # of letting it default to black, which confuses thresholding.
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        pil_img = pil_img.convert("RGBA")
        background = Image.new("RGB", pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[-1])
        pil_img = background
    else:
        pil_img = pil_img.convert("RGB")

    if max(pil_img.size) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(pil_img.size)
        new_size = (max(1, int(pil_img.width * scale)), max(1, int(pil_img.height * scale)))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if img is None or img.size == 0:
        raise ValueError(f"Could not read image at {image_path}: image data is empty.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Otsu's threshold is far gentler than a small-block adaptive threshold
    # for evenly-lit images, and still copes with moderate shadow gradients.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Only attempt deskew if there's a meaningful amount of dark (text) pixels
    # to estimate an angle from, and only apply it if the angle is non-trivial.
    dark_pixel_ratio = float((thresh < 128).sum()) / thresh.size
    if dark_pixel_ratio > 0.01:
        coords = np.column_stack(np.where(thresh < 128))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if 0.5 < abs(angle) < 15:  # ignore noise-driven angles from near-blank images
            (h, w) = thresh.shape
            center = (w // 2, h // 2)
            m = cv2.getRotationMatrix2D(center, angle, 1.0)
            thresh = cv2.warpAffine(
                thresh, m, (w, h), flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT, borderValue=255,
            )

    return thresh


def ocr_tesseract(image_path: str) -> str:
    """Offline OCR using tesseract. Good for typed/printed text and neat writing."""
    processed = preprocess_image(image_path)
    pil_img = Image.fromarray(processed)
    try:
        text = pytesseract.image_to_string(pil_img, config="--psm 6")
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(
            "The 'tesseract' OCR engine isn't installed or isn't on PATH. "
            "Install it (e.g. `apt-get install tesseract-ocr` on Linux, "
            "`brew install tesseract` on Mac, or the Windows installer), "
            "or switch to the 'claude_vision' OCR backend in the sidebar instead."
        ) from e
    return text.strip()


# ---------------------------------------------------------------------------
# Optional: Claude Vision backend (much better on messy handwriting)
# ---------------------------------------------------------------------------
def ocr_claude_vision(image_path: str, api_key: str | None = None) -> str:
    """
    High-accuracy handwriting transcription via the Claude API.
    Requires the `anthropic` package and an API key
    (pass explicitly or set the ANTHROPIC_API_KEY environment variable).
    """
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "The 'anthropic' package is not installed. Run: pip install anthropic"
        ) from e

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set ANTHROPIC_API_KEY or pass api_key= explicitly."
        )

    # Normalise the image first (handles EXIF rotation, transparency, huge
    # phone photos, and unsupported/odd encodings) and send a clean JPEG to
    # the API instead of the raw upload — this also sidesteps occasional
    # "unsupported image format" errors on unusual PNG/WEBP encodings.
    try:
        pil_img = Image.open(image_path)
        pil_img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(
            f"Could not read image at {image_path}: the file may be corrupted or "
            f"not actually an image. ({e})"
        ) from e

    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass

    if pil_img.mode in ("RGBA", "LA", "P"):
        pil_img = pil_img.convert("RGB")
    if max(pil_img.size) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(pil_img.size)
        pil_img = pil_img.resize(
            (max(1, int(pil_img.width * scale)), max(1, int(pil_img.height * scale))),
            Image.LANCZOS,
        )

    import io as _io
    buf = _io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    img_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    media_type = "image/jpeg"

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": img_b64},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Transcribe only the handwritten/printed TEXT from this exam "
                            "answer sheet image, question by question, preserving question "
                            "numbers (e.g. 'Q1', '2)', etc.) exactly as written. "
                            "Do NOT describe, interpret, or grade any diagrams, graphs, or "
                            "figures — just skip them entirely and write '[diagram omitted]' "
                            "in their place. Output plain text only, no commentary."
                        ),
                    },
                ],
            }
        ],
    )
    return "".join(block.text for block in response.content if hasattr(block, "text")).strip()


def extract_text(image_path: str, backend: str = "tesseract", api_key: str | None = None) -> str:
    """Single entry point used by the rest of the app."""
    if backend == "claude_vision":
        return ocr_claude_vision(image_path, api_key=api_key)
    return ocr_tesseract(image_path)