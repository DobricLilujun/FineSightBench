"""Natural-scene background loader for SynthText-style tasks.

Downloads a subset of the SynthText *bg_img* image set
(https://thor.robots.ox.ac.uk/scenetext/preproc/bg_img.tar.gz) by streaming
the tarball and extracting the first ``max_images`` JPEGs only. This keeps
disk usage low (~1 GB for 1500 images) while re-using the exact photographs
from the SynthText paper.

Users can supply their own directory via ``bg_dir``.
"""

from __future__ import annotations

import random
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image

_SYNTHTEXT_BG_URL = "https://thor.robots.ox.ac.uk/scenetext/preproc/bg_img.tar.gz"
_DEFAULT_BG_ROOT = Path("data/synthtext_bgs")
_DEFAULT_MAX_IMAGES = 1500


def _stream_extract_synthtext(dest: Path, max_images: int) -> None:
    """Stream ``bg_img.tar.gz`` from Oxford and extract up to *max_images* JPEGs.

    The tarball is NOT saved to disk; we pipe the HTTP response directly into
    ``tarfile`` in streaming mode and abort as soon as enough images are out.
    """
    img_dir = dest / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[textwild] streaming SynthText bg_img.tar.gz from Oxford "
        f"(will extract up to {max_images} images) …"
    )
    count = 0
    with urllib.request.urlopen(_SYNTHTEXT_BG_URL) as resp:
        # ``r|gz`` is streaming mode: members are read sequentially.
        with tarfile.open(fileobj=resp, mode="r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                name_lower = m.name.lower()
                if not (name_lower.endswith(".jpg") or name_lower.endswith(".jpeg")):
                    continue
                out_path = img_dir / Path(m.name).name
                if not out_path.exists():
                    with tf.extractfile(m) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())
                count += 1
                if count % 100 == 0:
                    print(f"[textwild]   extracted {count} images …")
                if count >= max_images:
                    break
    print(f"[textwild] done, {count} SynthText background images at {img_dir}")


def ensure_backgrounds(
    bg_dir: str | Path | None = None,
    max_images: int = _DEFAULT_MAX_IMAGES,
) -> Path:
    """Return a directory containing background images.

    * If *bg_dir* is given, it must exist and contain images.
    * Otherwise SynthText's ``bg_img`` is streamed into
      ``data/synthtext_bgs/images/`` (up to *max_images* files).
    """
    if bg_dir is not None:
        p = Path(bg_dir)
        if not p.exists():
            raise FileNotFoundError(f"bg_dir not found: {p}")
        return p

    root = _DEFAULT_BG_ROOT
    img_dir = root / "images"
    need_download = (not img_dir.exists()) or (not any(img_dir.iterdir()))
    if need_download:
        _stream_extract_synthtext(root, max_images=max_images)
    return img_dir


def list_backgrounds(
    bg_dir: str | Path | None = None,
    max_images: int = _DEFAULT_MAX_IMAGES,
) -> list[Path]:
    """Return a list of image paths available under *bg_dir*."""
    root = ensure_backgrounds(bg_dir, max_images=max_images)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
    if not imgs:
        raise RuntimeError(f"no background images found under {root}")
    return imgs


def sample_background(
    paths: list[Path], canvas_size: int, rng: random.Random
) -> Image.Image:
    """Pick a random background, center-crop to square, resize to *canvas_size*.

    Skips unreadable / corrupt JPEGs and retries up to 10 times.
    """
    last_err: Exception | None = None
    for _ in range(10):
        p = rng.choice(paths)
        try:
            img = Image.open(p)
            img.load()
            img = img.convert("RGB")
        except Exception as e:  # unreadable JPEG, skip and retry
            last_err = e
            continue
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        if img.size != (canvas_size, canvas_size):
            img = img.resize((canvas_size, canvas_size), Image.LANCZOS)
        return img
    raise RuntimeError(f"failed to load any background after 10 tries: {last_err}")
