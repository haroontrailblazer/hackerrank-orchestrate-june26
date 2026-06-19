"""Image loading + downscaling.

Downscaling the long edge before sending to a VLM is the single biggest lever
on image-token cost (a full-res photo can cost ~4.8k tokens vs ~1.5k downscaled).
We also compute a content hash so identical images are inspected only once.
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class ImagePayload:
    image_id: str
    media_type: str
    b64: str
    sha: str
    ok: bool = True
    error: str = ""


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_image_payload(path: Path, *, max_edge: int = 1024, quality: int = 82) -> ImagePayload:
    """Load, downscale (long edge -> max_edge), re-encode JPEG, base64.

    Returns ok=False with an error message instead of raising, so a single
    unreadable image never aborts a whole claim.
    """
    image_id = path.stem
    try:
        sha = _hash_file(path)
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            longest = max(w, h)
            if longest > max_edge:
                scale = max_edge / float(longest)
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality)
            data = buf.getvalue()
        return ImagePayload(
            image_id=image_id,
            media_type="image/jpeg",
            b64=base64.standard_b64encode(data).decode("ascii"),
            sha=sha,
        )
    except FileNotFoundError:
        return ImagePayload(image_id, "image/jpeg", "", "", ok=False, error="file_not_found")
    except Exception as exc:  # corrupt/unsupported image -> mark unusable, keep going
        return ImagePayload(image_id, "image/jpeg", "", "", ok=False, error=f"{type(exc).__name__}")
