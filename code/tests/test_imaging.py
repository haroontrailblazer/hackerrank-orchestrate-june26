"""Image loading: downscale, encode, hash, and graceful failure."""

from PIL import Image

from argus.imaging import load_image_payload


def _make_png(path, size):
    Image.new("RGB", size, (123, 80, 200)).save(path, format="PNG")


def test_downscales_long_edge_and_encodes_jpeg(tmp_path):
    p = tmp_path / "big.png"
    _make_png(p, (3000, 1500))
    payload = load_image_payload(p, max_edge=1024, quality=80)
    assert payload.ok
    assert payload.media_type == "image/jpeg"
    assert payload.b64 and payload.sha
    # decode back and confirm the long edge was capped
    import base64, io

    img = Image.open(io.BytesIO(base64.b64decode(payload.b64)))
    assert max(img.size) <= 1024


def test_small_image_not_upscaled(tmp_path):
    p = tmp_path / "small.png"
    _make_png(p, (200, 100))
    import base64, io

    payload = load_image_payload(p, max_edge=1024)
    img = Image.open(io.BytesIO(base64.b64decode(payload.b64)))
    assert max(img.size) == 200


def test_hash_is_deterministic(tmp_path):
    p = tmp_path / "x.png"
    _make_png(p, (300, 300))
    assert load_image_payload(p).sha == load_image_payload(p).sha


def test_missing_file_returns_not_ok(tmp_path):
    payload = load_image_payload(tmp_path / "nope.jpg")
    assert payload.ok is False and payload.error == "file_not_found"
