"""NVIDIA backend JSON extraction.

NVIDIA NIM / API-Catalog VLMs are OpenAI-compatible but do not reliably support
the strict json_schema `.parse()` path, so the backend asks for a JSON object and
parses it itself. This logic (tolerating prose/code-fences, partial output, and
garbage) is the part worth testing without a network call.
"""

from argus.llm.nvidia_backend import extract_json
from argus.schemas import ConversationAnalysis, ImageFinding


def test_extract_plain_json():
    out = extract_json(ConversationAnalysis, '{"asserted_issue_type":"dent","asserted_object_part":"door"}')
    assert out.asserted_issue_type == "dent"
    assert out.asserted_object_part == "door"


def test_extract_json_inside_code_fence_and_prose():
    content = 'Sure!\n```json\n{"detected_object":"car","damage_visible":true}\n```\nThat is my answer.'
    out = extract_json(ImageFinding, content)
    assert out.detected_object == "car"
    assert out.damage_visible is True


def test_extract_partial_json_fills_defaults():
    out = extract_json(ImageFinding, '{"detected_object":"laptop"}')
    assert out.detected_object == "laptop"
    assert out.usable is True  # schema default


def test_extract_garbage_returns_safe_defaults():
    out = extract_json(ConversationAnalysis, "I'm sorry, I can't do that.")
    assert out.asserted_issue_type == "unknown"  # safe default, never raises


def test_extract_empty_returns_safe_defaults():
    out = extract_json(ImageFinding, "")
    assert out.detected_object == "unknown"


def test_nvidia_settings_defaults():
    from argus.config import Settings

    s = Settings(provider="nvidia")
    assert s.vision_model == "meta/llama-3.2-90b-vision-instruct"
    assert s.reasoning_model == "meta/llama-3.3-70b-instruct"
    assert s.nvidia_base_url.endswith("/v1")


def test_nvidia_backend_constructs_without_key():
    # The class itself must build even with no key (so it can be in a fallback
    # chain); a real call would 401. Factory-level fallback is tested separately.
    from argus.llm.base import Usage
    from argus.llm.nvidia_backend import NvidiaBackend

    b = NvidiaBackend(
        vision_model="m", reasoning_model="r", base_url="https://x/v1",
        max_retries=1, timeout=10, usage=Usage(),
    )
    assert b.name == "nvidia"
