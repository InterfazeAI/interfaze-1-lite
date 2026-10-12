"""Runtime settings, all environment-driven so the container needs no config file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _flag(name: str) -> bool:
    """On unless the variable is "0"."""
    return os.environ.get(name, "1") != "0"


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    # vLLM's OpenAI-compatible server. Bound to loopback in the container because its
    # /sleep and /collective_rpc dev routes are unauthenticated.
    brain_url: str = field(default_factory=lambda: _env("BRAIN_URL", "http://127.0.0.1:8001"))
    # The name vLLM answers to: under `--served-model-name X` the repo id is a 404. Kept apart
    # from the checkpoint so the served name stays stable when the checkpoint changes.
    brain_served_name: str = field(default_factory=lambda: _env("BRAIN_SERVED_NAME", "interfaze-lite"))
    perception_url: str = field(default_factory=lambda: _env("PERCEPTION_URL", "http://127.0.0.1:8002"))
    diarize_url: str = field(default_factory=lambda: _env("DIARIZE_URL", "http://127.0.0.1:8003"))

    model_name: str = field(default_factory=lambda: _env("MODEL_NAME", "interfaze-lite"))

    # A tool-calling turn that never converges would otherwise loop until the request
    # times out; interfaze caps steps the same way.
    max_tool_steps: int = field(default_factory=lambda: _int("MAX_TOOL_STEPS", 8))
    max_new_tokens: int = field(default_factory=lambda: _int("MAX_NEW_TOKENS", 4096))
    # Guided decoding cut off at the cap emits unparseable JSON, not a short answer. 16384 cut a
    # ten-page "every element with its box" request at 20.7k tokens (interfaze wrote 34k); trimming
    # tool output to max_context_chars keeps the prompt near 70k of the brain's 131k context.
    max_structured_tokens: int = field(default_factory=lambda: _int("MAX_STRUCTURED_TOKENS", 32000))
    # Outlines require the segmentation pass that boxes alone do not, so this trades
    # latency for shape. Off restores box-only detection.
    detection_outlines: bool = field(default_factory=lambda: _flag("DETECTION_OUTLINES"))
    # The largest completion a caller may ask for, as interfaze publishes it: validated against an
    # internal default instead, the 32000 the SDKs send by default was rejected.
    max_output_tokens: int = field(default_factory=lambda: _int("MAX_OUTPUT_TOKENS", 32000))
    # Reuse the tool-selection turn's answer rather than generate a second one. A switch so a
    # quality regression can be backed out without new code.
    reuse_tool_turn_answer: bool = field(default_factory=lambda: _flag("REUSE_TOOL_TURN_ANSWER"))
    request_timeout_s: int = field(default_factory=lambda: _int("REQUEST_TIMEOUT_S", 900))
    tool_timeout_s: int = field(default_factory=lambda: _int("TOOL_TIMEOUT_S", 600))

    # Geometry the model sees. A dense 50-page scan yields thousands of lines; past a
    # few hundred the boxes crowd out the text they describe.
    max_layout_blocks: int = field(default_factory=lambda: _int("MAX_LAYOUT_BLOCKS", 400))

    # Bounds prefill on a large screenshot; not a quality fix (runaway box counts came from
    # upscaled fakes already near 1.5 MP). ScreenSpot-v2 tops out near 2 MP, so real ones pass.
    ground_max_pixels: int = field(default_factory=lambda: _int("GROUND_MAX_PIXELS", 4_194_304))
    # Photographs: at a native 1.48 MP the sword was boxed on the armour and "every object" looped
    # to the cap; at 1 MP both came back clean, and large photos looped far less.
    object_ground_max_pixels: int = field(default_factory=lambda: _int("OBJECT_GROUND_MAX_PIXELS", 1_048_576))

    # The caller's images, shown beside a tool's reading for what text cannot carry (a highlight,
    # a colour). 0 withholds them.
    answer_image_max_pixels: int = field(default_factory=lambda: _int("ANSWER_IMAGE_MAX_PIXELS", 1_048_576))

    max_context_chars: int = field(default_factory=lambda: _int("MAX_CONTEXT_CHARS", 120_000))

    # Supplied by the deployment rather than written here, so this file states no
    # dependency on any particular provider.
    component_names: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            n for n in os.environ.get("COMPONENT_NAMES", "").split(",") if n.strip()
        )
    )

    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))

    # Unset, with no ADMIN_KEY either, means open: a key on your own localhost buys nothing. Set it
    # on any deployment reachable from outside.
    api_key: str | None = field(default_factory=lambda: os.environ.get("API_KEY") or None)
    # The hosted deployment's key, sent by the interfaze API in `x-api-admin-key` as it is to
    # every model service. Some secret stores expose it in lowercase.
    admin_key: str | None = field(
        default_factory=lambda: os.environ.get("ADMIN_KEY") or os.environ.get("admin_key") or None
    )

    @property
    def brain_chat_url(self) -> str:
        return f"{self.brain_url.rstrip('/')}/v1/chat/completions"


settings = Settings()
