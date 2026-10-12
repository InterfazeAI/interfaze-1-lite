"""Config for the Interfaze Lite composite model.

Interfaze Lite is not a single set of weights. It is a router plus the perception
models that together answer one request. This config names the component checkpoints
and the policy for loading them.

Two modes:

- composite (default, `bundle_weights=False`) -- the repo carries only config and
  modeling code; each component is pulled from its own upstream checkpoint on first
  use. Small repo, no weight redistribution, upstream licences stay where they are.
- bundled (`bundle_weights=True`) -- component weights live in this repo, one folder
  per component. Self-contained and offline-capable, at roughly 44 GB and with the
  obligation to carry every upstream licence.
"""

from __future__ import annotations

import os

from transformers import PretrainedConfig

# Capability -> component, from COMPONENT_<CAPABILITY> at import time: this file declares the
# capabilities without naming what provides them.
_CAPABILITIES = (
    "brain",
    "segmenter",
    "ocr_vlm",
    "asr",
    "asr_fallback",
    "diarizer",
    "layout",
    "line_detector",
    "line_recognizer",
    "guard",
    "forecaster",
)

DEFAULT_COMPONENTS: dict[str, str] = {
    name: os.environ[f"COMPONENT_{name.upper()}"]
    for name in _CAPABILITIES
    if os.environ.get(f"COMPONENT_{name.upper()}")
}


class InterfazeLiteConfig(PretrainedConfig):
    model_type = "interfaze_lite"

    def __init__(
        self,
        components: dict[str, str] | None = None,
        bundle_weights: bool = False,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_context: int = 262_144,
        lazy_load: bool = True,
        detection_score_threshold: float = 0.5,
        detection_mask_threshold: float = 0.5,
        # Fraction of a detector line that must sit inside a layout block to belong to
        # it. Containment, not IoU -- see contracts.contained_fraction.
        ocr_stitch_min_containment: float = 0.5,
        # ~4.2 MP. At 1.05 MP a dense arXiv page left subscripts a few pixels tall: 664 of 715
        # failing olmOCR-Bench math tests had the content absent. Prefill and line detection scale with it.
        ocr_max_pixels: int = 4_194_304,
        # The budget a page is re-read at when its lines are small or its confidence low
        # (ocr_small_line_ratio, ocr_retry_confidence); one flat budget starves two-column papers.
        ocr_dense_max_pixels: int = 8_388_608,
        # An image smaller than this is read at twice its size: a 720x960 receipt photo misread
        # three of ten checked fields at 1x and none at 2x, faster.
        ocr_upscale_below_pixels: int = 2_097_152,
        # Above roughly this the line detector aborts in C++ ("RuntimeError: std::exception"); it
        # reads a smaller copy and its boxes are scaled up to the VLM's frame.
        line_detector_max_pixels: int = 1_500_000,
        # A line shorter than this (as a fraction of page height) counts as small text.
        ocr_small_line_ratio: float = 0.014,
        # Mean line detection confidence below which a page is re-read at the dense budget.
        ocr_retry_confidence: float = 0.75,
        # Line detection confidence below which a line the reader did not also read is
        # dropped as noise. On 50 arXiv pages, lines under 0.6 were 9 real and 58 not.
        ocr_unread_min_confidence: float = 0.6,
        # The same, on a page where the reader read no text at all. Real lines sit at 0.9
        # and up (2,439 of them on the arXiv pages, against 62 at 0.8-0.9).
        ocr_unread_alone_min_confidence: float = 0.9,
        # Raise with ocr_max_pixels, or dense pages truncate: the budget asks 16k tokens of a 4 MP
        # page and is clamped to this.
        ocr_max_new_tokens: int = 8192,
        # Matches what interfaze rasterises PDFs at. Lower loses small type;
        # higher just costs prefill the recogniser cannot use.
        pdf_render_dpi: int = 200,
        # The frame PDF boxes are reported in: interfaze renders pages at scale 2
        # (144 DPI), so an A4 page is 1190x1684 and clients draw on that.
        pdf_frame_dpi: int = 144,
        detect_max_new_tokens: int = 2048,
        # Read by nothing: left from the alternate detector, which no longer loads.
        lam_generation_mode: str = "mtp",
        # A co-located vLLM server for the document VLM (else OCR_VLM_URL); unset, it loads
        # in-process through transformers, roughly an order of magnitude slower.
        ocr_vlm_url: str | None = None,
        ocr_vlm_served_name: str = "document-reader",
        # Pages OCR'd at once. Matches the OCR server's --max-num-seqs; higher only
        # deepens its queue, and the line detector is competing for CPU at the same time.
        ocr_page_concurrency: int = 16,
        line_detector_pool_size: int = 4,
        max_pdf_pages: int = 50,
        # What the brain sees when grounding, as the service's settings of the same
        # names: screenshots up to ScreenSpot's ~2 MP untouched, photographs at 1 MP
        # (at 1.48 MP a sword's box landed on the armour's chest).
        ground_max_pixels: int = 4_194_304,
        object_ground_max_pixels: int = 1_048_576,
        asr_languages: list[str] | None = None,
        # Audio longer than this is decoded in batched windows of at most 30 s, as interfaze does:
        # sequentially, 95 minutes took 27. Shorter audio keeps the sequential decoder.
        asr_window_after_s: float = 120.0,
        asr_batch_size: int = 16,
        # Speakers are joined to words. Past this length the words' timing is estimated
        # from the segments instead of decoded: decoding it took 6.6 minutes for a
        # 95-minute recording, past the ~350 s a relayed request may wait for an answer.
        asr_exact_words_up_to_s: float = 1800.0,
        **kwargs,
    ):
        self.components = {**DEFAULT_COMPONENTS, **(components or {})}
        self.asr_window_after_s = asr_window_after_s
        self.asr_batch_size = asr_batch_size
        self.asr_exact_words_up_to_s = asr_exact_words_up_to_s
        self.bundle_weights = bundle_weights
        self.device_map = device_map
        self.max_context = max_context
        # Off means every component loads at __init__. On (default) means a caller who
        # only wants OCR never pulls the 28 GB brain.
        self.lazy_load = lazy_load
        self.detection_score_threshold = detection_score_threshold
        self.detection_mask_threshold = detection_mask_threshold
        self.ocr_stitch_min_containment = ocr_stitch_min_containment
        self.ocr_max_pixels = ocr_max_pixels
        self.ocr_dense_max_pixels = ocr_dense_max_pixels
        self.ocr_upscale_below_pixels = ocr_upscale_below_pixels
        self.line_detector_max_pixels = line_detector_max_pixels
        self.ocr_small_line_ratio = ocr_small_line_ratio
        self.ocr_retry_confidence = ocr_retry_confidence
        self.ocr_unread_min_confidence = ocr_unread_min_confidence
        self.ocr_unread_alone_min_confidence = ocr_unread_alone_min_confidence
        self.ocr_max_new_tokens = ocr_max_new_tokens
        self.pdf_render_dpi = pdf_render_dpi
        self.pdf_frame_dpi = pdf_frame_dpi
        self.detect_max_new_tokens = detect_max_new_tokens
        self.lam_generation_mode = lam_generation_mode
        self.ocr_vlm_url = ocr_vlm_url or os.environ.get("OCR_VLM_URL") or None
        self.ocr_vlm_served_name = ocr_vlm_served_name
        self.ocr_page_concurrency = ocr_page_concurrency
        # Detector worker processes (line_runtime.Runtime), so page workers stop queueing on
        # one engine. Each loads its own detection, recognition and layout weights, hence a bound.
        self.line_detector_pool_size = line_detector_pool_size
        self.ground_max_pixels = ground_max_pixels
        self.object_ground_max_pixels = object_ground_max_pixels
        self.max_pdf_pages = max_pdf_pages
        # The 25 languages the fast recogniser TDT v3 covers; anything else routes to the speech recogniser.
        self.asr_languages = asr_languages or [
            "bg",
            "hr",
            "cs",
            "da",
            "nl",
            "en",
            "et",
            "fi",
            "fr",
            "de",
            "el",
            "hu",
            "it",
            "lv",
            "lt",
            "mt",
            "pl",
            "pt",
            "ro",
            "sk",
            "sl",
            "es",
            "sv",
            "ru",
            "uk",
        ]
        super().__init__(torch_dtype=torch_dtype, **kwargs)

    def component(self, name: str) -> str:
        try:
            return self.components[name]
        except KeyError as exc:
            raise KeyError(f"unknown component {name!r}; known: {sorted(self.components)}") from exc
