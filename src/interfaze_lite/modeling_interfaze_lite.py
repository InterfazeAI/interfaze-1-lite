"""Interfaze Lite -- a composite perception + routing model for `transformers`.

    from transformers import AutoModel
    model = AutoModel.from_pretrained("interfaze-ai/interfaze-1-lite", trust_remote_code=True)

    model.ocr("invoice.png")                       # sections, lines, words, bounds, confidence
    model.transcribe("call.wav", by_speaker=True)  # speaker-attributed transcript
    model.detect("street.jpg", ["car", "dog"])     # labels, boxes, outlines; masks when asked
    model.ground("screen.png", ["the save button"])
    model.chat([{"role": "user", "content": "what does this say?"}], files=["scan.pdf"])

Design notes
------------
Components load lazily. Each accessor materialises its component on first use and
caches it, so an OCR-only caller never pulls the 28 GB brain. Set
`config.lazy_load=False` to force everything up front.

Every method returns the interfaze wire shapes -- four-corner `bounds`, OCR
`sections[].lines[].words[]`, `{text, chunks}` -- via the shared `contracts` module,
so this and the Dockerised service cannot drift apart.

This is the reference implementation. It runs the brain through plain `transformers`,
which means no paged attention and no continuous batching. For throughput, serve the
brain with vLLM using the container in the GitHub repo; the perception components are
identical either way.
"""

from __future__ import annotations

import base64
import concurrent.futures
import gc
import io
import os
import re
import threading
import time
from collections.abc import Sequence
from typing import Any

import torch
from transformers import PreTrainedModel

# trust_remote_code ships a sibling only when a `from .module import name` line names it
# (`from . import module` is not recognised), so each sibling is imported by name once.
try:  # the interfaze_lite package, and this repo as trust_remote_code imports it
    from . import contracts, forecasting, grounding
    from .agent import run as run_agent
    from .configuration_interfaze_lite import InterfazeLiteConfig
    from .contracts import OCRLine
    from .forecasting import MAX_CONTEXT, MAX_HORIZON
    from .grounding import MAX_TOKENS as GROUND_MAX_TOKENS
    from .grounding import MAX_TOKENS_UI as GROUND_MAX_TOKENS_UI
    from .guard import classifier_verdict
    from .line_runtime import Runtime, layout_pages, line_rows
    from .prompts import NUDGE
except ImportError:  # the files run flat, from a checkout
    import contracts  # type: ignore
    import forecasting  # type: ignore
    import grounding  # type: ignore
    from agent import run as run_agent  # type: ignore
    from configuration_interfaze_lite import InterfazeLiteConfig  # type: ignore
    from contracts import OCRLine  # type: ignore
    from forecasting import MAX_CONTEXT, MAX_HORIZON  # type: ignore
    from grounding import MAX_TOKENS as GROUND_MAX_TOKENS  # type: ignore
    from grounding import MAX_TOKENS_UI as GROUND_MAX_TOKENS_UI  # type: ignore
    from guard import classifier_verdict  # type: ignore
    from line_runtime import Runtime, layout_pages, line_rows  # type: ignore
    from prompts import NUDGE  # type: ignore


def _hf_token() -> str | None:
    """The HF token under whichever name the environment uses, mirrored to HF_TOKEN for gated components."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "hf_token"):
        value = os.environ.get(key)
        if value:
            os.environ.setdefault("HF_TOKEN", value)
            return value
    return None


def _fast_asr_enabled() -> bool:
    return os.environ.get("ENABLE_FAST_ASR", "0") not in ("0", "", "false", "False")


# The reference decoder's repetition guards: a window that compresses like a loop or scores
# too low is re-decoded hotter. Greedy alone looped "5-5-5" for 60 s of a card number.
_RECOGNISER_GUARDS = {
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}


def _language_code(detected: str | None) -> str | None:
    """ISO 639-1, as interfaze reports it. The recogniser names languages ("english")."""
    if not detected or len(detected) <= 3:
        return detected
    from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

    return TO_LANGUAGE_CODE.get(detected.lower(), detected)


def _explicit_language(language: str | None) -> str | None:
    """The caller's language when they named one; None for "" or "auto"."""
    return language if language and language != "auto" else None


def _segments(words: list) -> list:
    """Sentence-length spans from word timings, for a recogniser that only emits words."""
    spans, current = [], []
    for word in words:
        if current and word.start - current[-1].end > 1.0:
            spans.append(current)
            current = []
        current.append(word)
        if word.text.rstrip().endswith((".", "?", "!")):
            spans.append(current)
            current = []
    if current:
        spans.append(current)
    return [
        contracts.Word(text=" ".join(w.text for w in span), start=span[0].start, end=span[-1].end)
        for span in spans
    ]


def _load_image(src: str | bytes | Any):
    from PIL import Image, UnidentifiedImageError

    if hasattr(src, "convert"):
        return src.convert("RGB")

    if isinstance(src, (bytes, bytearray)):
        data, origin = bytes(src), "<bytes>"
    elif isinstance(src, str) and src.startswith(("http://", "https://")):
        # Some CDNs (Wikimedia) answer a missing User-Agent with an HTML page and a 200.
        resp = contracts.fetch_url(src, timeout=60, headers={"User-Agent": contracts.UA, "Accept": "image/*"})
        ctype = resp.headers.get("content-type", "")
        if ctype and not ctype.startswith("image/"):
            head = resp.text[:200].replace("\n", " ") if "text" in ctype else ""
            raise ValueError(f"{src} returned content-type {ctype!r}, not an image. {head}".strip())
        data, origin = resp.content, src
    else:
        return Image.open(src).convert("RGB")

    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(
            f"could not decode an image from {origin} ({len(data)} bytes, starts with {data[:16]!r})"
        ) from exc


class InterfazeLiteModel(PreTrainedModel):
    config_class = InterfazeLiteConfig
    _no_split_modules: list[str] = []  # No top-level parameters: each component is its own checkpoint.
    supports_gradient_checkpointing = False

    def __init__(self, config: InterfazeLiteConfig):
        super().__init__(config)
        self._cache: dict[str, Any] = {}
        # Pages are OCR'd on several threads and can race to build one component; a double load can OOM.
        self._cache_lock = threading.Lock()
        # Detectors run in worker processes (line_runtime): as threads they raced on Paddle's
        # process-wide oneDNN context and segfaulted the perception process.
        self._runtime: Runtime | None = None
        self._runtime_lock = threading.Lock()

        self._dtype = (
            getattr(torch, config.torch_dtype) if isinstance(config.torch_dtype, str) else config.torch_dtype
        )
        # No parameters, so the inherited PreTrainedModel.device raises StopIteration (see `device`).
        self._torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # cuDNN SDPA crashed the document reader on some shapes ("Expected mha_graph.execute(...)
        # .is_good()"); flash and mem-efficient SDPA stay available.
        if torch.cuda.is_available() and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        if not config.lazy_load:
            eager = ["brain", "ocr_vlm", "segmenter", "diarizer"]
            if _fast_asr_enabled():
                eager.insert(3, "asr")
            for name in eager:
                self._component(name)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *_model_args,
        config=None,
        token: str | None = None,
        revision: str | None = None,
        **kwargs,
    ):
        """Build the model from its config alone; components download on first use.

        Keyword arguments naming a config attribute override it. The stock loader would look
        for weights at the repo root, find none, and raise.
        """
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path,
                token=token,
                revision=revision,
                trust_remote_code=True,
                **{k: v for k, v in kwargs.items() if k == "cache_dir"},
            )
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        config._name_or_path = str(pretrained_model_name_or_path)
        model = cls(config)
        model.name_or_path = str(pretrained_model_name_or_path)
        model._hub_token = token
        return model

    def _detection_runtime(self) -> Runtime:
        """The pool of detector processes, started on first use."""
        with self._runtime_lock:
            if self._runtime is None:
                self._runtime = Runtime(
                    self.config.line_detector_pool_size,
                    self._line_detector_settings(),
                    self._layout_settings(),
                )
            return self._runtime

    @property
    def device(self):  # type: ignore[override]
        """Override PreTrainedModel.device, which assumes the module owns parameters."""
        return self._torch_device

    def _repo(self, name: str) -> str:
        """This repo's `name/` folder when bundled, fetched alone on first use; else upstream's."""
        if not self.config.bundle_weights:
            return self.config.component(name)
        root = self.name_or_path or ""
        if os.path.isdir(root):
            return os.path.join(root, name)
        from huggingface_hub import snapshot_download

        local = snapshot_download(  # A Hub id joined with the folder ("org/repo/brain") is no path.
            root,
            allow_patterns=[f"{name}/*"],
            token=getattr(self, "_hub_token", None) or _hf_token(),
            revision=getattr(self.config, "_commit_hash", None),
        )
        return os.path.join(local, name)

    def _component(self, name: str) -> Any:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        with self._cache_lock:
            if name not in self._cache:
                self._cache[name] = getattr(self, f"_load_{name}")()
            return self._cache[name]

    def _load_brain(self):
        """The brain, image-capable (it grounds boxes and reads images), FP8 weights dequantized on load.

        Transformers' FP8 kernels computed nonsense from weights vLLM serves correctly (looping
        tokens from four scripts); dequantized they compute in bf16, exact, for about 54 GB.
        """
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        repo = self._repo("brain")
        kwargs = {}
        quant = getattr(AutoConfig.from_pretrained(repo, token=_hf_token()), "quantization_config", None)
        if isinstance(quant, dict) and quant.get("quant_method") == "fp8":
            from transformers import FineGrainedFP8Config

            kwargs["quantization_config"] = FineGrainedFP8Config(
                activation_scheme=quant.get("activation_scheme", "dynamic"),
                weight_block_size=tuple(quant.get("weight_block_size") or (128, 128)),
                dequantize=True,
            )
        return {
            "model": AutoModelForImageTextToText.from_pretrained(
                repo,
                dtype=self._dtype,
                device_map=self.config.device_map,
                token=_hf_token(),
                **kwargs,
            ).eval(),
            "processor": AutoProcessor.from_pretrained(repo, token=_hf_token()),
        }

    def _load_segmenter(self):
        """The box-prompted segmenter, its pip loader patched to load with assign=True.

        Under torch 2.5+ the plain path leaves meta tensors and the predictor produces empty masks.
        """
        import logging

        import sam2.build_sam as build_sam
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        original = build_sam._load_checkpoint

        def patched(model, ckpt_path):
            if ckpt_path is not None:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
                model.load_state_dict(state, assign=True)
                logging.info("loaded the segmenter checkpoint (assign=True)")

        build_sam._load_checkpoint = patched
        try:
            if self.config.bundle_weights:
                # Bundled: this repo's config.yaml and model.pt, built as build_sam2 builds its own.
                from hydra.utils import instantiate
                from omegaconf import OmegaConf

                folder = self._repo("segmenter")
                cfg = OmegaConf.load(os.path.join(folder, "config.yaml"))
                OmegaConf.resolve(cfg)
                model = instantiate(cfg.model, _recursive_=True)
                patched(model, os.path.join(folder, "model.pt"))
                predictor = SAM2ImagePredictor(model.to(self._torch_device).eval())
            else:
                predictor = SAM2ImagePredictor.from_pretrained(
                    self.config.component("segmenter"), device=str(self._torch_device)
                )
        finally:
            build_sam._load_checkpoint = original

        if torch.cuda.is_available():
            predictor.model = predictor.model.half()
        return predictor

    def _load_ocr_vlm(self):
        """The document VLM via AutoModelForCausalLM, the one auto class its remote code maps.

        That class is the full vision-language model here; AutoModelForImageTextToText refuses its config.
        """
        import inspect
        import types

        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, GenerationMixin

        repo = self._repo("ocr_vlm")
        token = _hf_token()
        # Explicit, so a transformers default cannot reintroduce a backend this checkpoint breaks on.
        attn = os.environ.get("OCR_ATTN_IMPL", "sdpa")
        config = AutoConfig.from_pretrained(repo, trust_remote_code=True, token=token)
        # The vision tower reads its own config, which ships flash_attention_2 whatever the model says.
        config.vision_config.attn_implementation = attn
        if attn != "flash_attention_2":
            _stand_in_for_flash_attn()
        model = (
            AutoModelForCausalLM.from_pretrained(
                repo,
                config=config,
                torch_dtype=self._dtype,
                trust_remote_code=True,
                token=token,
                attn_implementation=attn,
            )
            .to(self._torch_device)
            .eval()
        )
        # Vision rotary frequencies are a non-persistent buffer, so transformers 5's meta-device
        # load left them zero and the reader described pages instead of reading them. Recomputed
        # as the module does (theta 10000).
        for module in model.vision_tower.modules():
            if type(module).__name__ == "VisionRotaryEmbedding":
                dim = module.inv_freq.numel() * 2
                steps = torch.arange(0, dim, 2, dtype=torch.float, device=module.inv_freq.device)
                module.inv_freq = 1.0 / (10000.0 ** (steps / dim))
        # Its prepare_inputs_for_generation indexes cache_position, which transformers 5 no longer
        # passes; from the release with `is_first_iteration`, the inherited one sends the image.
        if (
            "is_first_iteration"
            in inspect.signature(GenerationMixin.prepare_inputs_for_generation).parameters
        ):
            model.prepare_inputs_for_generation = types.MethodType(
                GenerationMixin.prepare_inputs_for_generation, model
            )
        return {
            "model": model,
            "processor": AutoProcessor.from_pretrained(repo, trust_remote_code=True, token=token),
        }

    def _line_detector_settings(self) -> dict:
        """How each worker builds its line detector: detection + recognition, on CPU.

        The GPU build conflicts with torch/vLLM's CUDA runtime; CPU costs about a second.
        """
        settings = {
            "lang": "en",
            "device": "cpu",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            # Each concurrent worker takes its share of the cores; all of them oversubscribed the CPU.
            "cpu_threads": max(1, (os.cpu_count() or 8) // max(1, self.config.line_detector_pool_size)),
        }
        if self.config.bundle_weights:
            # Named as saved: given a bare folder, PaddleOCR assumes its default ("Model name mismatch").
            detector, recognizer = self._repo("line_detector"), self._repo("line_recognizer")
            settings |= {
                "text_detection_model_name": _paddle_name(detector),
                "text_detection_model_dir": detector,
                "text_recognition_model_name": _paddle_name(recognizer),
                "text_recognition_model_dir": recognizer,
            }
        return settings

    def _layout_settings(self) -> dict | None:
        """How each worker builds its layout detector, or None when there is none.

        A small CPU detector types a page's blocks in milliseconds; a structured generation took ~60 s.
        """
        if self.config.bundle_weights:
            folder = self._repo("layout")
            return {"model_name": _paddle_name(folder), "model_dir": folder}
        name = self.config.components.get("layout")
        return {"model_name": name} if name else None

    def _load_asr(self):
        # Optional: imported inside `try` so trust_remote_code's import check does not require it.
        # Absent, _transcribe_fast falls back to the speech recogniser.
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise RuntimeError("ENABLE_FAST_ASR needs nemo_toolkit[asr] installed") from exc

        return nemo_asr.models.ASRModel.from_pretrained(self.config.component("asr"))

    def _load_asr_fallback(self):
        """The speech recogniser, placed by `device_map` rather than moved afterwards.

        Its tied `proj_out` is absent from the checkpoint and initialised on meta, so a later
        `.to(device)` failed every transcription ("Cannot copy out of meta tensor"), intermittently.
        """
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        repo = self._repo("asr_fallback")
        token = _hf_token()
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            repo, dtype=self._dtype, device_map=str(self._torch_device), token=token
        )
        model.tie_weights()
        processor = AutoProcessor.from_pretrained(repo, token=token)

        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            # No chunk_length_s: the recogniser's own long-form decode enables _RECOGNISER_GUARDS and
            # bounds memory per window. Pipeline chunking looped "5-5-5" through a card number,
            # glued words at the seams ("websitesince") and broke word alignment.
            batch_size=1,
        )

    def _load_diarizer(self):
        # Gated upstream: an anonymous fetch is a 401, so HF_TOKEN must be set (see _hf_token).
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(self._repo("diarizer"), token=_hf_token())
        return pipeline.to(self._torch_device)

    def _load_guard(self):
        """The text guard: a safety classifier answering Safe, Controversial or Unsafe and
        its own categories, which guard.classifier_verdict translates to S1-S14."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = self._repo("guard")
        token = _hf_token()
        return {
            "model": AutoModelForCausalLM.from_pretrained(
                repo, dtype=self._dtype, device_map=str(self._torch_device), token=token
            ).eval(),
            "tokenizer": AutoTokenizer.from_pretrained(repo, token=token),
        }

    def _load_forecaster(self):
        """The time-series model, compiled exactly as JigsawStack's forecast service compiles it."""
        import timesfm

        # Built in float32: TimesFM feeds float32 to layers made at the default dtype, which another
        # loader can leave at bf16 ("expected mat1 and mat2 to have the same dtype").
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self._repo("forecaster"), torch_compile=False, device=str(self._torch_device)
            )
        finally:
            torch.set_default_dtype(previous)
        model.model.float()
        model.compile(
            timesfm.ForecastConfig(
                max_context=MAX_CONTEXT,
                max_horizon=MAX_HORIZON,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        return model

    _GUARD_MAX_CHARS = 64_000  # Longer text is judged by its tail: it prefills in one pass on a shared card.

    @torch.inference_mode()
    def moderate(self, text: str) -> dict:
        """The guard's verdict on one user message: `output` is "safe", or "unsafe"
        followed by a comma-separated list of codes on the next line."""
        guard = self._component("guard")
        tokenizer, model = guard["tokenizer"], guard["model"]
        # Its template concatenates message.content, so the text goes in as a string, and
        # writes the assistant turn's opening itself.
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": (text or "")[-self._GUARD_MAX_CHARS :]}], tokenize=False
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        generated = out.sequences[0, prompt_len:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # P(Unsafe) over the three labels at the verdict token. Thresholding it, not taking
        # the likeliest label, is how the guard is calibrated (see _GUARD_UNSAFE_THRESHOLD).
        labels = [
            tokenizer.encode(" " + w, add_special_tokens=False)[0]
            for w in ("Safe", "Unsafe", "Controversial")
        ]
        p_unsafe = None
        for step, token in enumerate(generated.tolist()):
            if token in labels:
                probs = torch.softmax(out.scores[step][0].float(), dim=-1)[labels]
                p_unsafe = float(probs[1] / probs.sum())
                break
        answer = classifier_verdict(answer, p_unsafe, _GUARD_UNSAFE_THRESHOLD)
        return {
            "output": answer,
            "unsafe_probability": p_unsafe,
            "prompt_tokens": int(prompt_len),
            "completion_tokens": int(generated.shape[0]),
        }

    @torch.inference_mode()
    def forecast(self, y: dict, horizon: int) -> dict:
        """The next `horizon` values of a date -> value series, oldest first.

        JigsawStack's service, step for step: dates parsed and sorted, the step inferred
        from them, the horizon clamped to the compiled maximum, the last 1,024 points as
        context, and the forecast rounded to integers when every input was one.
        """
        import numpy as np

        integers = forecasting.all_integers(list(y.values()))
        dts, values = forecasting.parse_dates_sorted(y)
        if len(values) < forecasting.MIN_FORECAST_POINTS:
            raise ValueError("At least 5 unique dates are required")
        step = forecasting.infer_step(dts)
        horizon = max(1, min(int(horizon), MAX_HORIZON))
        series = np.asarray(values, dtype=np.float32)[-MAX_CONTEXT:]
        try:
            point, _ = self._component("forecaster").forecast(horizon=horizon, inputs=[series])
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        preds = point[0].tolist()
        preds = (
            np.rint(np.asarray(preds, dtype=np.float64)).astype(np.int64).tolist()
            if integers
            else [round(float(p), 4) for p in preds]
        )
        return {"timestamp": forecasting.future_timestamps(dts[-1], horizon, step), "value": preds}

    @torch.inference_mode()
    def detect(self, image, prompts: Sequence[str], *, return_masks: bool = False) -> dict:
        """Open-vocabulary detection, as the service's object_detection tool answers it.

        Boxes and labels come from the brain, asked the shared grounding question
        (`grounding.prompt`); outlines, and masks when asked for, from the segmenter.
        Returns `{"detected_objects": [{label, bounds, polygon?, mask?}], width, height}`
        in the image's own pixels.
        """
        wanted = [p for p in prompts if p and p.strip()]
        source = _load_image(image)
        if not wanted:
            return {"detected_objects": [], "width": source.width, "height": source.height}
        # Photographs are grounded at 1 MP: at 1.48 MP the sword's box landed on the
        # armour's chest and "every object" looped to the token cap.
        img = _fit_pixels(source, self.config.object_ground_max_pixels)
        raw = self._brain_vision(img, grounding.prompt(wanted, "object"), max_new_tokens=GROUND_MAX_TOKENS)
        found = grounding.to_pixels(grounding.parse(raw, ", ".join(wanted)), source.width, source.height)
        masks: list[str] = []
        outlines: list = []
        if found:
            boxes = [
                [
                    e["bounds"]["top_left"]["x"],
                    e["bounds"]["top_left"]["y"],
                    e["bounds"]["bottom_right"]["x"],
                    e["bounds"]["bottom_right"]["y"],
                ]
                for e in found
            ]
            masks, outlines = self._segment_boxes_with_outlines(source, boxes)
        objects = []
        for i, element in enumerate(found):
            entry: dict = {"label": element["label"], "bounds": element["bounds"]}
            if i < len(outlines) and outlines[i]:
                entry["polygon"] = outlines[i]
            if return_masks and i < len(masks):
                entry["mask"] = masks[i]
            objects.append(entry)
        return {"detected_objects": objects, "width": source.width, "height": source.height}

    def _fit_for_ocr(self, image):
        """The image as the OCR path bounds it (`ocr_max_pixels`): the grid /segment cuts
        masks on, whatever budget the boxes were grounded at."""
        return _fit_pixels(_load_image(image), self.config.ocr_max_pixels)

    def _segment_boxes_with_outlines(
        self, img, boxes: list[list[int]]
    ) -> tuple[list[str], list[list | None]]:
        """Masks and their outlines, from one segmentation pass."""
        import numpy as np

        predictor = self._component("segmenter")
        with (
            _SEGMENTER,
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()),
        ):
            predictor.set_image(np.array(img))
            masks, _scores, _ = predictor.predict(
                box=np.array(boxes, dtype=np.float32), multimask_output=False
            )

        masks = np.asarray(masks)
        if masks.ndim == 4:
            masks = masks[:, 0]
        elif masks.ndim == 2:
            masks = masks[None]
        masks = masks.astype(bool)
        return ([_encode_mask_array(m) for m in masks], [_outline_of(m) for m in masks])

    @torch.inference_mode()
    def ocr(self, source, *, return_markdown: bool = False, page_range: list[int] | None = None) -> dict:
        """OCR an image or a PDF; a .docx or plain-text file is read as text, without boxes.

        PDFs are rendered page by page and concatenated, matching the published contract:
        `sections` flattened across pages, `height` summed, `width` the widest page, and bounds
        left page-local, each section stamped with its page and that page's frame. `page_range`
        is 1-based inclusive and clamped; at most `config.max_pdf_pages` pages are read.
        """
        kind = contracts.document_kind(source)
        if kind == "pdf":
            return self._ocr_pdf(source, return_markdown=return_markdown, page_range=page_range)
        if kind == "docx":
            return _ocr_docx(source)
        if kind == "text":
            return _read_text_document(source)
        return self._ocr_image(source, return_markdown=return_markdown)

    def _ocr_pdf(self, source, *, return_markdown: bool, page_range) -> dict:
        import pymupdf  # `fitz` is the deprecated alias
        from PIL import Image

        data = contracts.fetch_bytes(source)
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            total = doc.page_count
            first, last = 1, total
            if page_range and len(page_range) == 2:
                # 1-based inclusive, per the tool schema. Clamp rather than raise: a
                # model asking for pages 1-50 of a 3-page file means "all of it".
                first = max(1, int(page_range[0]))
                last = min(total, int(page_range[1]))
                if first > last:
                    first, last = last, first

            # Capped: all 77 pages of a paper blew the request deadline and returned nothing.
            capped = min(last, first - 1 + self.config.max_pdf_pages)

            pages, frames = [], []
            for index in range(first - 1, capped):
                page = doc.load_page(index)
                rect = page.rect
                dpi = contracts.pdf_render_dpi(
                    rect.width,
                    rect.height,
                    dpi=self.config.pdf_render_dpi,
                    max_pixels=self.config.ocr_max_pixels,
                    small_pixels=self.config.ocr_upscale_below_pixels,
                )
                pixmap = page.get_pixmap(dpi=dpi)
                pages.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
                # Boxes are reported at interfaze's render scale whatever the read used:
                # a client draws them on the page rendered at that scale.
                scale = self.config.pdf_frame_dpi / 72.0
                frames.append((round(rect.width * scale), round(rect.height * scale)))

        fitted = [_fit_pixels(page, self.config.ocr_max_pixels) for page in pages]

        # Line detection (worker processes) and the read (HTTP to a continuously batching vLLM)
        # run across pages and alongside each other: they compete for different things, so a
        # barrier only adds their durations (9.0 s + 8.5 s vs 9.0 s overlapped). submit(), not
        # map(), which is that barrier.
        detect_workers = max(1, min(self.config.line_detector_pool_size, len(fitted)))
        read_workers = max(1, min(self.config.ocr_page_concurrency, len(fitted)))

        # One layout job for every page: cheap enough to finish inside the other stages, and
        # fanned out it would only take workers from line detection.
        def classify_pages() -> list[tuple[list[dict], str]]:
            return [self._detect_layout(img) for img in fitted]

        with concurrent.futures.ThreadPoolExecutor(max_workers=detect_workers + read_workers + 1) as pool:
            detect_jobs = [pool.submit(self._detect_text_lines, img) for img in fitted]
            read_jobs = [pool.submit(self._read_document, img) for img in fitted]
            layout_job = pool.submit(classify_pages)
            detector_lines = [job.result() for job in detect_jobs]
            documents = [job.result() for job in read_jobs]
            layouts = layout_job.result()

        results = []
        for img, frame, lines, (regions, markdown), (blocks, status) in zip(
            fitted, frames, detector_lines, documents, layouts, strict=True
        ):
            results.append(
                contracts.rescale_page(
                    self._assemble(
                        img,
                        lines,
                        regions,
                        markdown,
                        return_markdown=return_markdown,
                        layout=blocks,
                        layout_status=status,
                    ),
                    *frame,
                )
            )

        sections: list[dict] = []
        for offset, result in enumerate(results):
            # Bounds stay page-local, as interfaze keeps them; stacking pages onto one canvas
            # moved every box after the first page.
            sections.extend(
                contracts.backfill_sections(
                    result.get("sections") or [],
                    page=first + offset,
                    width=result.get("width") or 0,
                    height=result.get("height") or 0,
                )
            )

        page_text = "\n\n".join(r["text"] for r in results if r.get("text"))
        return {
            "text": page_text,
            # Each page's text on its own, so the model can be shown where pages start.
            "page_texts": [r.get("text") or "" for r in results],
            "context": page_text if return_markdown else None,
            "sections": sections,
            "has_text": any(r.get("has_text") for r in results),
            # interfaze's back-compat aggregate; it matches no page, hence each section's own frame.
            "width": max((r.get("width") or 0) for r in results) if results else 0,
            "height": sum((r.get("height") or 0) for r in results),
            "total_pages": total,
            "pages_processed": [first, last],
            "layout": [
                {**block, "page": offset + first}
                for offset, result in enumerate(results)
                for block in (result.get("layout") or [])
            ],
            # Per page, because a document can lose layout on page 7 alone and an
            # aggregate that reports only "ok" would read as a clean run.
            "layout_status": "; ".join(sorted({str(r.get("layout_status") or "not run") for r in results}))
            if results
            else "no pages",
        }

    def _ocr_image(self, image, *, return_markdown: bool = False) -> dict:
        """Two-stage OCR: a detector supplies geometry and confidence, a VLM supplies structure.

        The reader emits no confidence; the detector's scores feed `_denser_budget` and
        `contracts.drop_unread`.
        """
        # Bounded once, so detector and VLM share one pixel grid; capped in the VLM path alone,
        # the detector chewed a 16.8 MP render (65 s, not 5 s) against a different grid.
        source = contracts.enhance_scan(_load_image(image))
        img = _fit_pixels(source, self.config.ocr_max_pixels)
        if source.width * source.height < self.config.ocr_upscale_below_pixels:
            # Small type in a small image is a few pixels a glyph, which the reader
            # guesses at; twice the pixels and it reads it. Boxes are mapped back below.
            img = _fit_pixels(_upscale(source, 2), self.config.ocr_max_pixels)

        # Detection (CPU) and the read (HTTP to a GPU) contend for nothing, so _read_page runs them
        # together (serially 9.0 s + 8.5 s on a 0.69 MP page). `_denser_budget` needs the line
        # heights first, so a second pass is paid only when a page escalates.
        t0 = time.perf_counter()
        detector_lines, regions, vlm_markdown, layout, layout_status = self._read_page(img)
        t_page = time.perf_counter() - t0

        bigger = self._denser_budget(source, img, detector_lines)
        if bigger:
            img = bigger
            t0 = time.perf_counter()
            detector_lines, regions, vlm_markdown, layout, layout_status = self._read_page(img)
            t_page += time.perf_counter() - t0

        # One figure: the stages overlap, so splitting the wall-clock between them would
        # imply a division that does not exist.
        print(
            f"[ocr timing] {img.width}x{img.height} "
            f"detect+vlm(parallel)={t_page:.1f}s "
            f"lines={len(detector_lines)} regions={len(regions)}",
            flush=True,
        )
        return contracts.rescale_page(
            self._assemble(
                img,
                detector_lines,
                regions,
                vlm_markdown,
                return_markdown=return_markdown,
                t_detect=t_page,
                t_vlm=0.0,
                layout=layout,
                layout_status=layout_status,
            ),
            source.width,
            source.height,
        )

    def _detect_layout(self, img) -> tuple[list[dict], str]:
        """Typed blocks for one page (label, score, bounds in the image's frame) and a status.

        The detector's result wrapper differs between releases (a dict, a dict under "res", an
        object with .json); the status tells an unrecognised shape from an empty page. Returned,
        not stored, because pages are classified concurrently.
        """
        # Via _layout_settings: a bundled config names no layout component.
        try:
            if self._layout_settings() is None:
                return [], "unavailable: no layout component configured"
        except Exception as exc:
            return [], f"unavailable: {type(exc).__name__}: {exc}"[:200]

        try:
            pages = self._detection_runtime().run(layout_pages, _to_numpy(img)) or []
        except Exception as exc:
            return [], f"predict failed: {type(exc).__name__}: {exc}"[:200]

        if not pages:
            return [], "predict returned no pages"

        blocks: list[dict] = []
        shapes: list[str] = []
        for page in pages:
            data: Any = page
            for accessor in ("res", "json"):
                if isinstance(data, dict) and accessor in data:
                    data = data[accessor]
                elif not isinstance(data, dict) and hasattr(data, accessor):
                    data = getattr(data, accessor)
            if isinstance(data, dict) and "res" in data:
                data = data["res"]

            found: list = []
            if isinstance(data, dict):
                for key in ("boxes", "layout_boxes", "dt_polys", "det_boxes"):
                    if data.get(key):
                        found = data[key]
                        break
                if not found:
                    shapes.append(f"{type(page).__name__}:{sorted(data)[:6]}")
            else:
                shapes.append(type(page).__name__)

            for box in found:
                coords = box.get("coordinate") or box.get("bbox") if isinstance(box, dict) else None
                if not coords or len(coords) != 4:
                    continue
                x1, y1, x2, y2 = (float(v) for v in coords)
                if x2 <= x1 or y2 <= y1:
                    continue
                blocks.append(
                    {
                        "type": str(box.get("label") or "text").lower(),
                        "score": round(float(box.get("score") or 0.0), 4),
                        "bounds": contracts.to_bounds(
                            [x1, y1, x2, y2], src="xyxy", img_w=img.width, img_h=img.height
                        ).as_dict(),
                    }
                )

        blocks.sort(key=lambda b: (b["bounds"]["top_left"]["y"], b["bounds"]["top_left"]["x"]))
        if not blocks and shapes:
            return blocks, f"no boxes in result shape {shapes[0]}"[:200]
        return blocks, "ok"

    def _read_page(self, img):
        """Detect lines, classify layout, and read the document at the same time.

        Each thread waits on detector processes or the VLM socket, so none holds the GIL for the costly part.
        """

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            detect = pool.submit(self._detect_text_lines, img)
            read = pool.submit(self._read_document, img)
            layout = pool.submit(self._detect_layout, img)
            return detect.result(), *read.result(), *layout.result()

    def _denser_budget(self, source, img, detector_lines):
        """Re-fit the page at the dense budget when the first read looks pixel-starved.

        Short lines mean small type (a 1 MP read lost a paper's subscripts); low mean confidence
        means a degraded scan. Neither fires on an ordinary page.
        """
        if self.config.ocr_dense_max_pixels <= self.config.ocr_max_pixels:
            return None
        if img.width * img.height >= source.width * source.height:
            return None  # already showing every pixel there is
        if not detector_lines:
            return None

        heights = [line.bounds.height for line in detector_lines if line.bounds.height > 0]
        if not heights:
            return None
        median_height = sorted(heights)[len(heights) // 2]
        small = median_height / img.height < self.config.ocr_small_line_ratio

        scores = [line.average_confidence for line in detector_lines if line.average_confidence is not None]
        weak = bool(scores) and (sum(scores) / len(scores)) < self.config.ocr_retry_confidence

        if not (small or weak):
            return None
        denser = _fit_pixels(source, self.config.ocr_dense_max_pixels)
        if denser.width <= img.width:
            return None
        print(
            f"[ocr] re-reading at {denser.width}x{denser.height} (small_type={small} low_confidence={weak})",
            flush=True,
        )
        return denser

    def _assemble(
        self,
        img,
        detector_lines,
        regions,
        vlm_markdown,
        *,
        return_markdown: bool,
        t_detect: float = 0.0,
        t_vlm: float = 0.0,
        layout: list[dict] | None = None,
        layout_status: str = "not run",
    ) -> dict:
        """Stitch one page's detector geometry to its VLM text."""

        regions, vlm_markdown = contracts.restore_marks(regions, vlm_markdown, detector_lines)
        detector_lines = contracts.drop_unread(
            regions,
            detector_lines,
            self.config.ocr_unread_min_confidence,
            self.config.ocr_unread_alone_min_confidence,
        )
        stitched = (
            contracts.stitch(regions, detector_lines)
            if regions
            else contracts.merge_lines_into_sections(detector_lines)
        )

        # ONE SECTION PER PAGE, by interfaze's contract: consumers index `sections[0]` as the page
        # and derive page height as `height / sections.length`; a section per region scaled every
        # box by a third and put it on the wrong page.
        sections = [contracts.collapse_to_page(stitched)] if stitched else []

        # `text` is the reader's COMPLETE document; the detector's sure lines stand in only when it
        # read nothing. Built from regions, it lost whatever was read without a usable box.
        region_text = "\n".join(s.text for s in sections if s.text.strip())
        # Returned by the reader rather than stashed on self: pages are OCR'd
        # concurrently, and instance state would let one page's text overwrite another's.
        markdown = vlm_markdown or region_text
        return {
            "_timing": {"detect_s": round(t_detect, 2), "vlm_s": round(t_vlm, 2)},
            "text": markdown,
            "context": markdown if return_markdown else None,
            "sections": contracts.backfill_sections(
                [s.as_dict() for s in sections], page=1, width=img.width, height=img.height
            ),
            "has_text": bool(markdown.strip()),
            "layout": contracts.attach_text_to_layout(
                layout or [],
                detector_lines,
                min_containment=self.config.ocr_stitch_min_containment,
                regions=regions or (),
            ),
            "layout_status": layout_status,
            "width": img.width,
            "height": img.height,
            "total_pages": 1,
        }

    def _detect_text_lines(self, img) -> list[OCRLine]:
        """Lines with quads and confidence, from the line detector's detection + recognition."""
        # The C++ detector aborts (`RuntimeError: std::exception`) on large inputs and supplies only
        # geometry and confidence, so it reads a downscaled copy whose boxes are scaled back exactly.
        detect_img = _fit_pixels(img, self.config.line_detector_max_pixels)
        scale_x = img.width / detect_img.width
        scale_y = img.height / detect_img.height

        arr = _to_numpy(detect_img)
        rows = self._detection_runtime().run(line_rows, arr)

        lines: list[OCRLine] = []
        for quad, text, score in rows:
            if not (text or "").strip():
                continue
            if scale_x != 1.0 or scale_y != 1.0:
                quad = [[x * scale_x, y * scale_y] for x, y in quad]
            bounds = contracts.to_bounds(quad, src="quad", img_w=img.width, img_h=img.height)
            lines.append(
                OCRLine(
                    text=text,
                    bounds=bounds,
                    average_confidence=score,
                    words=contracts.split_words(text, bounds, score),
                )
            )
        return lines

    def _read_document(self, img) -> tuple[list[contracts.VLMRegion], str]:
        """Run the document reader: regions in reading order, plus markdown (`contracts.read_layout`)."""
        # Sent on the reader's own patch grid, so the frame its boxes are measured in is
        # exactly the image it saw; read_layout scales them back to `img`.
        sent = _snap_to_grid(img, _DOC_READER_GRID)
        budget = _token_budget(sent, self.config.ocr_max_new_tokens)

        # vLLM when configured: the largest latency lever here, against ~15 tok/s from bare
        # generate(), and without its 4x run-to-run variance.
        if self.config.ocr_vlm_url:
            return contracts.read_layout(self._vlm_via_server(sent, budget), sent.size, img)

        ocr = self._component("ocr_vlm")
        model, processor = ocr["model"], ocr["processor"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sent},
                    {"type": "text", "text": _DOC_READER_PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[sent], return_tensors="pt").to(self._torch_device)
        out = _generate(model, inputs, max_new_tokens=budget, eos_token_id=_stop_ids(model, processor))
        raw = processor.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return contracts.read_layout(raw, sent.size, img)

    def _vlm_via_server(self, img, budget: int) -> str:
        """One chat call to the co-located document reader's vLLM server."""
        import httpx

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        resp = httpx.post(
            f"{self.config.ocr_vlm_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": self.config.ocr_vlm_served_name,
                # Upstream's sampling. Measured this way against the reader it replaced.
                "temperature": 0.1,
                "top_p": 1.0,
                "max_tokens": budget,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            # The image placeholder spelled out, as upstream sends it: left
                            # to the server, vLLM inserts a newline before the instruction.
                            {"type": "text", "text": "<|img|><|imgpad|><|endofimg|>" + _DOC_READER_PROMPT},
                        ],
                    }
                ],
            },
            timeout=600,
        )
        resp.raise_for_status()
        return (resp.json()["choices"][0]["message"]["content"] or "").strip()

    # Word timestamps keep cross-attention tensors reachable without inference_mode: live memory
    # climbed 15.2 -> 18.2 GiB across requests until transcription OOMed.
    @torch.inference_mode()
    def transcribe(
        self,
        audio: str,
        *,
        by_speaker: bool = False,
        language: str = "auto",
        word_timestamps: bool = False,
        words_for_speakers: bool = False,
    ) -> dict:
        """Transcribe, and optionally attribute each word to a speaker.

        Returns `{text, language_detected: {code, confidence}, chunks}`: timed segments, words when
        `word_timestamps` or `words_for_speakers`, grouped by speaker when `by_speaker`. With
        ENABLE_FAST_ASR, `config.asr_languages` go to the fast recogniser and the rest to the
        speech recogniser -- a coverage choice, as their multilingual averages match.
        """
        requested = (language or "auto").lower()
        use_fast_asr = _fast_asr_enabled() and requested in self.config.asr_languages
        # Words cost four times what segments do, so they are decoded only when something reads
        # them; for the speaker join alone, a long recording's are estimated (`asr_exact_words_up_to_s`).
        need_words = by_speaker or word_timestamps or words_for_speakers
        estimate_after_s = None if word_timestamps else getattr(self.config, "asr_exact_words_up_to_s", None)

        if use_fast_asr:
            words, detected = self._transcribe_fast(audio), requested
            units = words if need_words else _segments(words)
        else:
            units, detected = self._transcribe_long_form(
                audio, requested, words=need_words, estimate_after_s=estimate_after_s
            )

        result: dict[str, Any] = {
            "text": " ".join(u.text for u in units),
            # The language the model decoded; the request's value only when none was detected.
            "language_detected": {"code": _language_code(detected) or requested, "confidence": 1.0},
        }

        # Always timed, as interfaze returns them: segments by default, words when asked.
        if by_speaker:
            turns = self.diarize(audio)
            result["chunks"] = contracts.group_by_speaker(
                contracts.attribute_speakers(units, turns, fill_nearest=True)
            )
        else:
            result["chunks"] = [u.as_dict() for u in units]

        return result

    def diarize(self, audio: str) -> list[contracts.SpeakerTurn]:
        """Who spoke when. Returns turns only -- no words. Join with `attribute_speakers`."""
        annotation = self._component("diarizer")(_waveform(audio))
        return [
            contracts.SpeakerTurn(speaker=str(label), start=float(seg.start), end=float(seg.end))
            for seg, _, label in contracts.speaker_tracks(annotation)
        ]

    def _transcribe_fast(self, audio: str) -> list[contracts.Word]:
        try:
            model = self._component("asr")
            out = model.transcribe([audio], timestamps=True)[0]
        except Exception as exc:
            # Its toolkit imports torchaudio.io (gone in torchaudio 2.9) and cannot fetch remote audio
            # here; the speech recogniser covers every language it does.
            print(
                f"[asr] the fast recogniser unavailable ({type(exc).__name__}: {exc}); "
                f"falling back to the speech recogniser",
                flush=True,
            )
            return self._transcribe_long_form(audio, words=True)[0]
        return [
            contracts.Word(text=w["word"], start=float(w["start"]), end=float(w["end"]))
            for w in out.timestamp["word"]
        ]

    def _transcribe_long_form(
        self,
        audio: str,
        language: str | None = None,
        *,
        words: bool = False,
        estimate_after_s: float | None = None,
    ) -> tuple[list[contracts.Word], str | None]:
        """Timed spans -- words, or segments -- plus the language actually used.

        `language` goes in as the decoder prompt; given none, `return_language` recovers the detected one.
        """
        from transformers.pipelines.audio_utils import ffmpeg_read

        asr = _one_at_a_time(self._component("asr_fallback"))
        samples = ffmpeg_read(contracts.fetch_bytes(audio), _ASR_RATE)
        if len(samples) > self.config.asr_window_after_s * _ASR_RATE:
            estimate = words and estimate_after_s is not None and len(samples) > estimate_after_s * _ASR_RATE
            if estimate:
                print(
                    f"[asr] {len(samples) / _ASR_RATE / 60:.0f} min: word timing estimated from segments",
                    flush=True,
                )
            units, detected = _transcribe_windows(
                asr,
                samples,
                language,
                words=words and not estimate,
                batch_size=self.config.asr_batch_size,
            )
            return (contracts.estimate_words(units) if estimate else units), detected

        generate = dict(_RECOGNISER_GUARDS)
        explicit_language = _explicit_language(language)
        if explicit_language:
            generate["language"] = explicit_language
        call: dict = {
            "return_language": True,
            "return_timestamps": "word" if words else True,
            "generate_kwargs": generate,
        }

        try:
            out = asr(audio, **call)
        except (TypeError, ValueError):
            # Older transformers reject return_language, and a bad language code is
            # the caller's mistake rather than a reason to return nothing.
            out = asr(audio, return_timestamps=True)
        except IndexError:
            # Word-timestamp collation (`_decode_asr`) indexes past its own list on some generations;
            # segment timing takes another path, so text and coarse timing survive.
            print("[asr] word timestamps failed to collate; using segment timing", flush=True)
            out = asr(audio, **{**call, "return_timestamps": True})

        chunks = out.get("chunks", [])
        detected = next((c.get("language") for c in chunks if c.get("language")), out.get("language"))
        words = [
            contracts.Word(
                text=chunk["text"].strip(),
                start=float(chunk["timestamp"][0]),
                end=float(
                    chunk["timestamp"][1] if chunk["timestamp"][1] is not None else chunk["timestamp"][0]
                ),
            )
            for chunk in chunks
        ]
        return words, (explicit_language or detected)

    @torch.inference_mode()
    def ground(self, image, prompts: Sequence[str] | None = None) -> dict:
        """Locate UI elements in a screenshot, as the service's gui_detection tool does.

        Runs on the brain, which grounds ScreenSpot-v2 well enough that a GUI specialist would
        cost VRAM without a matching gain. With no prompts it finds every interactive element.
        Returns `{"gui_elements": [{type, bounds}], width, height}` in the image's own pixels,
        `type` the phrase that found it, or "icon" when no prompts were given.
        """
        wanted = [p for p in (prompts or []) if p and p.strip()]
        everything = not wanted
        source = _load_image(image)
        img = _fit_pixels(source, self.config.ground_max_pixels)
        elements = []
        for phrase in wanted or ["interactive element (buttons, links, inputs, icons)"]:
            raw = self._brain_vision(
                img, grounding.prompt([phrase], "ui"), max_new_tokens=GROUND_MAX_TOKENS_UI
            )
            found = grounding.to_pixels(grounding.parse(raw, phrase), source.width, source.height)
            elements += [{"type": "icon" if everything else phrase, "bounds": e["bounds"]} for e in found]
        return {"gui_elements": elements, "width": source.width, "height": source.height}

    def _brain_vision(self, img, prompt: str, max_new_tokens: int = 4096) -> str:
        return self._brain_generate(
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}],
                }
            ],
            max_new_tokens=max_new_tokens,
        )

    def _brain_generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        tools: list[dict] | None = None,
        keep_markup: bool = False,
    ) -> str:
        """One brain turn through transformers, with thinking off, as the service runs
        every tool turn and grounding call. Images ride in the messages as PIL images.

        `keep_markup` keeps the tool-call tags, which decoding would otherwise strip."""
        brain = self._component("brain")
        model, processor = brain["model"], brain["processor"]
        images = [
            part["image"]
            for m in messages
            if isinstance(m.get("content"), list)
            for part in m["content"]
            if part.get("type") == "image"
        ]
        text = processor.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = processor(text=[text], images=images or None, return_tensors="pt").to(self._torch_device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=_stop_ids(model, processor)
        )
        generated = out[0][inputs["input_ids"].shape[1] :]
        if not keep_markup:
            return processor.decode(generated, skip_special_tokens=True).strip()
        text = processor.decode(generated, skip_special_tokens=False)
        return re.sub(r"<\|(?:im_end|endoftext)\|>", "", text).strip()

    def chat(
        self, messages: list[dict], *, files: Sequence[str] | None = None, max_new_tokens: int = 4096
    ) -> dict:
        """Answer a conversation, reading `files` with OCR, speech, detection and GUI
        grounding when it needs them.

        The service's tool loop, run in-process (`agent.run`): same prompts, same tools,
        same result shapes. Returns `{"content", "precontext": [{name, result}]}` -- prose
        for a person, each tool's full result for a program. Files are paths or URLs,
        referred to as ref-0, ref-1, ... in order.
        """
        return run_agent(
            messages,
            list(files or []),
            generate=lambda convo, tools=None, keep_markup=False: self._brain_generate(
                convo, max_new_tokens=max_new_tokens, tools=tools, keep_markup=keep_markup
            ),
            tools={
                "ocr": self._tool_ocr,
                "stt": self._tool_stt,
                "object_detection": self._tool_detect,
                "gui_detection": self._tool_ground,
            },
            show_images=self._show_images,
        )

    # Each returns (what the caller gets, what the model reads), as the service's tools do.
    def _tool_ocr(self, path: str, args: dict) -> tuple[dict, dict]:
        result = self.ocr(path, page_range=args.get("page_range"))
        full = {
            "extracted_text": result.get("text") or result.get("context") or "",
            "sections": result.get("sections") or [],
            "width": result.get("width"),
            "height": result.get("height"),
            **({"total_pages": result.get("total_pages")} if result.get("pages_processed") else {}),
        }
        # Pages marked, so the model knows what it has read and does not ask again.
        pages = result.get("page_texts") or []
        first = (result.get("pages_processed") or [1])[0]
        text = (
            "\n\n".join(f"--- page {first + i} ---\n{t}" for i, t in enumerate(pages))
            if len(pages) > 1
            else full["extracted_text"]
        )
        return full, {
            "extracted_text": text,
            "width": full["width"],
            "height": full["height"],
            "total_pages": result.get("total_pages"),
        }

    def _tool_stt(self, path: str, args: dict) -> tuple[dict, dict]:
        result = self.transcribe(path, by_speaker=bool(args.get("by_speaker")))
        return result, result

    def _tool_detect(self, path: str, args: dict) -> tuple[dict, dict]:
        result = self.detect(path, args.get("prompts") or [])
        return result, {
            "detected_objects": [
                {k: o[k] for k in ("label", "bounds") if k in o} for o in result["detected_objects"]
            ]
        }

    def _tool_ground(self, path: str, args: dict) -> tuple[dict, dict]:
        result = self.ground(path, args.get("prompts"))
        return result, result

    def _show_images(self, convo: list[dict], refs: dict[str, str]) -> list[dict]:
        """The caller's images, put beside the conversation once a tool has read them.

        Shown first, the model transcribes by eye and invents text; after the reading, the image
        carries what text cannot -- a highlighted item, a colour, a crossed-out line.
        """
        images = []
        for path in refs.values():
            try:
                if contracts.document_kind(path) != "image":
                    continue
                images.append(_fit_pixels(_load_image(path), self.config.object_ground_max_pixels))
            except Exception:
                continue
            if len(images) == _MAX_SHOWN_IMAGES:
                break
        users = [i for i, m in enumerate(convo) if m.get("role") == "user" and m.get("content") != NUDGE]
        if not images or not users:
            return convo
        message = convo[users[-1]]
        content = (
            message["content"]
            if isinstance(message["content"], list)
            else [{"type": "text", "text": message["content"]}]
        )
        shown = list(convo)
        shown[users[-1]] = {
            **message,
            "content": [*content, *({"type": "image", "image": image} for image in images)],
        }
        return shown


# Keys a vision-language generate() needs: reported "not used by the model", they mean the wrong
# auto class, and dropping them would silently discard the image (see `_generate`).
_ESSENTIAL_INPUTS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    }
)


def _stand_in_for_flash_attn() -> None:
    """Let the reader's vision code import flash_attn where it is not installed.

    It imports flash_attn at module level even under SDPA, and flash-attn has no wheel for most
    torch/CUDA pairings; the stand-in raises if anything calls it.
    """
    import importlib.machinery
    import importlib.util
    import sys
    import types

    if "flash_attn" in sys.modules or importlib.util.find_spec("flash_attn") is not None:
        return

    def flash_attn_varlen_func(*_args, **_kwargs):
        raise RuntimeError("flash_attn is not installed: run the document reader with OCR_ATTN_IMPL=sdpa")

    stand_in = types.ModuleType("flash_attn")
    stand_in.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    stand_in.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = stand_in


def _stop_ids(model, processor) -> list[int]:
    """The ids a turn ends on: the checkpoint's own, plus <|im_end|>.

    Their generation configs list only <|endoftext|>, so generate() read one page on to the
    8192-token cap -- 140 regions instead of 16.
    """
    ids = model.generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end not in ids:
        ids.append(im_end)
    return ids


def _generate(model, inputs, *, max_new_tokens: int, eos_token_id: list[int] | None = None):
    """generate() that drops surplus processor keys transformers rejects, but never _ESSENTIAL_INPUTS."""
    kwargs = dict(inputs)
    stop = {"eos_token_id": eos_token_id} if eos_token_id else {}
    for _ in range(3):
        try:
            return model.generate(**kwargs, **stop, max_new_tokens=max_new_tokens, do_sample=False)
        except ValueError as exc:
            if "not used by the model" not in str(exc):
                raise
            named = set(re.findall(r"'([A-Za-z_][A-Za-z0-9_]*)'", str(exc)))
            essential = named & _ESSENTIAL_INPUTS
            if essential:
                raise ValueError(
                    f"{model.__class__.__name__} rejected essential input(s) "
                    f"{sorted(essential)} -- this checkpoint needs an image-capable auto "
                    f"class (AutoModelForImageTextToText), not a text-only one."
                ) from exc
            droppable = [k for k in named if k in kwargs]
            if not droppable:
                raise
            for key in droppable:
                kwargs.pop(key)
    raise RuntimeError("generate() kept rejecting its own processor output")


_ASR_RATE = 16_000

# Most images shown beside the conversation once a tool has read them (`_show_images`).
_MAX_SHOWN_IMAGES = 4

# One recogniser call on the GPU at a time: six concurrent transcriptions OOMed the brain's vLLM
# engine sharing the card. Per call, not per recording, which kept others waiting ~90 s.
_RECOGNISER = threading.Lock()

# P("Unsafe") a verdict needs to block (see `moderate`). Chosen in evals/guard/guard_bench.py: the safe
# requests it flags score 0.93 or more, and every step up cost recall on every public set.
_GUARD_UNSAFE_THRESHOLD = 0.5

# Held between set_image and predict only: interleaved, one image's masks came from another's embedding.
_SEGMENTER = threading.Lock()


def _one_at_a_time(recogniser):
    def call(*args, **kwargs):
        with _RECOGNISER:
            return recogniser(*args, **kwargs)

    return call


def _loops(text: str, threshold: float = 2.4) -> bool:
    """Text that compresses like a repetition loop, by the reference decoder's own test."""
    import zlib

    raw = text.encode("utf-8")
    return len(raw) > 0 and len(raw) / len(zlib.compress(raw)) > threshold


def _transcribe_windows(
    asr, samples, language: str | None, *, words: bool, batch_size: int
) -> tuple[list[contracts.Word], str | None]:
    """Long audio decoded as interfaze's recogniser does: in windows (`contracts.speech_windows`), in batches.

    The language is detected once, on the first window: transformers 5.15 cannot split a batched
    result that carries languages.
    """
    windows = contracts.speech_windows(samples, _ASR_RATE)
    detected = _explicit_language(language)
    if not windows:
        return [], detected
    generate = dict(_RECOGNISER_GUARDS)
    if detected is None:
        s, e = windows[0]
        probe = asr(
            {"raw": samples[s:e], "sampling_rate": _ASR_RATE},
            return_language=True,
            return_timestamps=True,
            generate_kwargs=dict(generate),
        )
        detected = next(
            (c.get("language") for c in probe.get("chunks", []) if c.get("language")), probe.get("language")
        )
    if detected:
        generate["language"] = detected
    greedy = {"language": detected} if detected else {}

    def window(i):
        # Built per call: the pipeline consumes each input's "raw" as it reads it.
        s, e = windows[i]
        return {"raw": samples[s:e], "sampling_rate": _ASR_RATE}

    def decode(timestamps):
        """Every window, a batch at a time, halving the batch while it does not fit on the shared card.

        Once halved, the size sticks: reset each batch, it failed its way down every time.
        """
        # One batch per call, so another request waits for a batch, not the recording.
        outs, size, first = [], batch_size, 0
        while first < len(windows):
            indices = list(range(first, min(first + size, len(windows))))
            try:
                outs += asr(
                    [window(i) for i in indices],
                    batch_size=len(indices),
                    return_timestamps=timestamps,
                    generate_kwargs=dict(greedy),
                )
                first += len(indices)
                continue
            except torch.cuda.OutOfMemoryError:
                if len(indices) == 1:
                    raise
            # Retried past the except clause, whose live traceback holds the failed batch's tensors:
            # inside it, each half OOMed against the same 18 GB.
            gc.collect()
            torch.cuda.empty_cache()
            size = len(indices) // 2
            print(
                f"[asr] a batch of {len(indices)} windows did not fit in memory; going on {size} at a time",
                flush=True,
            )
        # A window that decoded into a loop is redone alone, where the guards'
        # temperature fallback works.
        return [
            asr(window(i), return_timestamps=timestamps, generate_kwargs=dict(generate))
            if _loops(out.get("text") or "")
            else out
            for i, out in enumerate(outs)
        ]

    try:
        outs = decode("word" if words else True)
    except IndexError:
        # Word collation inside transformers indexes past its own list on some
        # generations; segment timing comes from another path and survives.
        print("[asr] word timestamps failed to collate; using segment timing", flush=True)
        outs = decode(True)

    units: list[contracts.Word] = []
    for (s, e), out in zip(windows, outs, strict=True):
        offset, window_end = s / _ASR_RATE, e / _ASR_RATE
        for chunk in out.get("chunks", []):
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            start, end = chunk["timestamp"]
            start = offset + float(start or 0.0)
            end = offset + float(end) if end is not None else window_end
            units.append(contracts.Word(text=text, start=start, end=max(start, end)))
    return units, detected


def _waveform(audio, rate: int = 16_000) -> dict:
    """Audio decoded in memory, mono at `rate`, in the form the diarization pipeline takes.

    The pipeline's own decoder can come up a few samples short on compressed audio ("439895
    samples instead of the expected 441000") and fail the recording.
    """
    from transformers.pipelines.audio_utils import ffmpeg_read

    samples = ffmpeg_read(contracts.fetch_bytes(audio), rate)
    return {"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": rate}


def _paddle_name(folder: str) -> str:
    """A bundled PaddleOCR model's own name, which PaddleOCR checks, from the inference.yml beside it."""
    import yaml

    with open(os.path.join(folder, "inference.yml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["Global"]["model_name"]


def _fit_pixels(img, max_pixels: int):
    """Downscale to a pixel budget, preserving aspect ratio; visual tokens are roughly pixels/1024."""
    pixels = img.width * img.height
    if pixels <= max_pixels:
        return img
    scale = (max_pixels / pixels) ** 0.5
    # Land on the brain's 32px patch grid, or its processor pads and the coordinate frame
    # its boxes refer to stops matching the image they were measured on.
    width = max(32, int(img.width * scale) // 32 * 32)
    height = max(32, int(img.height * scale) // 32 * 32)
    return img.resize((width, height))


def _snap_to_grid(img, grid: int):
    """The nearest size whose sides are multiples of `grid` -- what the reader's own
    processor would resize to -- so the image sent is the image whose pixels its boxes
    count."""
    width = max(grid, round(img.width / grid) * grid)
    height = max(grid, round(img.height / grid) * grid)
    return img if (width, height) == img.size else img.resize((width, height))


# The document reader's patch grid: 14 px patches merged 2x2.
_DOC_READER_GRID = 28

# The document reader's layout instruction, verbatim from upstream's prompt map
# ("prompt_layout_all_en"). The model is trained against this wording, and
# contracts.read_layout parses the JSON it asks for.
_DOC_READER_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


def _upscale(img, factor: int):
    """`factor` times the size, on the same 32px grid `_fit_pixels` lands on."""
    from PIL import Image

    width = max(32, img.width * factor // 32 * 32)
    height = max(32, img.height * factor // 32 * 32)
    return img.resize((width, height), Image.LANCZOS)


def _token_budget(img, ceiling: int) -> int:
    """Output budget proportional to how much page there is, so a sparse page cannot run to the ceiling."""
    approx_visual_tokens = (img.width * img.height) // 1024
    return int(max(512, min(ceiling, approx_visual_tokens * 4)))


def _ocr_docx(source) -> dict:
    """Read a .docx's text directly: it holds the text exactly, which OCR could only degrade.

    No bounding boxes: a Word document has no pixel layout until rendered, so any box would be invented.
    """
    import xml.etree.ElementTree as ET
    import zipfile

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(contracts.fetch_bytes(source))) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))

    paragraphs = []
    for para in root.iter(f"{{{ns['w']}}}p"):
        text = "".join(node.text or "" for node in para.iter(f"{{{ns['w']}}}t"))
        if text.strip():
            paragraphs.append(text.strip())

    body = "\n".join(paragraphs)
    return {
        "context": body,
        "sections": [{"text": p, "lines": []} for p in paragraphs],
        "has_text": bool(body.strip()),
        "width": 0,
        "height": 0,
        "total_pages": 1,
        "note": "docx text read directly; no bounding boxes exist before rendering",
    }


def _read_text_document(source) -> dict:
    """Return a plain-text file as-is, without geometry (see `_ocr_docx`)."""
    raw = contracts.fetch_bytes(source)
    body = raw.decode("utf-8", errors="replace")
    lines = [line for line in body.splitlines() if line.strip()]
    return {
        "text": body,
        "context": body,
        "sections": [{"text": body, "lines": []}] if body.strip() else [],
        "has_text": bool(body.strip()),
        "width": 0,
        "height": 0,
        "total_pages": 1,
        "line_count": len(lines),
        "note": "text read directly; no bounding boxes exist before rendering",
    }


def _to_numpy(img):
    import numpy as np

    return np.array(img)


# Matching the published contract: outline of the largest region, at most this many
# vertices, simplified until it fits. A shape that cannot reach the cap without a
# tolerance that stops resembling it yields nothing rather than a coarse lie.
_OUTLINE_MAX_POINTS = 40
_OUTLINE_EPSILON = 2.0
_OUTLINE_UNAVAILABLE = False


def _outline_of(mask) -> list[list[int]] | None:
    """Outline of a boolean mask's largest region as [[x, y], ...], or None.

    An occluded object is described by its main body; `bounds` gives its full extent.
    """
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on the image
        # Said once: swallowed, it reads as "this image has no shapes".
        global _OUTLINE_UNAVAILABLE
        if not _OUTLINE_UNAVAILABLE:
            _OUTLINE_UNAVAILABLE = True
            print(f"[detect] outlines unavailable: {type(exc).__name__}: {exc}", flush=True)
        return None
    try:
        found = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # OpenCV 3 returned (image, contours, hierarchy); 4 returns (contours, hierarchy).
        contours = found[-2]
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 3:
            return None

        # Past a tenth of the region's diagonal, simplification no longer resembles the
        # shape, so give up rather than return a triangle standing in for an outline.
        x, y, w, h = cv2.boundingRect(largest)
        ceiling = 0.1 * float(np.hypot(w, h))
        epsilon = _OUTLINE_EPSILON
        simplified = cv2.approxPolyDP(largest, epsilon, True)
        for _ in range(24):
            if len(simplified) <= _OUTLINE_MAX_POINTS:
                break
            epsilon *= 1.5
            if epsilon > ceiling:
                return None
            simplified = cv2.approxPolyDP(largest, epsilon, True)
        if len(simplified) < 3 or len(simplified) > _OUTLINE_MAX_POINTS:
            return None
        return [[int(pt[0][0]), int(pt[0][1])] for pt in simplified]
    except Exception as exc:
        # An outline is an extra; never fail a detection over one -- but say so, or a
        # tracing bug looks exactly like an image with nothing in it.
        print(f"[detect] outline trace failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def _encode_mask_array(mask) -> str:
    """Boolean numpy mask -> base64 PNG, L-mode. Matches the deployed service."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray((mask * 255).astype("uint8"), mode="L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()
