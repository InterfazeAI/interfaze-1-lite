"""Guardrails: the `<guard>` directive, and turning the classifiers' output into a verdict.

Ported from interfaze `helpers/validate.ts` (`validateGuardAndNSFW`) and
`helpers/nsfw/index.ts`. The contract is theirs: which codes exist, what "ALL" expands
to, when a request is blocked, what `content` says when it is ("unsafe S1 S12_IMAGE"),
and the two precontext entries (`text_guardrail_classifier`,
`image_guardrail_classifier`). Pure: the classifiers are called by the orchestrator.
"""

from __future__ import annotations

import re
from typing import Any

TEXT_CODES = tuple(f"S{i}" for i in range(1, 15))
IMAGE_CODES = ("S1_IMAGE", "S12_IMAGE", "S15_IMAGE")
ALL_CODES = (*TEXT_CODES, *IMAGE_CODES)

# The guard model's fixed taxonomy in the contract's codes; S6 and S14 have no source category,
# Jailbreak has no code. Checked against Aegis 2.0's human labels in evals/guard/guard_bench.py.
GUARD_CATEGORY_CODES = {
    "Violent": ("S1", "S9"),
    "Non-violent Illegal Acts": ("S2",),
    "Sexual Content or Sexual Acts": ("S3", "S4", "S12"),
    "PII": ("S7",),
    "Suicide & Self-Harm": ("S11",),
    "Unethical Acts": ("S5", "S10"),
    "Politically Sensitive Topics": ("S13",),
    "Copyright Violation": ("S8",),
    "Jailbreak": (),
}


def classifier_verdict(answer: str, p_unsafe: float | None, threshold: float) -> str:
    """The guard model's answer as the text verdict `verdict` reads: "safe", or "unsafe"
    with the codes on the next line ("unsafe\nS1,S9").

    `answer` is the model's own "Safety: Unsafe\nCategories: Violent". Unsafe counts at
    `p_unsafe` >= `threshold`. Controversial counts as safe except for self-harm: the model calls
    people describing their own distress Controversial, and counting it took Aegis's self-harm
    prompts flagged S11 from 0.66 to 0.79 without flagging any safe test prompt.
    """
    label = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", answer or "")
    named = re.search(r"Categories:\s*(.*)", answer or "")
    categories = [c.strip() for c in named.group(1).split(",")] if named else []
    if label and label.group(1) == "Controversial" and "Suicide & Self-Harm" in categories:
        return "unsafe\nS11"
    if not label or label.group(1) != "Unsafe" or p_unsafe is None or p_unsafe < threshold:
        return "safe"
    codes = [code for c in categories for code in GUARD_CATEGORY_CODES.get(c, ())]
    return "unsafe\n" + ",".join(dict.fromkeys(codes)) if codes else "unsafe"


_GUARD_TAG = re.compile(r"<guard>([\s\S]*?)</guard>")


def extract(system_text: str | None) -> tuple[list[str] | None, str | None]:
    """The requested codes, and the system text with the directive removed.

    None when there is no directive -- including an empty `<guard></guard>`, which
    interfaze reads as no guard at all (`match(...)?.[1]?.trim() || undefined`).
    """
    if not system_text:
        return None, system_text
    match = _GUARD_TAG.search(system_text)
    if not match:
        return None, system_text
    stripped = _GUARD_TAG.sub("", system_text).strip()
    inner = match.group(1).strip()
    if not inner:
        return None, stripped
    if inner.upper() == "ALL":
        return list(ALL_CODES), stripped
    return [c.strip().upper() for c in inner.split(",") if c.strip()], stripped


def weighed_average(scores: list[float]) -> float:
    """Each score weighted by its own square, as interfaze averages image scores."""
    squares = sum(s * s for s in scores)
    if squares == 0:
        return 0.0
    return sum(s * s * s for s in scores) / squares


def image_result(adult: float, racy: float, gore: float, threshold: float = 0.5) -> dict[str, Any]:
    """One image's classification in interfaze's shape, from adult/racy/gore scores.

    interfaze reads these three off Azure's adult analysis: nudity is adult content,
    gore is gory content, NSFW is any of the three, and the NSFW score is their
    weighted average.
    """
    is_adult, is_racy, is_gore = adult >= threshold, racy >= threshold, gore >= threshold
    return {
        "nsfw": is_adult or is_gore or is_racy,
        "nudity": is_adult,
        "gore": is_gore,
        "nsfw_score": weighed_average([adult, gore, racy]),
        "nudity_score": adult,
        "gore_score": gore,
    }


def wants_images(codes: list[str]) -> dict[str, bool]:
    return {"nudity": "S12_IMAGE" in codes, "gore": "S1_IMAGE" in codes, "nsfw": "S15_IMAGE" in codes}


def verdict(
    guard_text: str, codes: list[str], images: list[dict[str, Any]] | None = None
) -> tuple[bool, str, list[dict[str, Any]]]:
    """(is_safe, content if blocked, precontext) exactly as interfaze computes them.

    `guard_text` is the text classifier's raw answer ("safe", or "unsafe\\nS1,S10").
    A verdict naming none of the requested codes is safe; with no codes requested,
    any unsafe verdict blocks.
    """
    is_safe = True
    text = re.sub(r"[\n,]+", " ", guard_text or "").strip()
    warnings = [w.strip() for w in re.split(r"[\n,]", guard_text or "") if w.strip()]
    matched_warnings = (
        [w for w in warnings if w in codes] if codes else [w for w in warnings if w != "unsafe"]
    )

    if text != "safe" and (any(c in warnings for c in codes) or not codes):
        is_safe = False
        text = f"unsafe {' '.join(matched_warnings)}".strip()

    precontext: list[dict[str, Any]] = [
        {"name": "text_guardrail_classifier", "result": "safe" if is_safe else matched_warnings},
    ]

    if images:
        checks = wants_images(codes)
        precontext.append(
            {"name": "image_guardrail_classifier", "result": images[0] if len(images) == 1 else images}
        )
        flagged = [
            code
            for code, key in (("S1_IMAGE", "gore"), ("S12_IMAGE", "nudity"), ("S15_IMAGE", "nsfw"))
            if checks[key] and any(r.get(key) for r in images)
        ]
        if flagged:
            if is_safe:
                is_safe = False
                text = f"unsafe {' '.join(flagged)}"
            else:
                text = f"{text} {' '.join(flagged)}"
    return is_safe, text, precontext


def extract_from_messages(messages: list[dict]) -> tuple[list[str] | None, list[dict]]:
    """The first directive's codes from any system turn, and the messages with every directive removed."""
    codes: list[str] | None = None
    out: list[dict] = []

    def take(text: str) -> str:
        nonlocal codes
        found, rest = extract(text)
        if found is not None and codes is None:
            codes = found
        return rest

    for message in messages:
        content = message.get("content")
        if message.get("role") != "system":
            out.append(message)
        elif isinstance(content, str):
            out.append({**message, "content": take(content)})
        elif isinstance(content, list):
            out.append(
                {
                    **message,
                    "content": [
                        {**part, "text": take(part["text"])}
                        if isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                        else part
                        for part in content
                    ],
                }
            )
        else:
            out.append(message)
    return codes, out
