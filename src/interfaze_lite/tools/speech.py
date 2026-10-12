"""Speech to text, optionally attributed to speakers."""

from __future__ import annotations

from .base import Tool, ToolContext, ToolResult, _post


async def _run_stt(args: dict, ctx: ToolContext) -> ToolResult:
    url = ctx.refs.resolve(args["file_ref_id"])
    by_speaker = bool(args.get("split_by_speaker", False))

    # The speaker join needs word timing, estimated on a long recording unless the caller
    # asked for word timestamps; otherwise segments come back by default.
    transcript = await _post(
        ctx,
        ctx.settings.perception_url,
        "/transcribe",
        {
            "url": url,
            "language": args.get("language_code", "auto"),
            "word_timestamps": bool(args.get("word_timestamps")),
            "words_for_speakers": by_speaker,
        },
    )
    words = transcript.get("chunks") or []

    if by_speaker:
        # ASR and diarization are separate services on purpose; the join happens here,
        # in the orchestrator, using max total temporal overlap.
        from ..contracts import SpeakerTurn, Word, attribute_speakers, group_by_speaker

        turns_raw = await _post(ctx, ctx.settings.diarize_url, "/diarize", {"url": url})
        turns = [SpeakerTurn(t["speaker"], t["start"], t["end"]) for t in turns_raw["turns"]]
        parsed = [
            Word(text=w["text"], start=w["timestamp"][0], end=w["timestamp"][1])
            for w in words
            if w.get("timestamp")
        ]
        chunks = group_by_speaker(attribute_speakers(parsed, turns, fill_nearest=True))
    else:
        # Always timed, as interfaze returns them, whether or not words were asked for.
        from ..contracts import split_long_chunks

        chunks = split_long_chunks(words)

    # {text, chunks}, as interfaze returns it for these requests: it sends no language_detected.
    return ToolResult(model_facing={"text": transcript.get("text", ""), "chunks": chunks})


SPEECH_TO_TEXT = Tool(
    name="stt",
    description="Convert speech to text.",
    parameters={
        "type": "object",
        "properties": {
            "file_ref_id": {
                "type": "string",
                "description": "The file reference id or url of the audio to transcribe.",
            },
            "split_by_speaker": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Whether to split the transcript by speaker. Use only when asked "
                    "to detect or separate speakers."
                ),
            },
            "language_code": {
                "type": "string",
                "default": "auto",
                "description": (
                    "ISO 639-1 two-letter code. When the prompt names or implies the "
                    "language, set it (e.g. 'id', 'en', 'zh'); for a list of "
                    "candidates pick the first. Use 'auto' only when the prompt gives "
                    "no language information."
                ),
            },
            "word_timestamps": {
                "type": "boolean",
                "default": False,
                "description": "Include word-level timestamps. Only when explicitly requested.",
            },
        },
        "required": ["file_ref_id"],
        "additionalProperties": False,
    },
    execute=_run_stt,
)
