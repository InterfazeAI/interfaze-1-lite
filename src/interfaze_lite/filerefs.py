"""Resolve the `ref-N` indirection the tool schemas are built around.

The model emits `ref-0` instead of reproducing a long URL or base64 blob in a tool call.
It still hallucinates filenames ("report.pdf" for "ref-0"), so an exact match on one
registered filename is accepted too; anything else raises with the valid ids listed.
"""

from __future__ import annotations

import base64
import binascii
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile

REF_PATTERN = re.compile(r"^ref-\d+$")
TEXT_URL = re.compile(r"https?://[^\s<>\"'()\[\]]+")
DATA_URI = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.S)


class FileRefError(ValueError):
    pass


@dataclass
class FileRef:
    ref_id: str
    url: str
    filename: str = ""
    mime: str = ""


@dataclass
class FileRefs:
    refs: dict[str, FileRef] = field(default_factory=dict)
    _temp_paths: list[Path] = field(default_factory=list)

    def add(self, url: str, filename: str = "", mime: str = "") -> FileRef:
        ref = FileRef(
            ref_id=f"ref-{len(self.refs)}",
            url=url,
            filename=filename,
            mime=mime or (mimetypes.guess_type(filename or url)[0] or ""),
        )
        self.refs[ref.ref_id] = ref
        return ref

    def add_data_uri(self, uri: str, filename: str = "") -> FileRef:
        """Materialise a data: URI to a temp file, so services get a path, not megabytes of base64."""
        match = DATA_URI.match(uri)
        if not match:
            raise FileRefError("malformed data: URI")
        mime, is_b64, payload = match.group(1) or "", bool(match.group(2)), match.group(3)
        try:
            blob = base64.b64decode(payload) if is_b64 else payload.encode()
        except (binascii.Error, ValueError) as exc:
            raise FileRefError(f"undecodable data: URI payload: {exc}") from exc

        suffix = mimetypes.guess_extension(mime) or ""
        with NamedTemporaryFile(delete=False, suffix=suffix) as fh:
            fh.write(blob)
            path = Path(fh.name)
        self._temp_paths.append(path)
        return self.add(str(path), filename=filename or path.name, mime=mime)

    def resolve(self, value: str) -> str:
        """Turn a tool argument into something a perception service can fetch."""
        if not value or not isinstance(value, str):
            raise FileRefError(f"expected a file reference or URL, got {value!r}")

        if value.startswith(("http://", "https://")):
            # Only a URL the request carries: a text-only question had the model invent one and OCR it.
            if any(ref.url == value for ref in self.refs.values()):
                return value
            raise FileRefError(
                f"{value} is not a file in this request; valid ids: {sorted(self.refs) or 'none'}"
            )
        if value.startswith("data:"):
            return self.add_data_uri(value).url
        if value in self.refs:
            return self.refs[value].url

        by_name = [r for r in self.refs.values() if r.filename == value]
        if len(by_name) == 1:
            return by_name[0].url
        if REF_PATTERN.match(value):
            raise FileRefError(
                f"{value} is not a registered reference; valid ids: {sorted(self.refs) or 'none'}"
            )
        raise FileRefError(
            f"{value!r} is not a file reference or URL. Use one of "
            f"{sorted(self.refs) or 'none'}, or a full http(s) URL."
        )

    def manifest(self) -> str:
        """The block injected into the prompt so the model knows which ids exist."""
        if not self.refs:
            return ""
        lines = [
            f"- {r.ref_id}: {r.filename or r.url}" + (f" ({r.mime})" if r.mime else "")
            for r in self.refs.values()
        ]
        return "All File References:\n" + "\n".join(lines)

    def cleanup(self) -> None:
        for path in self._temp_paths:
            path.unlink(missing_ok=True)
        self._temp_paths.clear()


def _rewrite_part(part: dict, refs: FileRefs) -> dict:
    """One content part: a file part registered and replaced by its text mention, any other as it was."""
    kind = part.get("type")
    if kind == "image_url":
        url = (part.get("image_url") or {}).get("url", "")
        ref = refs.add_data_uri(url) if url.startswith("data:") else refs.add(url)
        return {"type": "text", "text": f"[image {ref.ref_id}]"}
    if kind == "file":
        # OpenAI's generic file part, how SDKs send audio and documents; ignored, answers were invented.
        spec = part.get("file") or {}
        payload = spec.get("file_data") or spec.get("file_url") or ""
        filename = spec.get("filename") or ""
        if payload.startswith(("http://", "https://")):
            # The declared type: a presigned URL's path and the storage's label often say nothing.
            ref = refs.add(payload, filename=filename, mime=spec.get("format") or "")
        elif payload.startswith("data:"):
            ref = refs.add_data_uri(payload, filename=filename)
        elif payload:
            # Bare base64 with the type carried only by the filename.
            mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            ref = refs.add_data_uri(f"data:{mime};base64,{payload}", filename=filename)
        else:
            return part
        label = filename or ref.ref_id
        return {"type": "text", "text": f"[file {ref.ref_id}: {label}]"}
    if kind == "input_audio":
        audio = part.get("input_audio") or {}
        data, fmt = audio.get("data", ""), audio.get("format", "wav")
        # A bare URL wrapped as a data URI decodes to nothing and fails far from the cause.
        if data.startswith(("http://", "https://")):
            ref = refs.add(data, mime=f"audio/{fmt}")
        elif data.startswith("data:"):
            ref = refs.add_data_uri(data)
        else:
            ref = refs.add_data_uri(f"data:audio/{fmt};base64,{data}")
        return {"type": "text", "text": f"[audio {ref.ref_id}]"}
    return part


def _register_text_urls(messages: list[dict], refs: FileRefs) -> None:
    """Register each URL the user's own text carries, unless already a ref; search results' URLs are data."""
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        texts = (
            [content]
            if isinstance(content, str)
            else [p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text"]
        )
        for text in texts:
            for url in TEXT_URL.findall(text or ""):
                url = url.rstrip(".,;:!?")
                if not any(ref.url == url for ref in refs.refs.values()):
                    refs.add(url)


def extract_from_messages(messages: list[dict]) -> tuple[FileRefs, list[dict]]:
    """Register file parts and the user's text URLs as refs; each file part becomes a short text mention."""
    refs = FileRefs()
    rewritten: list[dict] = []

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            rewritten.append(message)
            continue
        rewritten.append({**message, "content": [_rewrite_part(part, refs) for part in content]})

    _register_text_urls(messages, refs)  # After attachments, so one also mentioned registers once.
    return refs, rewritten
