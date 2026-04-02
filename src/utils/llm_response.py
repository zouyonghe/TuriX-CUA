from __future__ import annotations

import json
import re


def _strip_code_fences(text: str) -> str:
    cleaned = re.sub(r"^```(json)?", "", text.strip())
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def normalize_llm_json_text(text: str) -> str:
    cleaned = _strip_code_fences(text)
    if not cleaned.startswith("data:"):
        return cleaned

    content_chunks: list[str] = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue

        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict) and any(
            key in parsed for key in ("action", "analysis", "current_state", "read_files")
        ):
            return json.dumps(parsed, ensure_ascii=False)

        if not isinstance(parsed, dict):
            continue

        for choice in parsed.get("choices", []):
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            content = delta.get("content")
            if content is None:
                content = message.get("content")
            if content:
                content_chunks.append(content)

    if content_chunks:
        return _strip_code_fences("".join(content_chunks))

    raise ValueError("LLM SSE response contained no assistant content")
