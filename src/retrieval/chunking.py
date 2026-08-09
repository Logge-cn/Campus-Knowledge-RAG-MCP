"""Markdown chunking and deterministic lexical tokenization."""

import re

from retrieval.config import CHUNK_OVERLAP, CHUNK_SIZE


def _split_long_unit(unit: str, max_length: int) -> list[str]:
    if len(unit) <= max_length:
        return [unit]
    parts = re.split(r"(?<=[。！？；.!?;])", unit)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if len(part) > max_length:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(part[index : index + max_length] for index in range(0, len(part), max_length))
        elif current and len(current) + len(part) > max_length:
            chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)
    return chunks


def split_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split Markdown at paragraph boundaries while retaining a small text overlap."""
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not normalized:
        return []
    units = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    units = [part for unit in units for part in _split_long_unit(unit, size)]
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = unit if not current else f"{current}\n\n{unit}"
        if current and len(candidate) > size:
            chunks.append(current)
            suffix = current[-overlap:].lstrip()
            current = f"{suffix}\n\n{unit}" if suffix else unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*", text.lower())
    chinese = [token for token in raw_tokens if len(token) == 1 and "\u4e00" <= token <= "\u9fff"]
    bigrams = [left + right for left, right in zip(chinese, chinese[1:])]
    return raw_tokens + bigrams
