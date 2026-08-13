"""Token-aware Markdown chunking and deterministic lexical tokenization."""

import re
from functools import lru_cache

from retrieval.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    DEFAULT_MODEL_PATH,
)


@lru_cache(maxsize=1)
def _model_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(DEFAULT_MODEL_PATH), local_files_only=True)
    except (ImportError, OSError, ValueError):
        return None


def token_count(text: str) -> int:
    """Count BGE tokens, with a deterministic lightweight fallback."""
    tokenizer = _model_tokenizer()
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False, verbose=False))
    return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*|[^\s]", text))


def _largest_prefix(text: str, max_tokens: int) -> str:
    low, high = 1, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if token_count(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def _largest_suffix(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if token_count(text[len(text) - middle :]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[len(text) - low :].lstrip() if low else ""


def _split_long_unit(unit: str, max_tokens: int) -> list[str]:
    if token_count(unit) <= max_tokens:
        return [unit]
    parts = re.split(r"(?<=[。！？；.!?;])", unit)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if token_count(part) > max_tokens:
            if current:
                chunks.append(current)
                current = ""
            remaining = part
            while token_count(remaining) > max_tokens:
                prefix = _largest_prefix(remaining, max_tokens)
                chunks.append(prefix)
                remaining = remaining[len(prefix) :]
            if remaining:
                current = remaining
        elif current and token_count(current + part) > max_tokens:
            chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)
    return chunks


def split_chunks(
    text: str,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    *,
    size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split Markdown near a target size while enforcing a hard token limit."""
    if size is not None:
        target_tokens = max_tokens = size
    if overlap is not None:
        overlap_tokens = overlap
    if not 0 <= overlap_tokens < target_tokens <= max_tokens:
        raise ValueError("chunk sizes must satisfy 0 <= overlap < target <= max")
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not normalized:
        return []
    units = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    units = [part for unit in units for part in _split_long_unit(unit, max_tokens)]
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = unit if not current else f"{current}\n\n{unit}"
        if current and (token_count(candidate) > max_tokens or token_count(current) >= target_tokens):
            chunks.append(current)
            available = max_tokens - token_count(unit) - 1
            suffix = _largest_suffix(current, min(overlap_tokens, max(0, available)))
            current = f"{suffix}\n\n{unit}" if suffix else unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_table_chunks(
    text: str,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    max_tokens: int = CHUNK_MAX_TOKENS,
) -> list[dict[str, int | str | bool]]:
    """Split a Markdown table only between rows and repeat its heading and header."""
    lines = [line.rstrip() for line in text.strip().splitlines()]
    separator_index = next(
        (
            index
            for index, line in enumerate(lines)
            if index > 0
            and line.lstrip().startswith("|")
            and re.fullmatch(r"\s*\|(?:\s*:?-+:?\s*\|)+\s*", line)
            and lines[index - 1].lstrip().startswith("|")
        ),
        None,
    )
    if separator_index is None:
        return []

    prefix_lines = lines[: separator_index + 1]
    rows = [line for line in lines[separator_index + 1 :] if line.lstrip().startswith("|")]
    if not rows:
        return [{"text": "\n".join(prefix_lines), "row_start": 0, "row_end": 0, "oversize_row": False}]

    prefix = "\n".join(prefix_lines)
    chunks: list[dict[str, int | str | bool]] = []
    current_rows: list[str] = []
    start_row = 1
    for row_number, row in enumerate(rows, 1):
        candidate = "\n".join([prefix, *current_rows, row])
        if current_rows and (token_count(candidate) > max_tokens or token_count("\n".join([prefix, *current_rows])) >= target_tokens):
            chunk_text = "\n".join([prefix, *current_rows])
            chunks.append(
                {
                    "text": chunk_text,
                    "row_start": start_row,
                    "row_end": row_number - 1,
                    "oversize_row": token_count(chunk_text) > max_tokens,
                }
            )
            current_rows = [row]
            start_row = row_number
        else:
            current_rows.append(row)
    chunk_text = "\n".join([prefix, *current_rows])
    chunks.append(
        {
            "text": chunk_text,
            "row_start": start_row,
            "row_end": len(rows),
            "oversize_row": token_count(chunk_text) > max_tokens,
        }
    )
    return chunks


def _markdown_cells(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.replace(r"\|", "|").strip() for cell in re.split(r"(?<!\\)\|", stripped)]


def table_retrieval_text(text: str, max_tokens: int = CHUNK_MAX_TOKENS) -> str:
    """Convert a Markdown table chunk into compact field-value text for retrieval."""
    lines = [line.rstrip() for line in text.strip().splitlines()]
    separator_index = next(
        (
            index
            for index, line in enumerate(lines)
            if index > 0
            and line.lstrip().startswith("|")
            and re.fullmatch(r"\s*\|(?:\s*:?-+:?\s*\|)+\s*", line)
            and lines[index - 1].lstrip().startswith("|")
        ),
        None,
    )
    if separator_index is None:
        return re.sub(r"\s+", " ", text).strip()

    title_lines = [re.sub(r"^#+\s*", "", line).strip() for line in lines[: separator_index - 1]]
    title = " ".join(line for line in title_lines if line)
    headers = _markdown_cells(lines[separator_index - 1])
    rows = [_markdown_cells(line) for line in lines[separator_index + 1 :] if line.lstrip().startswith("|")]

    output = []
    if title:
        output.append(f"表名：{title}")
    output.append(f"字段：{'、'.join(header for header in headers if header)}")
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        fields = [f"{header}：{value}" for header, value in zip(headers, padded) if header and value]
        if fields:
            output.append("；".join(fields))
    semantic_text = "\n".join(output)
    if token_count(semantic_text) <= max_tokens:
        return semantic_text
    return text


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*", text.lower())
    chinese = [token for token in raw_tokens if len(token) == 1 and "\u4e00" <= token <= "\u9fff"]
    bigrams = [left + right for left, right in zip(chinese, chinese[1:])]
    return raw_tokens + bigrams
