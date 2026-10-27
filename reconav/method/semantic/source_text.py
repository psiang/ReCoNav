"""Exact source-text occurrence matching shared by semantic validation stages."""

from __future__ import annotations


def lexical_occurrences(text: str, source: str) -> list[tuple[int, int]]:
    """Return exact, case-sensitive occurrences with lexical edge boundaries."""
    if not source:
        return []
    occurrences = []
    cursor = 0
    while True:
        start = text.find(source, cursor)
        if start < 0:
            return occurrences
        end = start + len(source)
        left_ok = (
            not _is_word_character(source[0])
            or start == 0
            or not _is_word_character(text[start - 1])
        )
        right_ok = (
            not _is_word_character(source[-1])
            or end == len(text)
            or not _is_word_character(text[end])
        )
        if left_ok and right_ok:
            occurrences.append((start, end))
        cursor = start + 1


def _is_word_character(value: str) -> bool:
    return value.isalnum() or value == "_"


__all__ = ["lexical_occurrences"]
