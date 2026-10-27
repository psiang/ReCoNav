"""Build messages for model requests."""

from __future__ import annotations

from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


def build_messages(system_prompt: str, user_prompt: str) -> list[ChatMessage]:
    """Keep stable task instructions in system and sample input in user."""
    if not system_prompt.strip():
        raise ValueError("system_prompt must not be empty")
    if not user_prompt.strip():
        raise ValueError("user_prompt must not be empty")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
