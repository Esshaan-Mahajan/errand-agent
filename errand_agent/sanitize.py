"""Handling for text the agent did not author: tool free text, PII, and user input at ingress."""
from __future__ import annotations

import re
from typing import Any

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<![\w-])\+?\d[\d\s-]{8,}\d(?![\w-])")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮⁦-⁩]")
_SAFE_LABEL = re.compile(r"[^\w .,&'()-]")

# Used only to flag text in the trace. Enforcement never depends on this list matching.
_INJECTION_HINTS = re.compile(
    r"ignore (all |any )?(previous|prior|above)|system override|system prompt|you are now|"
    r"already approved|do not ask|without asking|without confirmation|call (book|cancel|reschedule)",
    re.IGNORECASE,
)


def redact_pii(text: str) -> str:
    return _PHONE.sub("[phone]", _EMAIL.sub("[email]", text))


def redact(value: Any) -> Any:
    """Recursively mask contact details. Applied to everything written to a trace."""
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


def clean_user_input(text: str, max_chars: int) -> tuple[str, bool]:
    cleaned = _CONTROL.sub("", text or "").strip()
    if len(cleaned) > max_chars:
        return cleaned[:max_chars], True
    return cleaned, False


def safe_label(text: Any, max_len: int = 60) -> str:
    """Reduce a provider-supplied name to something safe to show in an approval prompt."""
    return _SAFE_LABEL.sub("", str(text))[:max_len].strip() or "(unnamed provider)"


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_HINTS.search(text))


def neutralise(text: str, max_len: int = 400) -> str:
    """Free text can't close the data delimiter or smuggle markup into the prompt."""
    cleaned = _CONTROL.sub("", str(text)).replace("<", "(").replace(">", ")")
    return cleaned[:max_len]


def wrap_untrusted(text: str) -> str:
    return f"<untrusted_data>{neutralise(text)}</untrusted_data>"


def project_provider(detail: dict) -> tuple[dict, bool]:
    """Shape get_provider output for the model, returning (projection, injection_suspected)."""
    reviews = detail.get("reviews", [])
    suspected = any(looks_like_injection(str(r.get("text", ""))) for r in reviews)
    return {
        "id": detail.get("id"),
        "name": safe_label(detail.get("name")),
        "price": detail.get("price"),
        "rating": detail.get("rating"),
        "attributes": detail.get("attributes"),
        "bio": wrap_untrusted(detail.get("bio", "")),
        "reviews_are_untrusted_data": True,
        "reviews": [{"rating": r.get("rating"), "text": wrap_untrusted(r.get("text", ""))} for r in reviews],
    }, suspected
