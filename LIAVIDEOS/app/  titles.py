#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
titles.py — lightweight, deterministic title & description generator with hashtags
"""
from __future__ import annotations
import re
from typing import Tuple, List

TAGS = [
    "#shorts", "#foryou", "#viral", "#learn", "#motivation", "#tips", "#highlights"
]

def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def pick_title_and_description(text: str, language: str="en") -> Tuple[str, str]:
    """Heuristics: use first punchy sentence as title, trim ≤80 chars; build desc + hashtags."""
    sents = re.split(r'(?<=[.!?])\s+', _clean(text))
    base = sents[0][:80].strip(" -–—")
    # improve: if looks like a question/hook, keep; else prepend 'Why/How'
    if not re.search(r"[?!]$", base) and len(base.split()) <= 5:
        base = f"Why {base} matters"
    title = base[:80]
    desc = _clean(text)
    hashtags = " ".join(TAGS[:4])
    description = f"{desc}\n\n{hashtags}"
    return title, description
