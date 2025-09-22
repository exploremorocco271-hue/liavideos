#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ranker.py — select top-N non-overlapping windows, snap to sentence boundaries, enforce <= max_duration
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import logging

LOGGER = logging.getLogger("ranker")

def _non_overlap_select(cands: List[Dict[str, Any]], max_n: int, min_gap: float) -> List[Dict[str, Any]]:
    """Greedy selection by score with no-overlap and >= min_gap spacing."""
    selected: List[Dict[str, Any]] = []
    for c in sorted(cands, key=lambda x: x["score"], reverse=True):
        ok = True
        for s in selected:
            if not (c["end"] + min_gap <= s["start"] or c["start"] >= s["end"] + min_gap):
                ok = False; break
        if ok:
            selected.append(c)
            if len(selected) >= max_n:
                break
    return sorted(selected, key=lambda x: x["start"])

def _snap_to_sentence_bounds(candidate: Dict[str, Any], sentences: List[Tuple[float,float,str]], max_duration: float) -> Dict[str, Any]:
    cs, ce = candidate["start"], candidate["end"]
    # expand to nearest sentence boundary inside limits
    starts = [s for (s,_,_) in sentences if s >= cs - 2 and s <= ce]
    ends = [e for (_,e,_) in sentences if e <= ce + 2 and e >= cs]
    if starts: cs = min(starts)
    if ends:   ce = max(ends)
    if ce - cs > max_duration:
        ce = cs + max_duration
    out = dict(candidate)
    out["start"], out["end"] = float(cs), float(ce)
    out["duration"] = float(ce - cs)
    return out

def finalize(candidates: List[Dict[str, Any]], sentences: List[Tuple[float,float,str]], max_shorts: int, min_gap: float, max_duration: float) -> List[Dict[str, Any]]:
    first = _non_overlap_select(candidates, max_shorts*3, min_gap)  # over-select, then snap & prune
    snapped = [_snap_to_sentence_bounds(c, sentences, max_duration) for c in first]
    final = _non_overlap_select(snapped, max_shorts, min_gap)
    # enforce cap again
    for f in final:
        if f["end"] - f["start"] > max_duration:
            f["end"] = f["start"] + max_duration
            f["duration"] = max_duration
    return final
