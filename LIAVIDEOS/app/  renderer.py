#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
renderer.py — render vertical 1080x1920 MP4 shorts with burned captions via ffmpeg
- Writes SRT for each short
- Letterbox with smart center-crop (opencv optional) or pad
"""
from __future__ import annotations
import logging
import os
import subprocess
import shlex
from typing import Dict, Any, List

import cv2  # optional but included
from moviepy.editor import VideoFileClip

from utils import ensure_dir, seconds_to_tc, which_ffmpeg

LOGGER = logging.getLogger("renderer")

W = 1080
H = 1920
FPS = 30

def _write_srt(words: List[Dict[str, Any]], start: float, end: float, path: str):
    """Combine words into ~1.5s lines for readability, within [start,end]."""
    idx = 1
    cur_words = []
    cur_start = start
    with open(path, "w", encoding="utf-8") as f:
        for w in words:
            ws, we = float(w["start"]), float(w["end"])
            if we < start or ws > end:
                continue
            cur_words.append(w["word"])
            if (we - cur_start) >= 1.5 or w["word"].endswith((".", "!", "?")):
                f.write(f"{idx}\n{seconds_to_tc(cur_start - start)} --> {seconds_to_tc(min(we, end) - start)}\n")
                f.write(" ".join(cur_words).strip() + "\n\n")
                idx += 1
                cur_words = []
                cur_start = we
        # flush remainder
        if cur_words:
            final_end = min(end, cur_start + 1.2)
            f.write(f"{idx}\n{seconds_to_tc(cur_start - start)} --> {seconds_to_tc(final_end - start)}\n")
            f.write(" ".join(cur_words).strip() + "\n\n")

def _collect_words(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    allw: List[Dict[str, Any]] = []
    for s in transcript.get("segments", []):
        for w in s.get("words", []):
            if "start" in w and "end" in w and "word" in w:
                allw.append({"start": float(w["start"]), "end": float(w["end"]), "word": w["word"]})
    return allw

def _detect_center_crop(input_path: str, target_ratio: float = H / W) -> str:
    """Return ffmpeg scale/crop filter string.
    Strategy: keep width, scale to height, then center-crop or pad to 1080x1920."""
    # quickly probe frame size using OpenCV
    cap = cv2.VideoCapture(input_path)
    ok, _ = cap.read()
    if not ok:
        return f"scale=-2:{H},pad={W}:{H}:(ow-iw)/2:(oh-ih)/2"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    src_ratio = height / max(width, 1)
    # If already vertical-ish, just scale and pad to 1080x1920
    if src_ratio >= 1.2:
        return f"scale=-2:{H},pad={W}:{H}:(ow-iw)/2:(oh-ih)/2"
    # If landscape, scale by height and center-crop width to 1080
    return f"scale=-2:{H},crop={W}:{H}:(iw-{W})/2:(ih-{H})/2"

def render_short(
    input_path: str,
    transcript: Dict[str, Any],
    start: float,
    end: float,
    out_mp4: str,
    out_srt: str
):
    ensure_dir(os.path.dirname(out_mp4))
    ensure_dir(os.path.dirname(out_srt))

    # Write SRT from word-level timings within [start,end]
    words = _collect_words(transcript)
    _write_srt(words, start, end, out_srt)

    vf = _detect_center_crop(input_path)
    # Burn captions with safe bottom margin (MarginV)
    sub_filter = f"subtitles='{out_srt}':force_style='Fontsize=48,Outline=2,Shadow=1,MarginV=80'"

    filter_chain = f"{vf},{sub_filter},fps={FPS}"
    ffmpeg = which_ffmpeg()
    cmd = (
        f'{ffmpeg} -y -ss {start:.3f} -to {end:.3f} -i "{input_path}" '
        f'-vf "{filter_chain}" -an -c:v libx264 -pix_fmt yuv420p -r {FPS} -b:v 5M -movflags +faststart -preset veryfast -t {end-start:.3f} "{out_mp4}.tmp.mp4"'
    )
    # First pass video only for accurate duration; then merge original audio segment
    subprocess.run(cmd, shell=True, check=True)

    # Extract audio segment
    cmd_a = f'{ffmpeg} -y -ss {start:.3f} -to {end:.3f} -i "{input_path}" -c:a aac -b:a 160k "{out_mp4}.tmp.aac"'
    subprocess.run(cmd_a, shell=True, check=True)

    # Merge
    cmd_m = f'{ffmpeg} -y -i "{out_mp4}.tmp.mp4" -i "{out_mp4}.tmp.aac" -c:v copy -c:a copy "{out_mp4}"'
    subprocess.run(cmd_m, shell=True, check=True)

    # Cleanup
    for ext in (".tmp.mp4", ".tmp.aac"):
        p = f"{out_mp4}{ext}"
        if os.path.exists(p):
            os.remove(p)

    LOGGER.info("Rendered %s", out_mp4)
