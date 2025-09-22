#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py — general helpers (ffprobe, paths, timecode, logging setup)
"""
from __future__ import annotations
import json
import logging
import math
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List
import tempfile
from urllib.parse import urlparse
from tenacity import retry, stop_after_attempt, wait_fixed

LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format=LOG_FORMAT)

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def run(cmd: str) -> subprocess.CompletedProcess:
    logging.getLogger("utils.run").info(cmd)
    return subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def which_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg.")
    return path

def which_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise RuntimeError("ffprobe not found in PATH. Please install ffmpeg/ffprobe.")
    return path

def ffprobe_duration(path: str) -> float:
    """Return duration (seconds) using ffprobe."""
    ffprobe = which_ffprobe()
    cmd = f'{ffprobe} -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{path}"'
    cp = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return float(cp.stdout.decode().strip())

def seconds_to_tc(sec: float) -> str:
    sec = max(sec, 0.0)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - math.floor(sec)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@dataclass
class Segment:
    start: float
    end: float
    text: str

def _is_http_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https")
    except Exception:
        return False

@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def download_from_url(url: str, outdir: str = "/workspace/downloads", cookies_path: str | None = None) -> str:
    """
    Download a remote video (YouTube, etc.) to a local MP4 using yt-dlp.
    Returns the absolute file path. Retries are automatic.
    """
    if not _is_http_url(url):
        raise ValueError(f"Not a valid http(s) URL: {url}")

    os.makedirs(outdir, exist_ok=True)

    # late import so the rest of the tool works even if yt-dlp isn't installed
    from yt_dlp import YoutubeDL

    ydl_opts = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(outdir, "%(id)s.%(ext)s"),
        "noprogress": True,
        "quiet": True,
        "retries": 5,
        "concurrent_fragment_downloads": 8,
    }
    if cookies_path and os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = cookies_path

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        # Preferred: filepath provided by yt-dlp
        fp = None
        if isinstance(info, dict):
            # playlist vs single video
            if "requested_downloads" in info and info["requested_downloads"]:
                fp = info["requested_downloads"][0].get("filepath")
            if not fp and "entries" in info and info["entries"]:
                e = info["entries"][0]
                if "requested_downloads" in e and e["requested_downloads"]:
                    fp = e["requested_downloads"][0].get("filepath")
            if not fp:
                # fallback: derive from template
                try:
                    fp = ydl.prepare_filename(info)
                except Exception:
                    pass

        if not fp:
            raise RuntimeError("yt-dlp did not return a filepath. Enable logs to diagnose.")

        # Ensure .mp4 extension (yt-dlp should already do this)
        base, ext = os.path.splitext(fp)
        if ext.lower() != ".mp4":
            new_fp = base + ".mp4"
            if os.path.exists(fp):
                shutil.move(fp, new_fp)
            fp = new_fp

        return os.path.abspath(fp)
