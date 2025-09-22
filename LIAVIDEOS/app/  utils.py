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
