#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_shorts.py — CLI orchestrator
Usage:
  python render_shorts.py --input /path/video.mp4 [--language en] [--max_shorts 6] [--max_duration 60] [--min_gap 30] [--outdir ./outputs] [--force_cpu]
"""
from __future__ import annotations
import argparse
import json
import logging
import os
from typing import Dict, Any, List

import pandas as pd

from utils import setup_logging, ensure_dir, save_json, load_json, ffprobe_duration
from transcribe import transcribe
from features import load_audio_rms, export_rms_csv, score_candidates, build_sentence_list
from ranker import finalize
from renderer import render_short
from titles import pick_title_and_description

def parse_args():
    ap = argparse.ArgumentParser(description="Auto-generate vertical shorts from a long video.")
    ap.add_argument("--input", required=True, help="Input video path")
    ap.add_argument("--language", default=None, help="Override language (e.g., en)")
    ap.add_argument("--max_shorts", type=int, default=6)
    ap.add_argument("--max_duration", type=float, default=60.0)
    ap.add_argument("--min_gap", type=float, default=30.0)
    ap.add_argument("--outdir", default="./outputs")
    ap.add_argument("--force_cpu", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("render_shorts")

    in_path = args.input
    outdir = ensure_dir(args.outdir)
    shorts_dir = ensure_dir(os.path.join(outdir, "shorts"))
    caps_dir = ensure_dir(os.path.join(outdir, "captions"))

    # 1) Detect duration
    duration = ffprobe_duration(in_path)
    logger.info("Input duration: %.2f sec", duration)

    # 2) Transcribe
    transcript = transcribe(in_path, args.language, outdir, force_cpu=args.force_cpu)
    # also store transcript again (already saved by transcribe.py)
    transcript_path = os.path.join(outdir, "transcript.json")

    # 2.5) Extract audio RMS (ffmpeg to wav in temp)
    wav_path = os.path.join(outdir, "audio.wav")
    os.system(f'ffmpeg -y -i "{in_path}" -vn -ac 1 -ar 16000 "{wav_path}" > /dev/null 2>&1')
    rms_df = load_audio_rms(wav_path, frame_sec=0.5)
    export_rms_csv(rms_df, outdir)

    # 3) Candidate windows + scores
    candidates = score_candidates(transcript, rms_df, duration, win_min=10.0, win_max=min(60.0, args.max_duration), stride=5.0)
    # 4) Rank/select without overlap (≥ min_gap)
    sentences = build_sentence_list(transcript)
    selected = finalize(candidates, sentences, max_shorts=args.max_shorts, min_gap=args.min_gap, max_duration=args.max_duration)

    # 5/6/7) Render, SRT, titles, manifest
    manifest_rows: List[Dict[str, Any]] = []
    for i, s in enumerate(selected, start=1):
        short_id = f"short_{i:03}"
        out_mp4 = os.path.join(shorts_dir, f"{short_id}.mp4")
        out_srt = os.path.join(caps_dir, f"{short_id}.srt")

        render_short(in_path, transcript, s["start"], s["end"], out_mp4, out_srt)
        title, description = pick_title_and_description(s["text"], transcript.get("language","en"))
        row = {
            "short_id": short_id,
            "start_sec": round(s["start"], 3),
            "end_sec": round(s["end"], 3),
            "duration": round(s["end"] - s["start"], 3),
            "title": title,
            "description": description,
            "score": round(float(s["score"]), 4),
            "source_path": os.path.abspath(in_path),
            "output_path": os.path.abspath(out_mp4),
            "language": transcript.get("language","")
        }
        manifest_rows.append(row)

    # Save manifests
    manifest_json = os.path.join(outdir, "manifest.json")
    manifest_csv = os.path.join(outdir, "manifest.csv")
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, ensure_ascii=False, indent=2)
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)

    print(json.dumps({"status": "ok", "count": len(manifest_rows), "manifest": manifest_json}, indent=2))

if __name__ == "__main__":
    main()
