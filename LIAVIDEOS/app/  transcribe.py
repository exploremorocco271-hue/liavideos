#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transcribe.py — WhisperX primary + Whisper fallback. Produces:
- transcript.json: { "language": str, "segments": [ {start,end,text,words:[{start,end,word}]} ] }
"""
from __future__ import annotations
import logging
import os
from typing import Dict, Any, List

import torch
from utils import ensure_dir, save_json

LOGGER = logging.getLogger("transcribe")

def _normalize_whisper_segments(segments) -> List[Dict[str, Any]]:
    norm = []
    for seg in segments:
        words = []
        # openai-whisper may return "words" if word_timestamps=True; else approximate
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                if w.get("word") is None:
                    continue
                words.append({
                    "start": float(w.get("start", seg["start"])),
                    "end": float(w.get("end", seg["end"])),
                    "word": w.get("word").strip()
                })
        else:
            words.append({"start": float(seg["start"]), "end": float(seg["end"]), "word": seg["text"].strip()})
        norm.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"].strip(),
            "words": words
        })
    return norm

def transcribe_whisperx(input_path: str, language: str | None, device: str) -> Dict[str, Any]:
    import whisperx  # type: ignore
    audio = whisperx.load_audio(input_path)

    model = whisperx.load_model("large-v2", device)
    result = model.transcribe(audio, language=language)

    # word-level alignment
    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)
    segments = []
    for s in result_aligned["segments"]:
        words = []
        for w in s.get("words", []):
            if w.get("word") is None: 
                continue
            words.append({"start": float(w["start"]), "end": float(w["end"]), "word": w["word"]})
        segments.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": s.get("text", "").strip(),
            "words": words
        })
    return {"language": result["language"], "segments": segments}

def transcribe_fallback_whisper(input_path: str, language: str | None, device: str) -> Dict[str, Any]:
    import whisper  # type: ignore
    model_name = "tiny" if device == "cpu" else "base"
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(input_path, language=language, word_timestamps=True, verbose=False)
    segments = _normalize_whisper_segments(result.get("segments", []))
    lang = result.get("language", language or "en")
    return {"language": lang, "segments": segments}

def transcribe(input_path: str, language: str | None, outdir: str, force_cpu: bool=False) -> Dict[str, Any]:
    ensure_dir(outdir)
    device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    LOGGER.info("Selected device: %s", device)
    try:
        if device == "cuda":
            LOGGER.info("Running WhisperX on GPU…")
            data = transcribe_whisperx(input_path, language, device)
        else:
            raise RuntimeError("Forcing fallback on CPU")
    except Exception as e:
        LOGGER.warning("WhisperX failed or unavailable (%s). Falling back to openai-whisper.", e)
        data = transcribe_fallback_whisper(input_path, language, "cpu")

    save_path = os.path.join(outdir, "transcript.json")
    save_json(data, save_path)
    LOGGER.info("Saved transcript to %s", save_path)
    return data
