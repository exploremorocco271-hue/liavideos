#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features.py — windowing, audio RMS, semantic/sentiment/linguistic scoring.
Outputs:
- audio_rms.csv
- list of candidate windows with scores
"""
from __future__ import annotations
import logging
import os
from typing import Dict, Any, List, Tuple

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util as st_util
from vadersentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils import Segment, ensure_dir

LOGGER = logging.getLogger("features")

def load_audio_rms(wav_path: str, frame_sec: float=0.5) -> pd.DataFrame:
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    hop_length = int(frame_sec * sr)
    frame_length = hop_length
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    t = np.arange(len(rms)) * frame_sec
    df = pd.DataFrame({"time": t, "rms": rms})
    return df

def export_rms_csv(df: pd.DataFrame, outdir: str) -> str:
    path = os.path.join(outdir, "audio_rms.csv")
    df.to_csv(path, index=False)
    LOGGER.info("Saved RMS to %s", path)
    return path

def segments_from_transcript(transcript: Dict[str, Any]) -> List[Segment]:
    segs = []
    for s in transcript.get("segments", []):
        segs.append(Segment(float(s["start"]), float(s["end"]), s.get("text", "")))
    return segs

def build_sentence_list(transcript: Dict[str, Any]) -> List[Tuple[float, float, str]]:
    """Return sentence-level list: [(start,end,text), ...]; tries to snap within original segments."""
    try:
        import spacy
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        use_spacy = True
    except Exception:
        use_spacy = False

    sentences = []
    for s in transcript.get("segments", []):
        text = s.get("text", "").strip()
        if not text:
            continue
        if use_spacy:
            doc = nlp(text)
            # naive subdivision: proportionally split the time across sentences
            total = len(text)
            cur_start = float(s["start"])
            cur_end = float(s["end"])
            dur = max(cur_end - cur_start, 1e-3)
            offset = 0
            for sent in doc.sents:
                frac = max(len(sent.text), 1) / total
                sent_dur = dur * frac
                sentences.append((cur_start + offset, min(cur_start + offset + sent_dur, cur_end), sent.text.strip()))
                offset += sent_dur
        else:
            # regex fallback on punctuation
            import re
            parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', text) if p.strip()]
            total = len(text)
            cur_start = float(s["start"])
            cur_end = float(s["end"])
            dur = max(cur_end - cur_start, 1e-3)
            acc = 0
            for p in parts:
                frac = max(len(p), 1) / total
                sent_dur = dur * frac
                sentences.append((cur_start + acc, min(cur_start + acc + sent_dur, cur_end), p))
                acc += sent_dur
    return sentences

def sliding_windows(duration: float, win_min: float=10.0, win_max: float=60.0, stride: float=5.0):
    t = 0.0
    while t + win_min <= duration:
        yield (t, min(t + win_max, duration))
        t += stride

def find_window_text(sentences: List[Tuple[float,float,str]], start: float, end: float) -> str:
    parts = [txt for (s,e,txt) in sentences if not (e <= start or s >= end)]
    return " ".join(parts).strip()

def score_candidates(
    transcript: Dict[str, Any],
    audio_rms_df: pd.DataFrame,
    duration: float,
    win_min: float=10.0,
    win_max: float=60.0,
    stride: float=5.0
) -> List[Dict[str, Any]]:
    """Score each sliding window using:
       - semantic punchiness (embedding norm + intra-similarity peaks)
       - sentiment magnitude (|compound|)
       - audio loudness peaks (mean normalized RMS)
       - linguistic cues (questions, imperatives, hook phrases)
    """
    sentences = build_sentence_list(transcript)
    texts = [s[2] for s in sentences] or [seg.get("text","") for seg in transcript.get("segments",[])]
    if not texts:
        texts = [""]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sent_emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    analyzer = SentimentIntensityAnalyzer()

    rms_norm = (audio_rms_df["rms"] - audio_rms_df["rms"].min()) / (audio_rms_df["rms"].ptp() + 1e-6)

    HOOK_WORDS = {"amazing","secret","mistake","watch","don’t","stop","before","until","hack","tip","trick","best","worst","why","how","what","avoid","never","always"}

    candidates = []
    for (ws, we) in tqdm(list(sliding_windows(duration, win_min, win_max, stride)), desc="Scoring windows"):
        text = find_window_text(sentences, ws, we)
        if len(text) < 20:
            continue

        # semantic "punchiness": reward variance & peaks of similarity across the window
        # (simple proxy: average cosine sim to the previous sentence; high change ⇒ interesting)
        idxs = [i for i,(s,e,_) in enumerate(sentences) if not (e <= ws or s >= we)]
        if idxs:
            sims = []
            for a,b in zip(idxs, idxs[1:]):
                sims.append(float(st_util.cos_sim(sent_emb[a], sent_emb[b]).cpu()))
            sem_score = float(np.clip(1.0 - (np.mean(sims) if sims else 0.5), 0, 1))  # lower sim ⇒ higher punch
        else:
            sem_score = 0.5

        # sentiment magnitude (peaks of emotion)
        sent_mag = float(np.mean([abs(analyzer.polarity_scores(sentences[i][2])["compound"]) for i in idxs]) if idxs else 0.0)

        # RMS loudness inside window
        mask = (audio_rms_df["time"] >= ws) & (audio_rms_df["time"] <= we)
        loud = float(audio_rms_df.loc[mask, "rms"].mean() if mask.any() else 0.0)
        loud_norm = float(((loud - audio_rms_df["rms"].min()) / (audio_rms_df["rms"].ptp() + 1e-6)))

        # linguistic hooks
        lw = sum(1 for w in HOOK_WORDS if f" {w} " in f" {text.lower()} ")
        ques = 1 if "?" in text else 0
        ling_score = min(1.0, 0.2*lw + 0.3*ques)

        # combined score (weights tuned empirically)
        score = 0.4*sem_score + 0.25*sent_mag + 0.25*loud_norm + 0.10*ling_score

        candidates.append({
            "start": float(ws),
            "end": float(we),
            "text": text,
            "score": float(score),
            "features": {
                "semantic": sem_score, "sentiment": sent_mag, "loudness": loud_norm, "ling": ling_score
            }
        })
    return candidates
