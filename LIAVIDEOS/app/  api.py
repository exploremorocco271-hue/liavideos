#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api.py — FastAPI wrapper exposing /healthz and /run
"""
from __future__ import annotations
import json
import os
import subprocess
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class RunRequest(BaseModel):
    input: str
    language: Optional[str] = None
    max_shorts: int = 6
    max_duration: float = 60.0
    min_gap: float = 30.0
    outdir: str = "./outputs"
    force_cpu: bool = False

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/run")
def run(req: RunRequest):
    if not os.path.exists(req.input):
        raise HTTPException(status_code=400, detail=f"Input not found: {req.input}")

    cmd = [
        "python", "/app/render_shorts.py",
        "--input", req.input,
        "--outdir", req.outdir,
        "--max_shorts", str(req.max_shorts),
        "--max_duration", str(req.max_duration),
        "--min_gap", str(req.min_gap)
    ]
    if req.language:
        cmd.extend(["--language", req.language])
    if req.force_cpu:
        cmd.append("--force_cpu")

    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # The CLI prints a small JSON summary; load & return it
        data = json.loads(cp.stdout.strip() or "{}")
        # also inline manifest content if handy
        manifest_path = data.get("manifest")
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                data["manifest_data"] = json.load(f)
        return data
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr or e.stdout or "render failed")
