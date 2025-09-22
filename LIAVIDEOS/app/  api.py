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
from utils import download_from_url, _is_http_url


from fastapi import FastAPI, HTTPException, Request
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
async def run_task(req: Request):
    body = await req.json()
    url = body.get("url") or body.get("input_url")
    input_path = body.get("input")
    cookies_path = body.get("cookies_path")
    language = body.get("language", "en")
    max_shorts = int(body.get("max_shorts", 6))
    max_duration = int(body.get("max_duration", 60))
    min_gap = int(body.get("min_gap", 30))
    outdir = body.get("outdir", "/workspace/outputs")
    force_cpu = bool(body.get("force_cpu", False))

    # Resolve to a local file
    if url and _is_http_url(url) and not input_path:
        dl_dir = "/workspace/downloads" if os.path.isdir("/workspace") else "./downloads"
        input_path = download_from_url(url, outdir=dl_dir, cookies_path=cookies_path)

    if not input_path:
        return {"error": "Provide 'url' or 'input'."}

    # Call the CLI as before (or import and call main())
    cmd = [
        "python3", "render_shorts.py",
        "--input", input_path,
        "--language", language,
        "--max_shorts", str(max_shorts),
        "--max_duration", str(max_duration),
        "--min_gap", str(min_gap),
        "--outdir", outdir,
    ]
    if force_cpu:
        cmd.append("--force_cpu")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": "pipeline failed", "stderr": proc.stderr}

    # Return manifest if available
    manifest_json = os.path.join(outdir, "manifest.json")
    if os.path.exists(manifest_json):
        with open(manifest_json, "r", encoding="utf-8") as f:
            return json.load(f)

    return {"status": "completed", "outdir": outdir}
