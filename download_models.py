#!/usr/bin/env python3
"""
Download the ONNX model files required by the AI engine at build/startup time.
Called from nixpacks.toml install phase so models are present when the app starts.
"""
import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "ai_engine" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx"
    ),
}


def download(name: str, url: str) -> None:
    dest = MODELS_DIR / name
    if dest.exists():
        print(f"[models] {name} already exists — skipping.")
        return
    print(f"[models] Downloading {name} ...", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[models] {name} downloaded ({dest.stat().st_size:,} bytes).", flush=True)
    except Exception as exc:
        print(f"[models] WARNING: Could not download {name}: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    for model_name, model_url in MODELS.items():
        download(model_name, model_url)
    print("[models] Done.", flush=True)
