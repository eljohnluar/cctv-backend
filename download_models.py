#!/usr/bin/env python3
"""
Download the ONNX and YOLO model files required by the AI engine at build/startup time.
Called during deployment build phase and at runtime fallback.
"""
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "ai_engine" / "models"
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

ROOT_MODELS = {
    "yolov8n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
    ),
}


def download(name: str, url: str, target_dir: Optional[Path] = None) -> None:
    dest_dir = target_dir or MODELS_DIR
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"[models] {name} already exists ({dest.stat().st_size:,} bytes) — skipping.")
        return
    print(f"[models] Downloading {name} to {dest_dir} ...", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[models] {name} downloaded successfully ({dest.stat().st_size:,} bytes).", flush=True)
    except Exception as exc:
        print(f"[models] WARNING: Could not download {name}: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    for model_name, model_url in MODELS.items():
        download(model_name, model_url, MODELS_DIR)
    for model_name, model_url in ROOT_MODELS.items():
        download(model_name, model_url, BACKEND_DIR)
    print("[models] Model verification completed.", flush=True)
