"""
Utility to ensure ONNX model files are present on disk.
Called at import time by FaceDetector and FaceRecognizer so models are
available on Railway (where .onnx files are gitignored and not deployed).
"""
from pathlib import Path
from utils.logger import logger

_MODELS_DOWNLOADED = False


def ensure_models_present() -> None:
    """Download ONNX models if they are missing (e.g. first boot on Railway)."""
    global _MODELS_DOWNLOADED
    if _MODELS_DOWNLOADED:
        return
    _MODELS_DOWNLOADED = True

    try:
        import importlib.util
        script = Path(__file__).resolve().parents[2] / "download_models.py"
        if not script.is_file():
            return
        spec = importlib.util.spec_from_file_location("download_models", script)
        if spec and spec.loader:
            import importlib
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[arg-type]
            if hasattr(mod, "MODELS") and hasattr(mod, "download"):
                for name, url in mod.MODELS.items():
                    mod.download(name, url)
    except Exception as exc:
        logger.warning("Model download helper failed (non-fatal): %s", exc)
