from pathlib import Path
import threading
from typing import Optional, Dict, Any, List
import numpy as np
from utils.config import settings
from utils.logger import logger
from ai_engine.face_recognition.model_loader import ensure_models_present


class FaceRecognizer:
    """
    Extracts 128-dimensional facial embeddings using DeepFace (FaceNet backbone)
    and matches against registered student embeddings stored in database.
    """
    def __init__(self):
        self.model_name = settings.RECOGNITION_MODEL
        self.threshold = settings.RECOGNITION_THRESHOLD
        self.sface = None
        # OpenCV's DNN Net keeps mutable input/output buffers. The live camera
        # and enrollment API can run simultaneously, so inference must be
        # serialized to prevent the buffer-shape assertion seen in OpenCV 5.
        self._sface_lock = threading.Lock()
        self.backend_name = "unavailable"
        try:
            import importlib.util
            if importlib.util.find_spec("deepface"):
                self.backend_name = "DeepFace"
        except Exception:
            pass

        # Download models if they were not committed to the repo (e.g. Railway).
        ensure_models_present()

        try:
            import cv2
            model_path = Path(settings.FACE_RECOGNITION_MODEL_PATH)
            if not model_path.is_absolute():
                model_path = Path(__file__).resolve().parents[2] / model_path
            if model_path.is_file() and hasattr(cv2, "FaceRecognizerSF_create"):
                self.sface = cv2.FaceRecognizerSF_create(str(model_path), "")
                if self.backend_name == "unavailable":
                    self.backend_name = "OpenCV SFace"
                logger.info("Face embedding backend available: %s", self.backend_name)
        except Exception as error:
            logger.warning("Could not load OpenCV SFace face embedding model: %s", error)

    def _extract_sface_embedding(
        self,
        face_image: np.ndarray,
        full_frame: np.ndarray | None = None,
        yunet_face_row: np.ndarray | None = None,
    ) -> Optional[List[float]]:
        """Extract a 128-d SFace embedding.

        When *full_frame* and *yunet_face_row* (the raw YuNet detection row
        containing bounding box + 5 landmark coordinates + confidence) are
        provided, the model's own ``alignCrop`` is used.  This is the path
        SFace was trained on and is significantly more accurate than a plain
        resize.  The plain-resize fallback is kept for callers that only have
        a cropped face patch (e.g. enrollment from an uploaded image).
        """
        if self.sface is None or face_image is None or face_image.size == 0:
            return None
        try:
            import cv2
            if full_frame is not None and yunet_face_row is not None:
                try:
                    aligned = self.sface.alignCrop(full_frame, yunet_face_row)
                    with self._sface_lock:
                        feature = self.sface.feature(aligned)
                    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
                    norm = np.linalg.norm(vector)
                    return (vector / norm).astype(float).tolist() if norm else None
                except Exception as align_err:
                    logger.debug("SFace alignCrop failed, falling back to resize: %s", align_err)

            # Fallback: plain resize — used for enrollment uploads without
            # YuNet landmark data.  Reject very small crops to avoid noisy
            # embeddings that could produce false matches.
            h, w = face_image.shape[:2]
            if h < 40 or w < 40:
                logger.debug("Face crop too small (%dx%d) for reliable embedding, skipping.", w, h)
                return None
            aligned_face = cv2.resize(face_image, (112, 112), interpolation=cv2.INTER_AREA)
            with self._sface_lock:
                feature = self.sface.feature(aligned_face)
            vector = np.asarray(feature, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(vector)
            return (vector / norm).astype(float).tolist() if norm else None
        except Exception as error:
            logger.warning("OpenCV SFace embedding failed: %s", error)
            return None

    def extract_embedding(
        self,
        face_image: np.ndarray,
        full_frame: np.ndarray | None = None,
        yunet_face_row: np.ndarray | None = None,
    ) -> Optional[List[float]]:
        """
        Generates embedding vector for a cropped face image.
        """
        if self.backend_name == "DeepFace":
            try:
                from deepface import DeepFace
                embeddings = DeepFace.represent(
                    img_path=face_image,
                    model_name=self.model_name,
                    enforce_detection=False,
                )
                if embeddings and len(embeddings) > 0:
                    return embeddings[0]["embedding"]
            except Exception as error:
                logger.warning("DeepFace embedding failed; trying OpenCV SFace: %s", error)

        return self._extract_sface_embedding(face_image, full_frame, yunet_face_row)

    def match_face(self, target_embedding: List[float], student_embeddings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Calculates cosine similarity against database embeddings.
        Returns closest match above confidence threshold.
        """
        if not target_embedding or not student_embeddings:
            return None

        target_vec = np.array(target_embedding)
        best_match = None
        best_score = -1.0

        for entry in student_embeddings:
            try:
                emb = np.array(entry["embedding"], dtype=float)
                if emb.shape != target_vec.shape:
                    continue
                target_norm = np.linalg.norm(target_vec)
                embedding_norm = np.linalg.norm(emb)
                if target_norm == 0 or embedding_norm == 0:
                    continue
                # Cosine similarity
                cosine_sim = np.dot(target_vec, emb) / (target_norm * embedding_norm)
            except (KeyError, TypeError, ValueError):
                continue
            if cosine_sim > best_score:
                best_score = float(cosine_sim)
                best_match = entry

        if best_match and best_score >= self.threshold:
            return {
                "student_id": best_match["student_id"],
                "confidence": best_score
            }

        return None

face_recognizer = FaceRecognizer()
