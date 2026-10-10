from pathlib import Path
from typing import List, Tuple
import numpy as np
from utils.config import settings
from utils.logger import logger
from ai_engine.face_recognition.model_loader import ensure_models_present


FaceBox = Tuple[int, int, int, int]


class FaceDetector:
    """
    Detects faces in live camera frames using complementary OpenCV Haar Cascades.

    Frames from window captures often have lower contrast than direct webcams, so
    detection is performed on a contrast-normalised, resized copy. Coordinates are
    then mapped back to the original OBS frame for drawing and recognition.
    """
    def __init__(self):
        self.cascades = []
        self.yunet = None
        self.backend_name = "unavailable"
        # Download models if they were not committed to the repo (e.g. Railway).
        ensure_models_present()
        try:
            import cv2
            model_path = Path(settings.FACE_DETECTION_MODEL_PATH)
            if not model_path.is_absolute():
                model_path = Path(__file__).resolve().parents[2] / model_path

            if model_path.is_file() and hasattr(cv2, "FaceDetectorYN_create"):
                self.yunet = cv2.FaceDetectorYN_create(
                    str(model_path),
                    "",
                    (320, 320),
                    settings.FACE_DETECTION_CONFIDENCE,
                    0.3,
                    5000,
                )
                self.backend_name = "OpenCV YuNet"
                logger.info("Face detector initialised with OpenCV YuNet.")

            for filename in (
                "haarcascade_frontalface_default.xml",
                "haarcascade_frontalface_alt2.xml",
            ):
                cascade = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
                if not cascade.empty():
                    self.cascades.append(cascade)

            if self.backend_name == "unavailable" and self.cascades:
                self.backend_name = "OpenCV Haar Cascade"
                logger.warning(
                    "YuNet model was not found at %s; using the less reliable Haar Cascade fallback.",
                    model_path,
                )
            elif not self.cascades and self.yunet is None:
                logger.warning("No OpenCV face cascade classifier could be loaded.")
        except Exception as e:
            logger.warning(f"Could not load Haar Cascade classifier: {e}")

    @staticmethod
    def _intersection_over_union(first: FaceBox, second: FaceBox) -> float:
        first_x, first_y, first_w, first_h = first
        second_x, second_y, second_w, second_h = second
        right = min(first_x + first_w, second_x + second_w)
        bottom = min(first_y + first_h, second_y + second_h)
        overlap_w = max(0, right - max(first_x, second_x))
        overlap_h = max(0, bottom - max(first_y, second_y))
        intersection = overlap_w * overlap_h
        union = first_w * first_h + second_w * second_h - intersection
        return intersection / union if union else 0.0

    def _merge_overlaps(self, boxes: List[FaceBox]) -> List[FaceBox]:
        """Keep the strongest (largest) candidate when cascades find the same face."""
        selected: List[FaceBox] = []
        for box in sorted(boxes, key=lambda candidate: candidate[2] * candidate[3], reverse=True):
            if all(self._intersection_over_union(box, kept) < 0.35 for kept in selected):
                selected.append(box)
        return selected

    def detect_faces(
        self,
        frame: np.ndarray,
        min_confidence: float | None = None,
        min_neighbors: int | None = None,
        min_size: int | None = None,
        detection_width: int | None = None,
    ) -> List[FaceBox]:
        """Detects bounding boxes (x, y, w, h) of faces in the frame.

        All tuning knobs are optional overrides; by default the configured
        runtime settings are used. Enrollment passes more permissive values so
        profile captures are not rejected outright.
        """
        boxes, _ = self.detect_faces_with_landmarks(
            frame, min_confidence, min_neighbors, min_size, detection_width
        )
        return boxes

    def detect_faces_with_landmarks(
        self,
        frame: np.ndarray,
        min_confidence: float | None = None,
        min_neighbors: int | None = None,
        min_size: int | None = None,
        detection_width: int | None = None,
    ) -> tuple[List[FaceBox], List]:
        """Detects faces and returns (boxes, yunet_rows).

        *yunet_rows* contains the raw YuNet detection row for each box
        (bounding box + 5 landmark coords + confidence scaled back to the
        original frame).  Callers that only need boxes should use
        ``detect_faces``; callers that also need landmarks for SFace alignment
        should use this method.

        When YuNet is unavailable (Haar Cascade fallback) *yunet_rows* is an
        empty list of the same length as *boxes* (all ``None``).
        """
        if (self.yunet is None and not self.cascades) or frame is None or frame.size == 0:
            return [], []

        confidence = settings.FACE_DETECTION_CONFIDENCE if min_confidence is None else min_confidence
        neighbors = settings.FACE_DETECTION_MIN_NEIGHBORS if min_neighbors is None else min_neighbors
        min_face = settings.FACE_DETECTION_MIN_SIZE if min_size is None else min_size
        width = settings.FACE_DETECTION_WIDTH if detection_width is None else detection_width

        try:
            import cv2
            frame_height, frame_width = frame.shape[:2]
            detection_width_px = max(1, width)
            scale = min(1.0, detection_width_px / frame_width)
            if scale < 1.0:
                detection_frame = cv2.resize(
                    frame,
                    (round(frame_width * scale), round(frame_height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                detection_frame = frame

            if self.yunet is not None:
                detection_height, detection_frame_width = detection_frame.shape[:2]
                self.yunet.setInputSize((detection_frame_width, detection_height))
                _, faces = self.yunet.detect(detection_frame)
                if faces is not None:
                    inverse_scale = 1.0 / scale
                    yunet_boxes: List[FaceBox] = []
                    yunet_rows: List = []
                    for face in faces:
                        if float(face[-1]) < confidence:
                            continue
                        # Scale the entire row (bbox + 10 landmark coords) back
                        # to the original frame dimensions so alignCrop works on
                        # the unresized frame.
                        scaled_row = face.copy().astype(float)
                        # Columns 0-3: x, y, w, h; columns 4-13: 5 landmarks (x,y each)
                        scaled_row[:14] *= inverse_scale
                        yunet_boxes.append((
                            round(float(face[0]) * inverse_scale),
                            round(float(face[1]) * inverse_scale),
                            round(float(face[2]) * inverse_scale),
                            round(float(face[3]) * inverse_scale),
                        ))
                        yunet_rows.append(scaled_row)
                    if yunet_boxes:
                        # Merge overlapping boxes, keeping corresponding rows in sync.
                        selected_boxes, selected_rows = self._merge_overlaps_with_rows(
                            yunet_boxes, yunet_rows
                        )
                        return selected_boxes, selected_rows

            gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            min_size_px = max(16, round(min_face * scale))
            candidates: List[FaceBox] = []
            for cascade in self.cascades:
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.08,
                    minNeighbors=neighbors,
                    minSize=(min_size_px, min_size_px),
                )
                candidates.extend((int(x), int(y), int(w), int(h)) for x, y, w, h in faces)

            inverse_scale = 1.0 / scale
            original_boxes = [
                (
                    round(x * inverse_scale),
                    round(y * inverse_scale),
                    round(w * inverse_scale),
                    round(h * inverse_scale),
                )
                for x, y, w, h in candidates
            ]
            merged = self._merge_overlaps(original_boxes)
            # No landmark data from Haar Cascades — return None rows.
            return merged, [None] * len(merged)
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return [], []

    def _merge_overlaps_with_rows(
        self, boxes: List[FaceBox], rows: List
    ) -> tuple[List[FaceBox], List]:
        """Like _merge_overlaps but keeps YuNet rows in sync with the kept boxes."""
        selected_boxes: List[FaceBox] = []
        selected_rows: List = []
        paired = sorted(
            zip(boxes, rows),
            key=lambda pair: pair[0][2] * pair[0][3],
            reverse=True,
        )
        for box, row in paired:
            if all(self._intersection_over_union(box, kept) < 0.35 for kept in selected_boxes):
                selected_boxes.append(box)
                selected_rows.append(row)
        return selected_boxes, selected_rows


    def draw_face_boxes(self, frame: np.ndarray, faces: List[FaceBox], labels: List[str] | None = None) -> np.ndarray:
        """Return a copy of a BGR frame with clear, dashboard-ready face boxes."""
        if frame is None or not faces:
            return frame

        try:
            import cv2
            annotated = frame.copy()
            for index, (x, y, width, height) in enumerate(faces):
                x = max(0, x)
                y = max(0, y)
                right = min(annotated.shape[1] - 1, x + width)
                bottom = min(annotated.shape[0] - 1, y + height)
                if right <= x or bottom <= y:
                    continue

                label = labels[index] if labels and index < len(labels) else "UNREGISTERED"
                is_enrolled = (
                    bool(label)
                    and label.upper() not in ("UNKNOWN", "NOT ENROLLED", "TRESPASSER", "FACE", "UNREGISTERED")
                    and not label.upper().startswith("UNKNOWN")
                    and not label.upper().startswith("NOT ENROLLED")
                    and not label.upper().startswith("TRESPASSER")
                    and not label.upper().startswith("UNREGISTERED")
                )

                if is_enrolled:
                    # Enrolled student: Emerald Green
                    box_color = (26, 218, 145)  # BGR
                    text_color = (6, 40, 24)     # Dark green text
                    display_label = label
                else:
                    # Unknown or not enrolled: Red (#ef4444 in BGR)
                    box_color = (68, 68, 239)    # BGR Red
                    text_color = (255, 255, 255) # White text
                    display_label = "UNREGISTERED"

                cv2.rectangle(annotated, (x, y), (right, bottom), box_color, 2, cv2.LINE_AA)
                (label_width, label_height), baseline = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                label_top = max(0, y - label_height - baseline - 8)
                cv2.rectangle(
                    annotated,
                    (x, label_top),
                    (x + label_width + 12, y),
                    box_color,
                    thickness=-1,
                )
                cv2.putText(
                    annotated,
                    display_label,
                    (x + 6, y - baseline - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )
            return annotated
        except Exception as error:
            logger.error(f"Error drawing face boxes: {error}")
            return frame

face_detector = FaceDetector()
