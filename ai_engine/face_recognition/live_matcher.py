import time
import threading
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np

from database.queries import create_alert_record, get_face_embeddings, mark_attendance, mark_time_out
from websocket_manager import publish_from_worker
from utils.logger import logger
from utils.uniform_policy import get_runtime_controls, get_uniform_policy
from ai_engine.security.compliance import compliance_checker
from .recognizer import face_recognizer
from ai_engine.voice.announcer import voice_announcer


FaceBox = Tuple[int, int, int, int]

# Minimum seconds between trespasser announcements per unknown face position.
# A per-session set prevents re-announcing after each detection frame.
_UNKNOWN_ANNOUNCE_COOLDOWN = 30.0


class LiveFaceMatcher:
    """Matches detected face crops and confirms attendance across consecutive scans."""

    def __init__(self):
        self._embeddings: List[Dict] = []
        self._embeddings_loaded_at = 0.0
        self._confirmations: Dict[int, int] = defaultdict(int)
        self._marked_today = set()
        self._time_out_today = set()
        self._marked_date = date.today()
        # More than one MJPEG client can invoke matching concurrently. Guard
        # confirmations and marking so one recognition emits one event only.
        self._state_lock = threading.Lock()

        # Track last time a trespasser announcement fired (global cooldown)
        self._last_unknown_announce: float = 0.0

    def reset_marked(self) -> None:
        """Clear today's confirmed markings and detection counts."""
        with self._state_lock:
            self._marked_today.clear()
            self._time_out_today.clear()
            self._confirmations.clear()
            self._last_unknown_announce = 0.0
        logger.info("Live face matcher marked attendance cache reset.")

    def refresh_embeddings(self) -> None:
        """Reload enrollment metadata on the next camera match."""
        with self._state_lock:
            self._embeddings_loaded_at = 0.0

    def _load_embeddings(self) -> List[Dict]:
        now = time.monotonic()
        if now - self._embeddings_loaded_at < 15:
            return self._embeddings
        try:
            self._embeddings = get_face_embeddings()
            self._embeddings_loaded_at = now
            logger.info("Loaded %d enrolled face embedding(s) for live matching.", len(self._embeddings))
        except Exception as error:
            logger.warning("Could not load enrolled face embeddings: %s", error)
            self._embeddings = []
            self._embeddings_loaded_at = now
        return self._embeddings

    @staticmethod
    def _crop(frame: np.ndarray, box: FaceBox) -> Optional[np.ndarray]:
        x, y, width, height = box
        frame_height, frame_width = frame.shape[:2]
        padding_x = round(width * 0.18)
        padding_y = round(height * 0.22)
        left = max(0, x - padding_x)
        top = max(0, y - padding_y)
        right = min(frame_width, x + width + padding_x)
        bottom = min(frame_height, y + height + padding_y)
        crop = frame[top:bottom, left:right]
        return crop if crop.size else None

    def match_frame(
        self,
        frame: np.ndarray,
        boxes: List[FaceBox],
        record_attendance: bool = True,
        hand_gesture_detected: bool = False,
        yunet_rows: List | None = None,
    ) -> List[str]:
        with self._state_lock:
            return self._match_frame(
                frame, boxes, record_attendance, hand_gesture_detected, yunet_rows or []
            )

    def _match_frame(
        self,
        frame: np.ndarray,
        boxes: List[FaceBox],
        record_attendance: bool = True,
        hand_gesture_detected: bool = False,
        yunet_rows: List | None = None,
    ) -> List[str]:
        """Return labels aligned with boxes and mark confirmed identities present today."""
        if frame is None or not boxes:
            return []

        today = date.today()
        if today != self._marked_date:
            self._marked_date = today
            self._marked_today.clear()
            self._time_out_today.clear()
            self._confirmations.clear()

        enrolled = self._load_embeddings()

        labels: List[str] = []
        seen_students: set = set()
        unknown_detected = False
        matched_confidences: Dict[int, float] = {}
        matched_face_boxes: Dict[int, FaceBox] = {}
        matched_gesture_requirements: Dict[int, bool] = {}

        rows = yunet_rows or [None] * len(boxes)
        for box, yunet_row in zip(boxes, rows):
            crop = self._crop(frame, box)
            embedding = (
                face_recognizer.extract_embedding(crop, full_frame=frame, yunet_face_row=yunet_row)
                if crop is not None
                else None
            )

            if not enrolled:
                # No enrolled faces in the system
                labels.append("UNREGISTERED")
                unknown_detected = True
                continue

            match = face_recognizer.match_face(embedding, enrolled) if embedding else None

            if not match:
                labels.append("UNREGISTERED")
                unknown_detected = True
                continue

            student_id = int(match["student_id"])
            student = next((item for item in enrolled if item.get("student_id") == student_id), {})
            name = student.get("student_name") or "MATCH"
            confidence = float(match.get("confidence", 0.0))
            requires_gesture = bool(student.get("gesture_enrolled", False))
            gesture_prompt = " · SHOW PALM" if requires_gesture and not hand_gesture_detected else ""
            labels.append(f"{name} {confidence * 100:.0f}%{gesture_prompt}")
            seen_students.add(student_id)
            matched_confidences[student_id] = max(matched_confidences.get(student_id, 0.0), confidence)
            matched_face_boxes[student_id] = box
            matched_gesture_requirements[student_id] = requires_gesture

        # --- Trespasser announcement (rate-limited) ---
        if unknown_detected and get_runtime_controls()["alert_mode_enabled"]:
            now = time.monotonic()
            if now - self._last_unknown_announce >= _UNKNOWN_ANNOUNCE_COOLDOWN:
                self._last_unknown_announce = now
                voice_announcer.announce_trespasser()

        if not record_attendance:
            self._confirmations.clear()
            return labels

        # --- Attendance confirmation ---
        for student_id in list(self._confirmations):
            if student_id not in seen_students:
                self._confirmations[student_id] = max(0, self._confirmations[student_id] - 1)

        for student_id in seen_students:
            self._confirmations[student_id] += 1
            if self._confirmations[student_id] >= 2:
                self._check_uniform_policy(student_id, enrolled, frame, matched_face_boxes.get(student_id))
            # Require 2 consecutive detections (~2 s) before marking to ensure stable match
            if self._confirmations[student_id] < 2:
                continue
            if matched_gesture_requirements.get(student_id, False) and not hand_gesture_detected:
                continue
            # Already checked in today: a later sighting records the departure.
            if student_id in self._marked_today:
                self._record_time_out(student_id, enrolled)
                continue

            try:
                record = mark_attendance(
                    student_id=student_id,
                    confidence=matched_confidences.get(student_id),
                )
                status = record["status"]
                self._marked_today.add(student_id)

                student_name = next(
                    (item.get("student_name") for item in enrolled if item.get("student_id") == student_id),
                    "Student",
                )
                student_code = next(
                    (item.get("student_code") for item in enrolled if item.get("student_id") == student_id),
                    "",
                )
                student_section = next(
                    (item.get("section") for item in enrolled if item.get("student_id") == student_id),
                    "",
                )

                logger.info("Attendance marked for student %s (%s) with status '%s'.", student_name, student_id, status)

                # --- Voice announcement: attendance marked ---
                voice_announcer.announce_attendance_marked(student_name)

                publish_from_worker({
                    "type": "attendance",
                    "student_id": student_id,
                    "student_name": student_name,
                    "student_code": student_code,
                    "section": student_section,
                    "status": status,
                    "confidence": matched_confidences.get(student_id),
                    "check_in_time": record.get("check_in_time"),
                    "class_date": record.get("class_date"),
                    "enrollment_photo_url": f"/api/students/{student_id}/enrollment-photo",
                    "record": {
                        **record,
                        "student_id": student_id,
                        "student_name": student_name,
                        "student_code": student_code,
                        "section": student_section,
                    },
                })
            except Exception as error:
                logger.warning("Could not mark attendance for student %s: %s", student_id, error)

        return labels

    def _record_time_out(self, student_id: int, enrolled: List[Dict]) -> None:
        """
        A confirmed sighting after the student is already checked in records
        their departure. Only a sighting past the configured Time out is stored,
        so walking past the camera during the day cannot set a wrong sign-out.
        """
        if student_id in self._time_out_today:
            return
        try:
            record = mark_time_out(student_id)
        except Exception as error:
            logger.warning("Could not record time out for student %s: %s", student_id, error)
            return
        if not record:
            return

        self._time_out_today.add(student_id)
        student = next((item for item in enrolled if item.get("student_id") == student_id), {})
        student_name = student.get("student_name") or "Student"
        logger.info(
            "Time out recorded for %s (%s) at %s.",
            student_name, student_id, record.get("check_out_time"),
        )
        voice_announcer.announce_time_out(student_name)
        publish_from_worker({
            "type": "attendance_time_out",
            "student_id": student_id,
            "student_name": student_name,
            "student_code": student.get("student_code", ""),
            "section": student.get("section", ""),
            "status": "time_out",
            "check_out_time": record.get("check_out_time"),
            "class_date": record.get("class_date"),
            "enrollment_photo_url": f"/api/students/{student_id}/enrollment-photo",
        })

    @staticmethod
    def _check_uniform_policy(
        student_id: int,
        enrolled: List[Dict],
        frame: np.ndarray,
        face_box: Optional[FaceBox],
    ) -> None:
        """Create and announce a clothing-policy violation for enrolled students only."""
        if face_box is None:
            return
        if not get_runtime_controls()["alert_mode_enabled"]:
            return
        allowed_colors = get_uniform_policy()["uniform_colors"]
        violation = compliance_checker.evaluate_uniform(student_id, frame, face_box, allowed_colors)
        if not violation or not compliance_checker.should_report(student_id, violation["color"]):
            return

        student_name = next(
            (item.get("student_name") for item in enrolled if item.get("student_id") == student_id),
            "Student",
        )
        detected_color = violation["color"]
        description = f"Uniform policy violation: {student_name} is wearing {detected_color} clothing."
        try:
            alert = create_alert_record("compliance_violation", description, student_id=student_id)
            compliance_checker.mark_reported(student_id, detected_color)
            voice_announcer.announce(
                f"Uniform policy violation. {student_name}, please wear an approved uniform."
            )
            publish_from_worker({"type": "alert", **alert})
            logger.info("%s", description)
        except Exception as error:
            logger.warning("Could not record uniform policy violation for student %s: %s", student_id, error)


live_face_matcher = LiveFaceMatcher()
