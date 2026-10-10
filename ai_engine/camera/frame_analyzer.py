import threading
import time
from typing import Any, Dict, List, Optional

from ai_engine.camera.rtsp_client import camera_stream
from ai_engine.face_recognition.detector import face_detector
from ai_engine.face_recognition.live_matcher import live_face_matcher
from ai_engine.gesture_detection import gesture_detector
from ai_engine.security.weapon_detection import weapon_detector
from ai_engine.voice.announcer import voice_announcer
from database.queries import create_alert_record
from utils.config import settings
from utils.logger import logger
from utils.uniform_policy import get_runtime_controls
from websocket_manager import publish_from_worker

MATCH_INTERVAL = 1.0
IDLE_STOP_SECONDS = 5.0


class FrameAnalyzer:
    """
    Runs threat, face and gesture detection on a background thread and caches the
    results, so the MJPEG display loop only has to draw and encode. Detection lags
    the video by a frame or two instead of holding the video back.

    Viewers attach while they have the live stream open; the thread winds down a
    few seconds after the last one leaves, matching the previous behaviour where
    analysis only ran for connected viewers.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._viewers = 0
        self._last_seen = 0.0
        self._state: Dict[str, Any] = {
            "threats": [],
            "faces": [],
            "labels": [],
            "hand": {"detected": False, "open_palm": False, "bbox": None},
            "gesture_detected": False,
        }

    def attach(self) -> None:
        with self._lock:
            self._viewers += 1
            self._last_seen = time.monotonic()
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._loop, name="frame-analyzer", daemon=True)
            self._thread.start()
        logger.info("Frame analyzer started.")

    def detach(self) -> None:
        with self._lock:
            self._viewers = max(0, self._viewers - 1)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            state = dict(self._state)
        state["hand"] = dict(state["hand"])
        return state

    def _should_stop(self) -> bool:
        with self._lock:
            if self._viewers > 0:
                self._last_seen = time.monotonic()
                return False
            return time.monotonic() - self._last_seen > IDLE_STOP_SECONDS

    def _publish(self, **updates) -> None:
        with self._lock:
            self._state.update(updates)

    def _handle_threats(self, frame, alert_mode_enabled: bool) -> List[Dict[str, Any]]:
        if not settings.WEAPON_DETECTION_ENABLED or frame is None:
            return []
        threats = weapon_detector.scan_threats(frame)
        if not alert_mode_enabled:
            return threats
        for threat in threats:
            threat_class = threat["class"]
            if not weapon_detector.claim_alert(threat_class):
                continue
            description = f"Security policy violation: {threat_class.title()} detected."
            try:
                alert = create_alert_record("weapon_detected", description, None, None, "high")
                voice_announcer.announce(f"Security violation. {threat_class} detected.")
                publish_from_worker({"type": "alert", **alert})
                logger.warning("%s", description)
            except Exception as error:
                logger.warning("Could not record detected %s: %s", threat_class, error)
        return threats

    def _loop(self) -> None:
        import cv2

        threat_interval = max(0.2, settings.THREAT_SCAN_INTERVAL)
        detection_interval = 1.0 / max(0.5, settings.FACE_DETECTION_FPS)
        last_controls_check = 0.0
        last_threat_time = 0.0
        last_detection_time = 0.0
        last_match_time = 0.0
        last_gesture_time = 0.0
        runtime_controls = get_runtime_controls()
        face_labels: List[Any] = []

        while True:
            if self._should_stop():
                break

            now = time.monotonic()
            due_threat = settings.WEAPON_DETECTION_ENABLED and now - last_threat_time >= threat_interval
            due_face = settings.FACE_DETECTION_ENABLED and now - last_detection_time >= detection_interval
            due_hand = now - last_gesture_time >= MATCH_INTERVAL

            # Copying a frame costs ~2.7 MB, so only do it when something is due.
            if not (due_threat or due_face or due_hand):
                time.sleep(0.05)
                continue

            frame = camera_stream.get_latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            if now - last_controls_check >= 1.0:
                runtime_controls = get_runtime_controls()
                last_controls_check = now
            if runtime_controls["camera_flip_horizontal"]:
                frame = cv2.flip(frame, 1)

            if due_threat:
                last_threat_time = now
                self._publish(threats=self._handle_threats(frame, runtime_controls["alert_mode_enabled"]))

            if due_hand:
                last_gesture_time = now
                hand = gesture_detector.detect_hand(frame)
                self._publish(hand=hand, gesture_detected=hand["open_palm"])

            if due_face:
                last_detection_time = now
                detected_faces, yunet_rows = face_detector.detect_faces_with_landmarks(frame)
                hand = self._state["hand"]
                if detected_faces and (now - last_match_time >= MATCH_INTERVAL or len(detected_faces) != len(face_labels)):
                    face_labels = live_face_matcher.match_frame(
                        frame,
                        detected_faces,
                        camera_stream.attendance_recording,
                        hand["open_palm"],
                        yunet_rows,
                    )
                    last_match_time = now
                elif not detected_faces:
                    face_labels = []
                self._publish(faces=detected_faces, labels=face_labels)

        logger.info("Frame analyzer stopped.")


frame_analyzer = FrameAnalyzer()
