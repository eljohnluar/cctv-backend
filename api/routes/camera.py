import asyncio
import time
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from ai_engine.camera.frame_analyzer import frame_analyzer
from ai_engine.camera.rtsp_client import camera_stream
from ai_engine.face_recognition.detector import face_detector
from ai_engine.security.weapon_detection import weapon_detector
from ai_engine.voice.announcer import voice_announcer
from utils.config import settings
from utils.uniform_policy import get_runtime_controls

router = APIRouter(prefix="/camera", tags=["camera"])


class AttendanceRecordingRequest(BaseModel):
    enabled: bool


class VoiceTestRequest(BaseModel):
    message: str = "Voice announcer is ready."


async def _mjpeg_frames() -> AsyncIterator[bytes]:
    """
    Publish the newest camera frame with whatever detections are already known.
    Detection itself runs on the analyzer thread, so a slow YOLO pass delays a
    bounding box by a frame instead of stalling the video.
    """
    import cv2

    frame_analyzer.attach()
    frame_interval = 1.0 / max(1, settings.STREAM_MAX_FPS)
    last_controls_check = 0.0
    last_sent_at = 0.0
    last_seq = 0
    runtime_controls = get_runtime_controls()

    try:
        while True:
            frame, seq = camera_stream.read_if_new(last_seq)
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            last_seq = seq

            now = time.monotonic()
            if now - last_sent_at < frame_interval:
                continue
            last_sent_at = now

            if now - last_controls_check >= 1.0:
                runtime_controls = get_runtime_controls()
                last_controls_check = now

            display_frame = cv2.flip(frame, 1) if runtime_controls["camera_flip_horizontal"] else frame
            overlay = frame_analyzer.snapshot()

            if settings.WEAPON_DETECTION_ENABLED and overlay["threats"]:
                display_frame = weapon_detector.draw_threat_boxes(display_frame, overlay["threats"])

            if settings.FACE_DETECTION_ENABLED and overlay["faces"]:
                display_frame = face_detector.draw_face_boxes(display_frame, overlay["faces"], overlay["labels"])

            hand = overlay["hand"]
            if hand["detected"] and hand["bbox"]:
                x, y, width, height = hand["bbox"]
                label = "OPEN PALM" if overlay["gesture_detected"] else "HAND"
                color = (80, 210, 120) if overlay["gesture_detected"] else (60, 190, 240)
                cv2.rectangle(display_frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(display_frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            success, encoded = cv2.imencode(
                ".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, settings.STREAM_JPEG_QUALITY]
            )
            if success:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
    finally:
        frame_analyzer.detach()


@router.get("/stream")
async def stream_camera():
    """Stream the latest camera frame as MJPEG for the web dashboard."""
    try:
        import cv2  # noqa: F401
    except ImportError as error:
        raise HTTPException(status_code=503, detail="OpenCV is not installed; the camera stream is unavailable.") from error

    if not camera_stream.is_running:
        camera_stream.start()

    return StreamingResponse(
        _mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@router.get("/snapshot")
def camera_snapshot():
    """
    Return the newest camera frame as a single JPEG.

    Face enrollment reads this instead of drawing the MJPEG <img> onto a canvas:
    a cross-origin stream taints the canvas, and toDataURL() then throws
    "Tainted canvases may not be exported" instead of producing a sample.
    """
    try:
        import cv2
    except ImportError as error:
        raise HTTPException(status_code=503, detail="OpenCV is not installed; the camera snapshot is unavailable.") from error

    frame = camera_stream.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="The camera has not produced a frame yet.")

    encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not encoded:
        raise HTTPException(status_code=500, detail="The camera frame could not be encoded.")
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/attendance-recording")
async def get_attendance_recording():
    return {
        "ready": camera_stream.is_connected,
        "attendance_recording": camera_stream.attendance_recording,
    }


@router.post("/attendance-recording")
async def set_attendance_recording(payload: AttendanceRecordingRequest):
    if payload.enabled and not camera_stream.is_connected:
        raise HTTPException(status_code=409, detail="The camera is not ready. Please ensure the camera source is active.")
    camera_stream.attendance_recording = payload.enabled
    return {
        "ready": camera_stream.is_connected,
        "attendance_recording": camera_stream.attendance_recording,
    }


@router.post("/test-voice")
async def test_voice(payload: VoiceTestRequest = VoiceTestRequest()):
    """Play a short test through the server speaker and connected dashboards."""
    voice_announcer.announce(payload.message)
    return {"success": True, "message": "Voice announcement queued."}
