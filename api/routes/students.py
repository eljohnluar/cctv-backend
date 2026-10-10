from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from typing import List, Optional
from api.middleware.auth import get_optional_account, require_password_confirmation
from api.models.student import StudentCreate, StudentUpdate, StudentResponse
from database.queries import (
    get_all_students,
    create_student_record,
    update_student_record,
    delete_student_record,
    replace_face_embeddings,
    reset_all_face_enrollments,
    reset_student_face_enrollment,
)
from ai_engine.face_recognition.detector import face_detector
from ai_engine.face_recognition.recognizer import face_recognizer
from ai_engine.face_recognition.live_matcher import live_face_matcher
from database.storage import delete_face_image, download_face_image, upload_face_image
from utils.logger import logger
from utils.sections import resolve_allowed_sections, year_level_of
from utils.uniform_policy import get_gesture_attendance_settings, save_student_gesture_enrollment
from ai_engine.gesture_detection import gesture_detector

router = APIRouter(prefix="/students", tags=["Students"])


def _guard_section(section: Optional[str], claims: Optional[dict]) -> None:
    """Reject a section the signed-in teacher does not handle."""
    allowed = resolve_allowed_sections(claims)
    if allowed is None:
        return
    if not allowed:
        raise HTTPException(status_code=403, detail="No sections are assigned to your account yet. Ask an administrator to assign your year levels and sections.")
    if (section or "") not in allowed:
        raise HTTPException(status_code=403, detail="That section is outside the year levels and sections assigned to your account.")


@router.get("", response_model=List[StudentResponse])
def list_students(section: Optional[str] = None, claims: Optional[dict] = Depends(get_optional_account)):
    """Retrieve enrolled students, limited to the sections the caller handles"""
    students = get_all_students(section=section)
    allowed = resolve_allowed_sections(claims)
    if allowed is not None:
        students = [student for student in students if student.get("section") in allowed]
    return students

@router.post("", response_model=StudentResponse)
def add_student(student_in: StudentCreate, claims: Optional[dict] = Depends(get_optional_account), _: dict = Depends(require_password_confirmation)):
    """Enroll a new student record"""
    data = student_in.dict()
    _guard_section(data.get("section"), claims)
    if data.get("section"):
        data["grade_level"] = year_level_of(data["section"]) or data.get("grade_level")
    created = create_student_record(data)
    return created

@router.get("/{student_id}", response_model=StudentResponse)
def get_student(student_id: int, claims: Optional[dict] = Depends(get_optional_account)):
    """Get student details by numerical ID"""
    all_s = list_students(claims=claims)
    match = next((s for s in all_s if s["id"] == student_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Student not found")
    return match


@router.get("/{student_id}/enrollment-photo")
def get_front_enrollment_photo(student_id: int):
    """Serve only the front enrollment capture for the attendance confirmation."""
    student = next((item for item in get_all_students() if item["id"] == student_id), None)
    expected_path = f"students/{student_id}/enrollment-front.jpg"
    if not student or student.get("face_storage_path") != expected_path:
        raise HTTPException(status_code=404, detail="A front enrollment photo is not available for this student.")
    try:
        return Response(
            content=download_face_image(expected_path),
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=60"},
        )
    except Exception as error:
        logger.warning("Could not retrieve front enrollment photo for student %s: %s", student_id, error)
        raise HTTPException(status_code=404, detail="The front enrollment photo could not be retrieved.") from error

@router.put("/{student_id}", response_model=StudentResponse)
def update_student(student_id: int, updates: StudentUpdate, claims: Optional[dict] = Depends(get_optional_account)):
    """Update student information"""
    data = {k: v for k, v in updates.dict().items() if v is not None}
    _guard_section(data.get("section"), claims)
    if data.get("section"):
        data["grade_level"] = year_level_of(data["section"])
    updated = update_student_record(student_id, data)
    if not updated:
        raise HTTPException(status_code=404, detail="Student not found")
    return updated

@router.delete("/{student_id}")
def delete_student(student_id: int, _: dict = Depends(require_password_confirmation)):
    """Remove student record"""
    success = delete_student_record(student_id)
    if not success:
        raise HTTPException(status_code=404, detail="Student not found")
    return {"success": True, "message": "Student deleted"}

@router.post("/face-enroll")
async def enroll_face(
    student_id: int = Form(...),
    images: List[UploadFile] = File(...),
    _: dict = Depends(require_password_confirmation),
):
    """
    Enroll four face-angle samples for facial recognition.
    Each sample produces a separate embedding so live recognition can match the
    student at different angles.
    """
    try:
        import cv2
        import numpy as np

        required_angles = ("front", "left", "right", "upward")
        if len(images) != len(required_angles):
            raise HTTPException(status_code=422, detail="Capture all four required face angles before enrolling.")

        embeddings = []
        stored_paths = []
        last_box = None
        require_hand_gesture = get_gesture_attendance_settings()["gesture_attendance_enabled"]
        for angle, image in zip(required_angles, images):
            content = await image.read()
            frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise HTTPException(status_code=400, detail=f"The {angle} image could not be decoded.")
            if require_hand_gesture and not gesture_detector.is_open_palm(frame):
                raise HTTPException(
                    status_code=422,
                    detail=f"Show an open palm with your face before capturing the {angle} enrollment photo.",
                )

            faces, landmark_rows = face_detector.detect_faces_with_landmarks(frame)
            if not faces:
                # Profile captures are harder for the detector than the front
                # view. Retry with relaxed thresholds before giving up.
                faces, landmark_rows = face_detector.detect_faces_with_landmarks(
                    frame,
                    min_confidence=0.5,
                    min_neighbors=4,
                    min_size=20,
                    detection_width=960,
                )
            if not faces:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"No face was detected in the {angle} photo. "
                        "Ensure your face is clearly visible, well-lit, and centred in the frame, then retake the photo."
                    ),
                )

            largest_index = max(range(len(faces)), key=lambda i: faces[i][2] * faces[i][3])
            x, y, width, height = faces[largest_index]
            yunet_row = (
                landmark_rows[largest_index]
                if landmark_rows and largest_index < len(landmark_rows)
                else None
            )
            x = max(0, x)
            y = max(0, y)
            width = min(frame.shape[1] - x, width)
            height = min(frame.shape[0] - y, height)
            if width <= 0 or height <= 0:
                raise HTTPException(status_code=422, detail=f"No face was detected in the {angle} capture. Retake it in good lighting.")
            last_box = (x, y, width, height)

            padding_x = round(width * 0.2)
            padding_y = round(height * 0.25)
            left = max(0, x - padding_x)
            top = max(0, y - padding_y)
            right = min(frame.shape[1], x + width + padding_x)
            bottom = min(frame.shape[0], y + height + padding_y)
            face_crop = frame[top:bottom, left:right]
            # Match the live-feed pipeline exactly: landmark-aligned SFace
            # embeddings (alignCrop). Resize-only enrollment vectors scored
            # too low against aligned live vectors, so enrolled faces kept
            # showing as UNREGISTERED.
            embedding = face_recognizer.extract_embedding(
                face_crop, full_frame=frame, yunet_face_row=yunet_row
            )
            if not embedding:
                raise HTTPException(
                    status_code=503,
                    detail="Face embedding is unavailable. Ensure the OpenCV SFace model is present and restart the API.",
                )
            if len(embedding) != 128:
                raise HTTPException(status_code=422, detail=f"The recognition model returned an unsupported embedding for the {angle} capture.")

            encoded_success, encoded_face = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not encoded_success:
                raise HTTPException(status_code=500, detail=f"The {angle} face could not be encoded for storage.")
            embeddings.append(embedding)
            stored_paths.append(upload_face_image(student_id, encoded_face.tobytes(), angle))

        try:
            replace_face_embeddings(student_id, embeddings)
            update_student_record(student_id, {
                "has_face": True,
                "face_storage_path": stored_paths[0],
            })
            save_student_gesture_enrollment(student_id, require_hand_gesture)
            live_face_matcher.refresh_embeddings()
        except Exception:
            for path in stored_paths:
                delete_face_image(path)
            raise
        
        return {
            "success": True,
            "message": f"Four facial embeddings generated and registered for student {student_id}",
            "student_id": student_id,
            "has_face": True,
            "gesture_enrolled": require_hand_gesture,
            "face_storage_path": stored_paths[0],
            "samples": len(embeddings),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during face enrollment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-enrollments")
def reset_all_student_enrollments(_: dict = Depends(require_password_confirmation)):
    """Reset all student face enrollments and embeddings."""
    count = reset_all_face_enrollments()
    live_face_matcher.refresh_embeddings()
    live_face_matcher.reset_marked()
    return {"success": True, "message": f"Reset face enrollment for {count} students", "count": count}


@router.delete("/{student_id}/face")
def reset_single_student_face(student_id: int, _: dict = Depends(require_password_confirmation)):
    """Reset face enrollment for a single student."""
    success = reset_student_face_enrollment(student_id)
    if not success:
        raise HTTPException(status_code=404, detail="Student not found")
    live_face_matcher.refresh_embeddings()
    return {"success": True, "message": f"Face enrollment reset for student {student_id}"}

