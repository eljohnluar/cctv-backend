from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from datetime import date
from api.middleware.auth import get_optional_account, require_password_confirmation
from api.models.attendance import (
    AttendanceManualMark,
    AttendanceRecord,
    AttendanceStats,
    TodayAttendanceResponse
)
from database.queries import (
    get_attendance_for_date,
    mark_attendance,
    get_all_students,
    reset_attendance_for_date,
)
from ai_engine.face_recognition.live_matcher import live_face_matcher
from websocket_manager import publish_from_worker
from utils.logger import logger
from utils.sections import resolve_allowed_sections

router = APIRouter(prefix="/attendance", tags=["Attendance"])


def _scope_to_account(records: List[dict], students: List[dict], claims: Optional[dict]):
    """Keep only the rows belonging to sections the signed-in teacher handles."""
    allowed = resolve_allowed_sections(claims)
    if allowed is None:
        return records, students
    return (
        [record for record in records if record.get("section") in allowed],
        [student for student in students if student.get("section") in allowed],
    )


def _guard_student_scope(student_id: int, claims: Optional[dict]) -> None:
    """Reject a manual check-in for a student the signed-in teacher does not handle."""
    allowed = resolve_allowed_sections(claims)
    if allowed is None:
        return
    student = next((item for item in get_all_students() if item.get("id") == student_id), None)
    if not student or student.get("section") not in allowed:
        raise HTTPException(status_code=403, detail="That student is outside the year levels and sections assigned to your account.")

def calculate_stats(records: List[dict], total_enrolled: int) -> AttendanceStats:
    total = len(records)
    present = sum(1 for r in records if r.get("status") == "present")
    late = sum(1 for r in records if r.get("status") == "late")
    time_out = sum(1 for r in records if r.get("status") == "time_out")
    effective_total = total_enrolled if total_enrolled > 0 else total
    rate = round(((present + late + time_out) / effective_total * 100), 1) if effective_total > 0 else 0.0
    return AttendanceStats(
        total=effective_total,
        present=present,
        late=late,
        time_out=time_out,
        rate=rate
    )

@router.get("/today", response_model=TodayAttendanceResponse)
def get_today_attendance(claims: Optional[dict] = Depends(get_optional_account)):
    """Fetch attendance records and live summary metrics for today"""
    today_str = date.today().isoformat()
    records, all_students = _scope_to_account(get_attendance_for_date(today_str), get_all_students(), claims)
    stats = calculate_stats(records, len(all_students))
    return TodayAttendanceResponse(
        date=today_str,
        stats=stats,
        records=records
    )

@router.get("/date/{query_date}", response_model=TodayAttendanceResponse)
def get_attendance_by_date(query_date: str, claims: Optional[dict] = Depends(get_optional_account)):
    """Fetch attendance records for a specific historical date (YYYY-MM-DD)"""
    records, all_students = _scope_to_account(get_attendance_for_date(query_date), get_all_students(), claims)
    stats = calculate_stats(records, len(all_students))
    return TodayAttendanceResponse(
        date=query_date,
        stats=stats,
        records=records
    )

@router.post("/manual")
def manual_mark(payload: AttendanceManualMark, claims: Optional[dict] = Depends(get_optional_account)):
    """Record attendance and derive status from the check-in time."""
    _guard_student_scope(payload.student_id, claims)
    record = mark_attendance(
        student_id=payload.student_id,
        confidence=payload.confidence,
        check_in_time=payload.check_in_time,
    )
    return {
        "success": True,
        "message": f"Attendance updated for student {payload.student_id}",
        "record": record
    }

@router.post("/reset")
def reset_attendance(_: dict = Depends(require_password_confirmation)):
    """Reset all marked attendance records for today"""
    today_str = date.today().isoformat()
    try:
        count = reset_attendance_for_date(today_str)
    except Exception as e:
        logger.warning(f"Database reset attendance failed: {e}")
        count = 0

    # Clear in-memory live face matcher cache so students can be recognized again
    live_face_matcher.reset_marked()

    # Broadcast reset event over WebSocket so all connected dashboards clear marked state
    publish_from_worker({
        "type": "attendance_reset",
        "class_date": today_str,
    })

    return {
        "success": True,
        "message": f"Attendance records reset for {today_str}",
        "deleted_count": count
    }
