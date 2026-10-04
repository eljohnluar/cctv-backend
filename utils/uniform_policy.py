"""Persistent, local runtime configuration for the uniform-color policy."""

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_UNIFORM_COLORS = ("blue", "dark-blue", "light-blue", "white", "red", "dark-red", "light-red", "black")
VALID_VOICE_GENDERS = ("female", "male")
_SETTINGS_PATH = Path(__file__).resolve().parents[1] / ".runtime-settings.json"
_LOCK = threading.Lock()


def _read_settings() -> Dict[str, Any]:
    try:
        return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_settings(data: Dict[str, Any]) -> None:
    temp_path = _SETTINGS_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(data), encoding="utf-8")
    temp_path.replace(_SETTINGS_PATH)


def get_uniform_policy() -> Dict[str, List[str]]:
    """Return allowed uniform colors. An empty list means no colors are allowed."""
    with _LOCK:
        colors = _read_settings().get("uniform_colors", [])
        return {"uniform_colors": [color for color in colors if color in VALID_UNIFORM_COLORS]}


def save_uniform_policy(colors: List[str]) -> Dict[str, List[str]]:
    """Persist a validated uniform-color policy for the running backend."""
    normalized = list(dict.fromkeys(color.lower().strip() for color in colors))
    invalid = [color for color in normalized if color not in VALID_UNIFORM_COLORS]
    if invalid:
        raise ValueError(f"Unsupported uniform color: {invalid[0]}")

    with _LOCK:
        data = _read_settings()
        data["uniform_colors"] = normalized
        _write_settings(data)
    return {"uniform_colors": normalized}


def get_voice_settings() -> Dict[str, str]:
    """Return the preferred gender for system voice announcements."""
    with _LOCK:
        gender = _read_settings().get("voice_gender", "female")
        return {"voice_gender": gender if gender in VALID_VOICE_GENDERS else "female"}


def save_voice_settings(gender: str) -> Dict[str, str]:
    """Persist the preferred gender for system voice announcements."""
    normalized = gender.lower().strip()
    if normalized not in VALID_VOICE_GENDERS:
        raise ValueError("Voice gender must be female or male.")
    with _LOCK:
        data = _read_settings()
        data["voice_gender"] = normalized
        _write_settings(data)
    return {"voice_gender": normalized}


def get_gesture_attendance_settings() -> Dict[str, bool]:
    """Return whether an open palm is required to confirm attendance."""
    with _LOCK:
        enabled = _read_settings().get("gesture_attendance_enabled", False)
        return {"gesture_attendance_enabled": bool(enabled)}


def save_gesture_attendance_settings(enabled: bool) -> Dict[str, bool]:
    """Persist the open-palm attendance and enrollment confirmation mode."""
    with _LOCK:
        data = _read_settings()
        data["gesture_attendance_enabled"] = bool(enabled)
        _write_settings(data)
    return {"gesture_attendance_enabled": bool(enabled)}


def get_gesture_enrolled_students() -> Dict[str, bool]:
    """Return the student IDs whose enrollment requires an open-palm check-in.

    This lives with the runtime settings so deployments with the original
    Supabase schema work without requiring a database migration.
    """
    with _LOCK:
        saved = _read_settings().get("gesture_enrolled_students", {})
        return {str(student_id): bool(enabled) for student_id, enabled in saved.items()} if isinstance(saved, dict) else {}


def save_student_gesture_enrollment(student_id: int, enabled: bool) -> None:
    """Record whether one student's latest enrollment included the gesture."""
    with _LOCK:
        data = _read_settings()
        enrolled = data.get("gesture_enrolled_students", {})
        if not isinstance(enrolled, dict):
            enrolled = {}
        enrolled[str(student_id)] = bool(enabled)
        data["gesture_enrolled_students"] = enrolled
        _write_settings(data)


def get_runtime_controls() -> Dict[str, Any]:
    """Return persisted controls that affect live camera behavior."""
    with _LOCK:
        data = _read_settings()
        return {
            "camera_flip_horizontal": bool(data.get("camera_flip_horizontal", False)),
            "announcer_enabled": bool(data.get("announcer_enabled", True)),
            "announcer_volume": max(0, min(100, int(data.get("announcer_volume", 100)))),
            "alert_mode_enabled": bool(data.get("alert_mode_enabled", True)),
        }


def save_runtime_controls(
    camera_flip_horizontal: bool,
    announcer_enabled: bool,
    announcer_volume: int,
    alert_mode_enabled: bool,
) -> Dict[str, Any]:
    """Persist live camera, announcement, and alert-mode controls."""
    if not 0 <= int(announcer_volume) <= 100:
        raise ValueError("Announcer volume must be between 0 and 100.")
    with _LOCK:
        data = _read_settings()
        result = {
            "camera_flip_horizontal": bool(camera_flip_horizontal),
            "announcer_enabled": bool(announcer_enabled),
            "announcer_volume": int(announcer_volume),
            "alert_mode_enabled": bool(alert_mode_enabled),
        }
        data.update(result)
        _write_settings(data)
    return result


def get_schedule_settings() -> Dict[str, Any]:
    """Return the configured check-in time, late grace window and check-in timeout."""
    with _LOCK:
        settings = _read_settings()
        checkin_time = settings.get("checkin_time", "08:00")
        grace = max(0, min(180, int(settings.get("late_grace_minutes", 30))))
        timeout = max(grace, min(1440, int(settings.get("attendance_timeout_minutes", 120))))
        return {
            "checkin_time": checkin_time,
            "late_grace_minutes": grace,
            "attendance_timeout_minutes": timeout,
        }


def save_schedule_settings(
    checkin_time: str,
    late_grace_minutes: int = 30,
    attendance_timeout_minutes: int = 120,
) -> Dict[str, Any]:
    """Persist the check-in time (HH:MM, 24-hour), grace window and timeout."""
    parts = checkin_time.strip().split(":")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError("Check-in time must be in HH:MM format (24-hour).")
    h, m = int(parts[0]), int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError("Invalid hour or minute in check-in time.")
    if not 0 <= int(late_grace_minutes) <= 180:
        raise ValueError("Late grace minutes must be between 0 and 180.")
    if not 1 <= int(attendance_timeout_minutes) <= 1440:
        raise ValueError("Attendance timeout must be between 1 and 1440 minutes.")
    if int(attendance_timeout_minutes) < int(late_grace_minutes):
        raise ValueError("Attendance timeout must be at least as long as the late grace window.")
    formatted = f"{h:02d}:{m:02d}"

    with _LOCK:
        data = _read_settings()
        data["checkin_time"] = formatted
        data["late_grace_minutes"] = int(late_grace_minutes)
        data["attendance_timeout_minutes"] = int(attendance_timeout_minutes)
        _write_settings(data)
    return {
        "checkin_time": formatted,
        "late_grace_minutes": int(late_grace_minutes),
        "attendance_timeout_minutes": int(attendance_timeout_minutes),
    }


def _as_local_datetime(value: Any) -> datetime:
    now = value or datetime.now()
    if isinstance(now, str):
        try:
            now = datetime.fromisoformat(now.replace("Z", "+00:00"))
        except Exception:
            now = datetime.now()
    if now.tzinfo is not None:
        # The configured schedule is a local wall-clock time. Convert timestamps
        # received in UTC (for example, from a manual API request) before
        # comparing their clock time to the schedule.
        now = now.astimezone()
    return now


def _time_out_cutoff(now: datetime) -> Optional[datetime]:
    """The clock the schedule labels 'Time out': check-in start plus the timeout."""
    schedule = get_schedule_settings()
    try:
        hour, minute = map(int, schedule.get("checkin_time", "08:00").split(":"))
        return now.replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0,
        ) + timedelta(minutes=int(schedule.get("attendance_timeout_minutes", 120)))
    except Exception:
        return None


def is_past_time_out(at: Any = None) -> bool:
    """True when the given moment is later than the configured Time out."""
    now = _as_local_datetime(at)
    cutoff = _time_out_cutoff(now)
    return bool(cutoff) and now > cutoff


def determine_checkin_status(checkin_dt: Any = None) -> str:
    """
    Determine attendance status based on set check-in time:
    - Before the grace cutoff: 'present'
    - After the grace cutoff but inside the attendance window: 'late'
    - After the Time out: 'time_out'
    """
    schedule = get_schedule_settings()
    checkin_time_str = schedule.get("checkin_time", "08:00")
    grace_minutes = int(schedule.get("late_grace_minutes", 30))
    timeout_minutes = int(schedule.get("attendance_timeout_minutes", 120))

    now = _as_local_datetime(checkin_dt)

    try:
        hour, minute = map(int, checkin_time_str.split(":"))
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        late_cutoff = target + timedelta(minutes=grace_minutes)
        attendance_cutoff = target + timedelta(minutes=timeout_minutes)
        if now > attendance_cutoff:
            return "time_out"
        if now > late_cutoff:
            return "late"
        return "present"
    except Exception:
        return "present"
