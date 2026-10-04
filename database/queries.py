import ast
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from database.supabase_client import get_supabase
from utils.logger import logger
from utils.uniform_policy import determine_checkin_status, get_gesture_enrolled_students, is_past_time_out


class DatabaseUnavailableError(RuntimeError):
    """Raised when a request cannot be fulfilled from the configured database."""


def _client():
    client = get_supabase()
    if client is None:
        raise DatabaseUnavailableError(
            "Database is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY, then restart the API."
        )
    return client


def _database_error(action: str, error: Exception) -> DatabaseUnavailableError:
    logger.error("Database %s failed: %s", action, error)
    return DatabaseUnavailableError(f"Database {action} failed. Check the API logs and Supabase configuration.")


def _format_attendance_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for row in rows:
        student = row.get("students") or {}
        records.append({
            "id": row.get("id"),
            "student_id": row.get("student_id"),
            "student_name": student.get("full_name", "Unknown"),
            "student_code": student.get("student_id", ""),
            "section": student.get("section"),
            "status": row.get("status"),
            "check_in_time": row.get("check_in_time"),
            "check_out_time": row.get("check_out_time"),
            "confidence": row.get("confidence"),
            "class_date": row.get("class_date"),
        })
    return records


def get_all_students(section: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        query = _client().table("students").select("*")
        if section:
            query = query.eq("section", section)
        enrolled_by_gesture = get_gesture_enrolled_students()
        return [
            {**student, "gesture_enrolled": enrolled_by_gesture.get(str(student.get("id")), False)}
            for student in query.order("full_name").execute().data
        ]
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("student query", error) from error


def create_student_record(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = _client().table("students").insert(data).execute()
        if not result.data:
            raise DatabaseUnavailableError("Database did not return the created student.")
        return result.data[0]
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("student insert", error) from error


def update_student_record(student_id: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        result = _client().table("students").update(data).eq("id", student_id).execute()
        return result.data[0] if result.data else None
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("student update", error) from error


def _parse_embedding(value: Any) -> Optional[List[float]]:
    """Normalise Supabase vector values, which may be returned as a list or string."""
    if value is None:
        return None
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        vector = [float(item) for item in parsed]
        return vector if vector else None
    except (ValueError, TypeError, SyntaxError):
        logger.warning("Skipping an invalid face embedding returned by Supabase.")
        return None


def replace_face_embedding(student_id: int, embedding: List[float]) -> Dict[str, Any]:
    """Replace a student's enrolled embedding with the latest verified sample."""
    return replace_face_embeddings(student_id, [embedding])[0]


def replace_face_embeddings(student_id: int, embeddings: List[List[float]]) -> List[Dict[str, Any]]:
    """Replace all of a student's face samples with a verified set of embeddings."""
    if not embeddings:
        raise ValueError("At least one face embedding is required.")
    try:
        client = _client()
        client.table("face_embeddings").delete().eq("student_id", student_id).execute()
        result = client.table("face_embeddings").insert([
            {"student_id": student_id, "embedding": embedding}
            for embedding in embeddings
        ]).execute()
        if not result.data:
            raise DatabaseUnavailableError("Database did not return the saved face embeddings.")
        return result.data
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("face embedding save", error) from error


def get_face_embeddings() -> List[Dict[str, Any]]:
    """Load enrolled vectors and their student identity data for live matching."""
    try:
        gesture_enrolled_students = get_gesture_enrolled_students()
        result = (
            _client()
            .table("face_embeddings")
            .select("student_id, embedding, students(full_name, student_id, section)")
            .execute()
        )
        embeddings = []
        for row in result.data or []:
            vector = _parse_embedding(row.get("embedding"))
            student = row.get("students") or {}
            if vector:
                embeddings.append({
                    "student_id": row.get("student_id"),
                    "student_name": student.get("full_name", "Unknown"),
                    "student_code": student.get("student_id", ""),
                    "section": student.get("section", ""),
                    "gesture_enrolled": gesture_enrolled_students.get(str(row.get("student_id")), False),
                    "embedding": vector,
                })
        return embeddings
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("face embedding query", error) from error


def delete_student_record(student_id: int) -> bool:
    try:
        result = _client().table("students").delete().eq("id", student_id).execute()
        return bool(result.data)
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("student deletion", error) from error


def get_attendance_for_date(target_date: str) -> List[Dict[str, Any]]:
    try:
        result = (
            _client()
            .table("attendance")
            .select("*, students(full_name, student_id, section)")
            .eq("class_date", target_date)
            .order("check_in_time")
            .execute()
        )
        return _format_attendance_records(result.data)
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("attendance query", error) from error


def get_attendance_in_range(date_from: str, date_to: str, section: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        result = (
            _client()
            .table("attendance")
            .select("*, students(full_name, student_id, section)")
            .gte("class_date", date_from)
            .lte("class_date", date_to)
            .order("class_date", desc=True)
            .order("check_in_time")
            .execute()
        )
        records = _format_attendance_records(result.data)
        return [record for record in records if not section or record["section"] == section]
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("attendance report query", error) from error


def mark_attendance(
    student_id: int,
    confidence: Optional[float] = None,
    check_in_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Store a check-in and derive present, late or time out from its timestamp."""
    try:
        checked_in_at = check_in_time or datetime.now().astimezone()
        # Schedules use the backend's local wall-clock time, while stored
        # timestamps are UTC. Normalize first so the correct local date and
        # status are derived even when callers submit an ISO UTC timestamp.
        checked_in_at = checked_in_at.astimezone()
        class_date = checked_in_at.date().isoformat()
        existing = (
            _client()
            .table("attendance")
            .select("*")
            .eq("student_id", student_id)
            .eq("class_date", class_date)
            .limit(1)
            .execute()
        )
        if existing.data and existing.data[0].get("check_in_time"):
            # The first check-in wins, so recognising the same student again
            # cannot move their arrival time or downgrade a recorded time out.
            return existing.data[0]
        status = determine_checkin_status(checked_in_at)
        data = {
            "student_id": student_id,
            "class_date": class_date,
            "check_in_time": checked_in_at.astimezone(timezone.utc).isoformat(),
            "status": status,
            "confidence": confidence,
        }
        result = _client().table("attendance").upsert(data, on_conflict="student_id,class_date").execute()
        if not result.data:
            raise DatabaseUnavailableError("Database did not return the attendance record.")
        return result.data[0]
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("attendance update", error) from error


def mark_time_out(student_id: int, check_out_time: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """
    Record the departure stamp on today's attendance row.

    Only a departure later than the configured Time out is stored, so a student
    passing the camera during the day cannot overwrite their real sign-out. The
    row must already exist from a check-in, and the first sign-out wins.
    """
    checked_out_at = (check_out_time or datetime.now().astimezone()).astimezone()
    if not is_past_time_out(checked_out_at):
        return None
    try:
        result = (
            _client()
            .table("attendance")
            .update({
                "check_out_time": checked_out_at.astimezone(timezone.utc).isoformat(),
                "status": "time_out",
            })
            .eq("student_id", student_id)
            .eq("class_date", checked_out_at.date().isoformat())
            .is_("check_out_time", None)
            .execute()
        )
        return result.data[0] if result.data else None
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("attendance time out update", error) from error


def reset_attendance_for_date(target_date: Optional[str] = None) -> int:
    today_str = target_date or date.today().isoformat()
    try:
        result = _client().table("attendance").delete().eq("class_date", today_str).execute()
        return len(result.data or [])
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("attendance reset", error) from error


def get_alerts_list(is_resolved: Optional[bool] = None, limit: int = 20) -> List[Dict[str, Any]]:
    try:
        query = _client().table("alerts").select("*").order("created_at", desc=True).limit(limit)
        if is_resolved is not None:
            query = query.eq("is_resolved", is_resolved)
        return query.execute().data
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("alerts query", error) from error


def create_alert_record(
    alert_type: str,
    description: str,
    student_id: Optional[int] = None,
    image_url: Optional[str] = None,
    severity: str = "medium",
) -> Dict[str, Any]:
    try:
        data = {
            "type": alert_type,
            "description": description,
            "student_id": student_id,
            "image_url": image_url,
            "severity": severity,
            "is_resolved": False,
        }
        result = _client().table("alerts").insert(data).execute()
        if not result.data:
            raise DatabaseUnavailableError("Database did not return the created alert.")
        return result.data[0]
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("alert insert", error) from error


def update_alert_record(alert_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        result = _client().table("alerts").update(updates).eq("id", alert_id).execute()
        return result.data[0] if result.data else None
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("alert update", error) from error


def reset_all_alerts() -> int:
    try:
        result = _client().table("alerts").delete().neq("id", 0).execute()
        return len(result.data or [])
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("alerts reset", error) from error


# ── Accounts (teachers / administrators) ─────────────────────────────────────

PUBLIC_USER_FIELDS = "id, username, email, full_name, role, is_active, year_levels, sections, last_login_at, created_at, updated_at"


def list_user_accounts(role: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        query = _client().table("users").select(PUBLIC_USER_FIELDS)
        if role:
            query = query.eq("role", role)
        return query.order("full_name").execute().data
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("accounts query", error) from error


def find_account(identifier: str) -> Optional[Dict[str, Any]]:
    """Look up a login row by username or email, returning every column."""
    try:
        result = (
            _client()
            .table("users")
            .select("*")
            .or_(f"username.eq.{identifier},email.eq.{identifier}")
            .execute()
        )
        return result.data[0] if result.data else None
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("account lookup", error) from error


def create_account_record(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = _client().table("users").insert(data).execute()
        if not result.data:
            raise DatabaseUnavailableError("Database did not return the created account.")
        return {key: value for key, value in result.data[0].items() if key != "password_hash"}
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("account insert", error) from error


def update_account_record(user_id: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        result = _client().table("users").update(data).eq("id", user_id).execute()
        if not result.data:
            return None
        return {key: value for key, value in result.data[0].items() if key != "password_hash"}
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("account update", error) from error


def delete_account_record(user_id: int) -> bool:
    try:
        result = _client().table("users").delete().eq("id", user_id).execute()
        return bool(result.data)
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("account deletion", error) from error


# ── Audit trail ──────────────────────────────────────────────────────────────

def insert_audit_event(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = _client().table("audit_log").insert(event).execute()
        if not result.data:
            raise DatabaseUnavailableError("Database did not return the audit event.")
        return result.data[0]
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("audit event insert", error) from error


def list_audit_events(
    limit: int = 100,
    action: Optional[str] = None,
    search: Optional[str] = None,
) -> List[Dict[str, Any]]:
    try:
        query = _client().table("audit_log").select("*").order("created_at", desc=True).limit(limit)
        if action:
            query = query.eq("action", action)
        if search:
            pattern = search.replace(",", " ")
            query = query.or_(
                f"action.ilike.%{pattern}%,actor_username.ilike.%{pattern}%,"
                f"target.ilike.%{pattern}%,description.ilike.%{pattern}%"
            )
        return query.execute().data
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise _database_error("audit log query", error) from error
