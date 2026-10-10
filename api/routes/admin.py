from datetime import date
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from api.middleware.auth import require_admin, require_password_confirmation, verify_account_password
from database.queries import (
    DatabaseUnavailableError,
    create_account_record,
    delete_account_record,
    get_alerts_list,
    get_all_students,
    get_attendance_for_date,
    get_attendance_in_range,
    clear_audit_log,
    list_audit_events,
    list_user_accounts,
    update_account_record,
)
from database.supabase_client import get_supabase
from utils.audit import record_audit
from utils.config import settings
from utils.demo_accounts import (
    create_fallback,
    delete_fallback,
    hash_password,
    list_fallback,
    update_fallback,
)
from utils.logger import logger
from utils.sections import normalise_letters, normalise_years

router = APIRouter(prefix="/admin", tags=["Admin"])

AdminClaims = Annotated[dict, Depends(require_admin)]


class TeacherCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: Optional[str] = Field(default="password123")
    full_name: Optional[str] = None
    email: Optional[str] = None
    role: str = Field(default="teacher", description="teacher or admin")
    year_levels: Optional[List[str]] = Field(default_factory=list, description="Year levels handled, e.g. '1st Year'")
    sections: Optional[List[str]] = Field(default_factory=list, description="Section letters handled, e.g. 'A'")


class TeacherUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = Field(default=None, min_length=4)
    is_active: Optional[bool] = None
    role: Optional[str] = None
    year_levels: Optional[List[str]] = None
    sections: Optional[List[str]] = None


def _accounts_available() -> bool:
    return get_supabase() is not None


def _list_accounts(role: Optional[str] = None) -> List[dict]:
    if _accounts_available():
        return list_user_accounts(role)
    return list_fallback(role)


def _account_exists(username: str) -> bool:
    client = get_supabase()
    if not client:
        return False
    result = client.table("users").select("id").eq("username", username).execute()
    return bool(result.data)


def _guard_last_admin(target_id: int, action: str) -> None:
    """Prefer keeping at least one usable administrator account in the system."""
    admins = [account for account in _list_accounts("admin") if account.get("is_active", True)]
    if len(admins) > 1 or not any(admin.get("id") == target_id for admin in admins):
        return
    raise HTTPException(status_code=409, detail=f"Cannot {action}: at least one administrator account must stay active.")


def _normalise_role(role: str) -> str:
    cleaned = (role or "teacher").strip().lower()
    if cleaned not in ("teacher", "admin"):
        raise HTTPException(status_code=422, detail="Role must be either 'teacher' or 'admin'.")
    return cleaned


def _assigned_years(values) -> List[str]:
    return normalise_years(values)


def _assigned_letters(values) -> List[str]:
    return normalise_letters(values)


@router.get("/summary")
def admin_summary(_: AdminClaims):
    """Headline counts for the administrator dashboard."""
    teachers = _list_accounts("teacher")
    admins = _list_accounts("admin")
    students = get_all_students()
    today_records = get_attendance_for_date(date.today().isoformat())
    unresolved = get_alerts_list(is_resolved=False, limit=200)
    sections = sorted({student.get("section") for student in students if student.get("section")})

    return {
        "teacher_count": len(teachers),
        "active_teacher_count": len([account for account in teachers if account.get("is_active", True)]),
        "admin_count": len(admins),
        "student_count": len(students),
        "section_count": len(sections),
        "attendance_today": {
            "total": len(today_records),
            "present": len([record for record in today_records if record.get("status") == "present"]),
            "late": len([record for record in today_records if record.get("status") == "late"]),
            "time_out": len([record for record in today_records if record.get("status") == "time_out"]),
        },
        "unresolved_alert_count": len(unresolved),
        "sections": sections,
        "recent_accounts": teachers[:5],
    }


@router.get("/teachers")
def list_teachers(
    _: AdminClaims,
    role: Optional[str] = Query(default=None, description="Filter by teacher or admin"),
    search: Optional[str] = Query(default=None),
):
    accounts = _list_accounts(role)
    if search:
        needle = search.strip().lower()
        accounts = [
            account
            for account in accounts
            if needle in str(account.get("username", "")).lower()
            or needle in str(account.get("full_name", "")).lower()
            or needle in str(account.get("email", "")).lower()
        ]
    return {"accounts": accounts, "count": len(accounts)}


@router.post("/teachers")
def create_teacher(payload: TeacherCreateRequest, request: Request, claims: AdminClaims):
    """Provision a teacher (or additional administrator) account."""
    username = payload.username.strip().lower().replace(" ", "")
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")

    raw_password = (payload.password or "").strip()
    effective_password = raw_password if len(raw_password) >= 4 else "password123"

    role = _normalise_role(payload.role)
    year_levels = _assigned_years(payload.year_levels)
    letters = _assigned_letters(payload.sections)
    email = (payload.email or f"{username}@smartcctv.edu").strip().lower()
    record = {
        "username": username,
        "email": email,
        "password_hash": hash_password(effective_password),
        "full_name": (payload.full_name or username).strip(),
        "role": role,
        "registration_code": settings.ADMIN_REGISTRATION_CODE if role == "admin" else settings.TEACHER_PROVISIONING_CODE,
        "is_active": True,
        "year_levels": year_levels,
        "sections": letters,
    }

    account = None
    if _accounts_available():
        if _account_exists(username):
            raise HTTPException(status_code=409, detail=f"Username '{username}' is already registered.")
        try:
            account = create_account_record(record)
        except Exception as err:
            logger.warning("Supabase account insert failed (%s). Using fallback store.", err)
            account = create_fallback(record)
    else:
        try:
            account = create_fallback(record)
        except ValueError as conflict:
            raise HTTPException(status_code=409, detail=str(conflict))

    record_audit(
        "account_created",
        f"{role.capitalize()} account '{username}' was created.",
        actor=claims,
        target=username,
        request=request,
    )
    logger.info("Administrator '%s' created %s account '%s'", claims.get("sub"), role, username)
    return {"account": account, "message": f"{role.capitalize()} account created."}


@router.put("/teachers/{user_id}")
def update_teacher(
    user_id: int,
    payload: TeacherUpdateRequest,
    request: Request,
    claims: AdminClaims,
    confirm_password: Optional[str] = Header(default=None, alias="X-Confirm-Password"),
):
    updates = {key: value for key, value in payload.model_dump().items() if value is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No changes were submitted.")

    if "is_active" in updates:
        if not confirm_password:
            raise HTTPException(
                status_code=401,
                detail="Enter your password to change an account's active state.",
            )
        if not verify_account_password(claims.get("sub", ""), confirm_password):
            raise HTTPException(status_code=401, detail="Password does not match your account.")

    if "role" in updates:
        updates["role"] = _normalise_role(updates["role"])
    if "year_levels" in updates:
        updates["year_levels"] = _assigned_years(updates["year_levels"])
    if "sections" in updates:
        updates["sections"] = _assigned_letters(updates["sections"])
    if "password" in updates:
        updates["password_hash"] = hash_password(updates.pop("password"))

    existing = next((account for account in _list_accounts() if account.get("id") == user_id), None)
    if not existing:
        raise HTTPException(status_code=404, detail="Account not found.")

    acting_on_self = str(claims.get("uid")) == str(user_id) or claims.get("sub") == existing.get("username")
    if acting_on_self and (updates.get("is_active") is False or ("role" in updates and updates["role"] != "admin")):
        raise HTTPException(status_code=409, detail="You cannot deactivate or demote your own administrator account.")
    if existing.get("role") == "admin" and (updates.get("is_active") is False or updates.get("role") == "teacher"):
        _guard_last_admin(user_id, "remove administrator access")

    if _accounts_available():
        account = update_account_record(user_id, updates)
    else:
        account = update_fallback(user_id, updates)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found.")

    changed = ", ".join(sorted(key for key in updates if key != "password_hash"))
    record_audit(
        "account_updated",
        f"Account '{account.get('username')}' was updated ({changed}).",
        actor=claims,
        target=str(account.get("username")),
        request=request,
    )
    return {"account": account, "message": "Account updated."}


@router.delete("/teachers/{user_id}")
def delete_teacher(user_id: int, request: Request, claims: AdminClaims, _: dict = Depends(require_password_confirmation)):
    existing = next((account for account in _list_accounts() if account.get("id") == user_id), None)
    if not existing:
        raise HTTPException(status_code=404, detail="Account not found.")
    if str(claims.get("uid")) == str(user_id) or claims.get("sub") == existing.get("username"):
        raise HTTPException(status_code=409, detail="You cannot delete the account you are signed in with.")
    if existing.get("role") == "admin":
        _guard_last_admin(user_id, "delete the final administrator account")

    if _accounts_available():
        delete_account_record(user_id)
    else:
        delete_fallback(user_id)

    record_audit(
        "account_deleted",
        f"Account '{existing.get('username')}' was deleted.",
        actor=claims,
        target=str(existing.get("username")),
        request=request,
        severity="warning",
    )
    return {"message": f"Account '{existing.get('username')}' deleted."}


@router.get("/attendance")
def attendance_overview(
    _: AdminClaims,
    date_from: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    section: Optional[str] = None,
):
    """Attendance across every class for a date range, summarised per day."""
    end = date_to or date.today().isoformat()
    start = date_from or end
    records = get_attendance_in_range(start, end, section)

    per_day: dict = {}
    for record in records:
        bucket = per_day.setdefault(
            record.get("class_date"),
            {"date": record.get("class_date"), "total": 0, "present": 0, "late": 0, "absent": 0},
        )
        bucket["total"] += 1
        status = record.get("status")
        if status == "late":
            bucket["late"] += 1
        elif status == "absent":
            bucket["absent"] += 1
        else:
            bucket["present"] += 1

    return {
        "records": records,
        "daily": sorted(per_day.values(), key=lambda item: str(item["date"]), reverse=True),
        "total": len(records),
        "date_from": start,
        "date_to": end,
    }


@router.get("/audit-log")
def audit_log(
    _: AdminClaims,
    limit: int = Query(default=100, le=500),
    action: Optional[str] = None,
    search: Optional[str] = None,
):
    if not _accounts_available():
        raise HTTPException(
            status_code=503,
            detail="Audit history needs Supabase. Configure SUPABASE_URL and SUPABASE_SERVICE_KEY, then run the audit schema.",
        )

    try:
        events = list_audit_events(limit=limit, action=action, search=search)
    except DatabaseUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail="The audit_log table is missing. Run backend/database/admin_schema.sql in the Supabase SQL Editor.",
        ) from error

    actions = sorted({event.get("action") for event in events if event.get("action")})
    return {"events": events, "actions": actions, "count": len(events)}


@router.delete("/audit-log")
def reset_audit_log(
    request: Request,
    claims: AdminClaims,
    _: dict = Depends(require_password_confirmation),
):
    """Erase the audit trail, gated on the administrator's own password.

    The wipe is itself recorded afterwards, so a cleared log is never a silent
    gap in the history.
    """
    if not _accounts_available():
        raise HTTPException(
            status_code=503,
            detail="Audit history needs Supabase. Configure SUPABASE_URL and SUPABASE_SERVICE_KEY, then run the audit schema.",
        )

    try:
        deleted = clear_audit_log()
    except DatabaseUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail="The audit_log table is missing. Run backend/database/admin_schema.sql in the Supabase SQL Editor.",
        ) from error

    record_audit(
        "audit_log_reset",
        f"Administrator '{claims.get('sub')}' cleared {deleted} audit log event(s).",
        actor=claims,
        target="audit_log",
        request=request,
        severity="critical",
    )
    logger.warning("Audit log cleared by '%s' (%s events removed)", claims.get("sub"), deleted)

    return {
        "success": True,
        "deleted_count": deleted,
        "message": f"Audit log reset. {deleted} event(s) were deleted.",
    }
