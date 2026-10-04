import csv
import io
from collections import defaultdict
from datetime import date, timedelta
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from api.middleware.auth import get_optional_account
from database.queries import get_all_students, get_attendance_in_range
from utils.sections import resolve_allowed_sections

router = APIRouter(prefix="/reports", tags=["Reports"])

# A time out row is still an attended day: the student checked in and stayed
# past the configured Time out.
ATTENDED_STATUSES = {"present", "late", "time_out"}


def _parse_date_range(date_from: Optional[str], date_to: Optional[str]) -> Tuple[date, date]:
    try:
        end = date.fromisoformat(date_to) if date_to else date.today()
        start = date.fromisoformat(date_from) if date_from else end - timedelta(days=7)
    except ValueError as error:
        raise HTTPException(status_code=422, detail="dateFrom and dateTo must use YYYY-MM-DD.") from error
    if start > end:
        raise HTTPException(status_code=422, detail="dateFrom cannot be after dateTo.")
    return start, end


def _visible_students(claims: Optional[dict], section: Optional[str]):
    students = get_all_students(section or None)
    allowed = resolve_allowed_sections(claims)
    if allowed is not None:
        students = [student for student in students if student.get("section") in allowed]
    return students


def _records_for_range(
    date_from: Optional[str],
    date_to: Optional[str],
    section: Optional[str],
    claims: Optional[dict] = None,
):
    start, end = _parse_date_range(date_from, date_to)
    records = get_attendance_in_range(start.isoformat(), end.isoformat(), section or None)
    allowed = resolve_allowed_sections(claims)
    if allowed is not None:
        records = [record for record in records if record.get("section") in allowed]
    return start, end, records


@router.get("/summary")
def get_summary(
    date_from: Optional[str] = Query(None, alias="dateFrom"),
    date_to: Optional[str] = Query(None, alias="dateTo"),
    section: Optional[str] = None,
    claims: Optional[dict] = Depends(get_optional_account),
):
    """Get attendance metrics calculated from database records."""
    start, end, records = _records_for_range(date_from, date_to, section, claims)
    total_students = len(_visible_students(claims, section))
    present = sum(record["status"] == "present" for record in records)
    late = sum(record["status"] == "late" for record in records)
    time_out = sum(record["status"] == "time_out" for record in records)

    daily_rates = []
    records_by_date = defaultdict(list)
    for record in records:
        records_by_date[record["class_date"]].append(record)
    for daily_records in records_by_date.values():
        attended = sum(record["status"] in ATTENDED_STATUSES for record in daily_records)
        daily_rates.append((attended / total_students) * 100 if total_students else 0)

    return {
        "avg_rate": round(sum(daily_rates) / len(daily_rates), 1) if daily_rates else 0,
        "total_present": present,
        "total_late": late,
        "total_time_out": time_out,
        "total_students": total_students,
        "date_from": start.isoformat(),
        "date_to": end.isoformat(),
        "section": section or "All Sections",
    }


@router.get("/trend")
def get_trend(
    date_from: Optional[str] = Query(None, alias="dateFrom"),
    date_to: Optional[str] = Query(None, alias="dateTo"),
    section: Optional[str] = None,
    claims: Optional[dict] = Depends(get_optional_account),
):
    """Get daily attendance-rate data calculated from the database."""
    start, end, records = _records_for_range(date_from, date_to, section, claims)
    total_students = len(_visible_students(claims, section))
    records_by_date = defaultdict(list)
    for record in records:
        records_by_date[record["class_date"]].append(record)

    trend = []
    current_date = start
    while current_date <= end:
        daily_records = records_by_date[current_date.isoformat()]
        attended = sum(record["status"] in ATTENDED_STATUSES for record in daily_records)
        rate = round((attended / total_students) * 100, 1) if total_students else 0
        trend.append({"date": current_date.strftime("%a %b %d"), "rate": rate})
        current_date += timedelta(days=1)
    return trend


@router.get("/records")
def get_records(
    date_from: Optional[str] = Query(None, alias="dateFrom"),
    date_to: Optional[str] = Query(None, alias="dateTo"),
    section: Optional[str] = None,
    claims: Optional[dict] = Depends(get_optional_account),
):
    """Return detailed attendance rows for the selected report range."""
    _, _, records = _records_for_range(date_from, date_to, section, claims)
    return records


@router.get("/export/csv")
def export_csv(
    date_from: Optional[str] = Query(None, alias="dateFrom"),
    date_to: Optional[str] = Query(None, alias="dateTo"),
    section: Optional[str] = None,
    claims: Optional[dict] = Depends(get_optional_account),
):
    """Export attendance rows from the database as CSV."""
    start, end, records = _records_for_range(date_from, date_to, section, claims)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Student ID", "Student Name", "Section", "Status", "Check-in Time", "Time Out", "Confidence"])
    for record in records:
        confidence = f"{record['confidence'] * 100:.1f}%" if record["confidence"] is not None else ""
        writer.writerow([
            record["class_date"], record["student_code"], record["student_name"], record["section"],
            record["status"], record["check_in_time"] or "", record.get("check_out_time") or "", confidence,
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_report_{start}_to_{end}.csv"},
    )
