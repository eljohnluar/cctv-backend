from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime

class AttendanceManualMark(BaseModel):
    student_id: int = Field(..., description="ID of student to mark")
    confidence: Optional[float] = Field(default=1.0, description="Confidence score")
    check_in_time: Optional[datetime] = Field(
        default=None,
        description="Optional check-in timestamp. Status is calculated from this time.",
    )

class AttendanceRecord(BaseModel):
    id: int
    student_id: int
    student_name: str
    student_code: str
    section: Optional[str] = None
    status: str
    check_in_time: Optional[str] = None
    check_out_time: Optional[str] = None
    confidence: Optional[float] = None
    class_date: str

class AttendanceStats(BaseModel):
    total: int
    present: int
    late: int
    time_out: int
    rate: float

class TodayAttendanceResponse(BaseModel):
    date: str
    stats: AttendanceStats
    records: List[AttendanceRecord]
