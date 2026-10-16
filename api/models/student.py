from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class StudentBase(BaseModel):
    student_id: str = Field(..., description="Unique student identification number/code", example="STU-001")
    full_name: str = Field(..., description="Full legal name of the student", example="Maria Santos")
    section: Optional[str] = Field(None, description="Classroom section", example="Section A")
    grade_level: Optional[str] = Field(None, description="Grade level", example="Grade 10")
    photo_url: Optional[str] = Field(None, description="Avatar photo URL")

class StudentCreate(StudentBase):
    teacher_id: Optional[int] = Field(None, description="ID of the teacher who manages this student")


class StudentUpdate(BaseModel):
    student_id: Optional[str] = None
    full_name: Optional[str] = None
    section: Optional[str] = None
    grade_level: Optional[str] = None
    photo_url: Optional[str] = None
    teacher_id: Optional[int] = None

class StudentResponse(StudentBase):
    id: int
    teacher_id: Optional[int] = None
    has_face: bool = False
    gesture_enrolled: bool = False
    face_storage_path: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True

