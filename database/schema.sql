-- ==============================================================================
-- SmartCCTV Database Schema (Supabase PostgreSQL)
-- ==============================================================================

-- Enable vector extension for facial embeddings (FaceNet 128-dimensional vectors)
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Students Table
CREATE TABLE IF NOT EXISTS students (
    id BIGSERIAL PRIMARY KEY,
    student_id TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    section TEXT,
    grade_level TEXT,
    photo_url TEXT,
    face_storage_path TEXT,
    has_face BOOLEAN DEFAULT FALSE,
    gesture_enrolled BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE students ADD COLUMN IF NOT EXISTS face_storage_path TEXT;
ALTER TABLE students ADD COLUMN IF NOT EXISTS gesture_enrolled BOOLEAN DEFAULT FALSE;

-- Private bucket for cropped enrollment face images. The backend service key uploads to it.
INSERT INTO storage.buckets (id, name, public)
VALUES ('face-enrollments', 'face-enrollments', FALSE)
ON CONFLICT (id) DO NOTHING;

-- 2. Face Embeddings Table (128-dim vectors from FaceNet)
CREATE TABLE IF NOT EXISTS face_embeddings (
    id BIGSERIAL PRIMARY KEY,
    student_id BIGINT REFERENCES students(id) ON DELETE CASCADE,
    embedding vector(128),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Attendance Table
CREATE TABLE IF NOT EXISTS attendance (
    id BIGSERIAL PRIMARY KEY,
    student_id BIGINT REFERENCES students(id) ON DELETE CASCADE,
    class_date DATE NOT NULL,
    check_in_time TIMESTAMPTZ,
    check_out_time TIMESTAMPTZ,
    status TEXT DEFAULT 'present' CHECK (status IN ('present', 'late', 'time_out')),
    confidence FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(student_id, class_date)
);

-- Attendance is created only when a student checks in. Remove legacy absent
-- rows and restrict stored statuses to the two time-derived values.
DELETE FROM attendance WHERE status = 'absent';
ALTER TABLE attendance DROP CONSTRAINT IF EXISTS attendance_status_check;
ALTER TABLE attendance
    ADD CONSTRAINT attendance_status_check CHECK (status IN ('present', 'late'));

-- 4. Alerts Table (Weapon detections, trespassers, unknown faces)
CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    type TEXT NOT NULL,
    student_id BIGINT REFERENCES students(id) ON DELETE SET NULL,
    image_url TEXT,
    description TEXT NOT NULL,
    severity TEXT DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    is_resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Compliance Violations Table
CREATE TABLE IF NOT EXISTS compliance_violations (
    id BIGSERIAL PRIMARY KEY,
    student_id BIGINT REFERENCES students(id) ON DELETE SET NULL,
    violation_type TEXT,
    image_url TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 6. Users Table (System Administrators / Faculty Teachers)
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'teacher' CHECK (role = 'teacher'),
    registration_code TEXT NOT NULL DEFAULT 'TEACHER2026',
    is_active BOOLEAN DEFAULT TRUE,
    last_login_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(class_date);
CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id);
CREATE INDEX IF NOT EXISTS idx_alerts_unresolved ON alerts(is_resolved) WHERE is_resolved = FALSE;
CREATE INDEX IF NOT EXISTS idx_students_student_id ON students(student_id);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);

-- Enable Supabase Realtime for attendance, alerts, and users
ALTER PUBLICATION supabase_realtime ADD TABLE attendance;
ALTER PUBLICATION supabase_realtime ADD TABLE alerts;

