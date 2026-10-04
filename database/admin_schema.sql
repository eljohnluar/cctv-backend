-- ==============================================================================
-- SmartCCTV Administrator role + Audit log migration (Supabase PostgreSQL)
--
-- Run this file in the Supabase SQL Editor AFTER backend/database/schema.sql
-- and backend/database/users_schema.sql. It is safe to re-run.
--
-- What it does:
--   1. Allows the 'admin' role in the users table (was CHECK (role = 'teacher'))
--   2. Creates the audit_log table the Administrator console reads from
--   3. Seeds a default administrator account
--
-- Seeded admin sign-in:  username: admin   password: password123
-- ==============================================================================

-- ── 1. Permit administrators ─────────────────────────────────────────────────
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check;
ALTER TABLE users
    ADD CONSTRAINT users_role_check CHECK (role IN ('teacher', 'admin'));

-- Keep the registration code column honest for both roles
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_registration_code_check;
ALTER TABLE users
    ADD CONSTRAINT users_registration_code_check
    CHECK (registration_code IN ('TEACHER2026', 'ADMIN2026'));

-- ── 2. Row Level Security for the widened role set ───────────────────────────
-- The API connects with the service_role key, so these policies only matter if
-- something queries the table directly with the anon/authenticated key.
DROP POLICY IF EXISTS "Allow registration with teacher code" ON users;
CREATE POLICY "Allow registration with admin code"
    ON users
    FOR INSERT
    TO anon, authenticated
    WITH CHECK (role = 'admin' AND registration_code = 'ADMIN2026');

DROP POLICY IF EXISTS "Allow users update" ON users;
CREATE POLICY "Allow users update"
    ON users
    FOR UPDATE
    TO anon, authenticated
    USING (true)
    WITH CHECK (role IN ('teacher', 'admin'));

-- ── 3. Audit log ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    action TEXT NOT NULL,
    description TEXT,
    actor_username TEXT,
    actor_role TEXT,
    target TEXT,
    severity TEXT NOT NULL DEFAULT 'info' CHECK (severity IN ('info', 'warning', 'critical')),
    ip_address TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor_username);

ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;

-- Only the service role may read or write the audit trail. Deliberately no
-- anon/authenticated policy: audit history is administrator-only data.
DROP POLICY IF EXISTS "Service role full access on audit_log" ON audit_log;
CREATE POLICY "Service role full access on audit_log"
    ON audit_log
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- ── 4. Default administrator account ─────────────────────────────────────────
-- password123 hashed with the API's 'smartcctv_salt_' prefix (SHA-256).
INSERT INTO users (username, email, password_hash, full_name, role, registration_code, is_active)
VALUES (
    'admin',
    'admin@smartcctv.edu',
    '7159e84ba3e9b7b0df87feee43f9495493fa2666abdc606bf303f4de31b053fe',
    'System Administrator',
    'admin',
    'ADMIN2026',
    TRUE
)
ON CONFLICT (username) DO NOTHING;

-- ── 5. Teacher section scope (college year levels) ───────────────────────────
-- A teacher handles a set of year levels crossed with a set of section letters.
-- Empty arrays scope the teacher to nothing, so an account sees no rosters until
-- an administrator assigns at least one year level and one section letter.
ALTER TABLE users ADD COLUMN IF NOT EXISTS year_levels JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE users ADD COLUMN IF NOT EXISTS sections JSONB NOT NULL DEFAULT '[]'::jsonb;

-- ── 6. Convert the roster from high school to college ────────────────────────
-- Grade 7-10 become 1st-4th Year; Grade 11/12 also land in 4th Year.
UPDATE students
SET grade_level = CASE grade_level
    WHEN 'Grade 7'  THEN '1st Year'
    WHEN 'Grade 8'  THEN '2nd Year'
    WHEN 'Grade 9'  THEN '3rd Year'
    WHEN 'Grade 10' THEN '4th Year'
    WHEN 'Grade 11' THEN '4th Year'
    WHEN 'Grade 12' THEN '4th Year'
    ELSE grade_level
  END
WHERE grade_level LIKE 'Grade %';

-- Sections become year-scoped so the 5 sections in each year stay distinct:
-- 'Section A' + '4th Year' -> '4th Year - Section A'.
-- The NOT LIKE guard keeps this safe to re-run.
UPDATE students
SET section = grade_level || ' - ' || section
WHERE section LIKE 'Section %'
  AND section NOT LIKE '% - %'
  AND grade_level IN ('1st Year', '2nd Year', '3rd Year', '4th Year');

-- ── 7. Verify ────────────────────────────────────────────────────────────────
-- Expect one 'admin' row and one 'teacher' row, then a zero-row audit_log.
SELECT username, role, is_active, year_levels, sections FROM users ORDER BY username;
SELECT COUNT(*) AS audit_events FROM audit_log;
SELECT section, grade_level, COUNT(*) FROM students GROUP BY section, grade_level ORDER BY section;
