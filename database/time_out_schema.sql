-- ============================================================
-- SmartCCTV - Time out tracking migration
-- Apply this to an existing Supabase project (SQL Editor). Safe to re-run.
--
-- Adds a departure stamp to attendance and lets the status column hold the
-- 'time_out' mark, which is recorded when a student is recognised after the
-- Time out configured in Settings (check-in time + attendance timeout).
-- ============================================================

-- 0. Departure stamp. NULL until a sign-out past the Time out is captured.
ALTER TABLE attendance ADD COLUMN IF NOT EXISTS check_out_time TIMESTAMPTZ;

-- 1. Replace the status check so it also allows 'time_out'.
--    'absent' is kept in the list because older databases still hold legacy
--    absent rows; dropping it from the check would fail validation on them.
ALTER TABLE attendance DROP CONSTRAINT IF EXISTS attendance_status_check;
ALTER TABLE attendance ADD CONSTRAINT attendance_status_check
    CHECK (status IN ('present', 'late', 'time_out', 'absent'));

-- 2. Confirm the change.
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'attendance'
ORDER BY ordinal_position;

SELECT conname, pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE conrelid = to_regclass('public.attendance') AND contype = 'c';

SELECT status, count(*) AS rows
FROM attendance
GROUP BY status
ORDER BY rows DESC;

-- ── Optional cleanup ─────────────────────────────────────────────────────────
-- Attendance is meant to exist only when a student actually checks in. If you
-- would rather purge the legacy absent rows (this DELETES data, so run the
-- SELECT in step 2 first and confirm the count), uncomment these lines and run
-- them, then re-run step 1 with ('present', 'late', 'time_out') only.
--
-- DELETE FROM attendance WHERE status = 'absent';
-- ALTER TABLE attendance DROP CONSTRAINT IF EXISTS attendance_status_check;
-- ALTER TABLE attendance ADD CONSTRAINT attendance_status_check
--     CHECK (status IN ('present', 'late', 'time_out'));
