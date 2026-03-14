-- FAKE-SHA: analysis_records table for Supabase (PostgreSQL)
-- Run this in the Supabase SQL Editor to create the table.
-- https://supabase.com/dashboard -> SQL Editor -> New query

CREATE TABLE IF NOT EXISTS analysis_records (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    title           TEXT NOT NULL DEFAULT '',
    url             TEXT NOT NULL DEFAULT '',
    text            TEXT NOT NULL DEFAULT '',
    verdict         TEXT NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    summary         TEXT NOT NULL DEFAULT '',
    indicators      JSONB NOT NULL DEFAULT '[]',
    mode            TEXT NOT NULL DEFAULT 'selection_only',
    extraction_source TEXT
);

-- Optional: enable Row Level Security (RLS) if you add auth later
-- ALTER TABLE analysis_records ENABLE ROW LEVEL SECURITY;

-- Optional: create an index for common queries (e.g. by verdict or date)
-- CREATE INDEX idx_analysis_records_verdict ON analysis_records(verdict);
-- CREATE INDEX idx_analysis_records_created_at ON analysis_records(created_at DESC);
