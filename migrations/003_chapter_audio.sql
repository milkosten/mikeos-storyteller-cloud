-- mikeos-storyteller-cloud: per-chapter spoken AUDIO layer (MikeStoryteller app).
--
-- Adds a small table tracking the TTS-synth state of each chapter of a shared
-- city story. The worker PRE-WARMS the tts.osmike.com cache after a story is
-- marked ready (a bonus — a tts hiccup must never fail the story), and records
-- the per-chapter result here so `GET /api/story/{id}` can report `audio_ready`.
--
-- The MP3 bytes themselves are NOT stored here — tts.osmike.com caches by
-- sha256(text+voice) on unlimited disk, and the audio proxy re-fetches (cache
-- hit -> near-instant). We only track synth STATE + size/duration for reporting.
--
-- Also adds a `lang` column to city_stories (default 'en') so a story's language
-- is threaded through to the TTS call. Existing rows default to 'en'.
--
-- House rules honoured:
--   - No reserved SQL keyword is used as a column name (`bytes`->`byte_len`).
--   - Idempotent (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).
--   - Timestamps are timestamptz.
--   - Applied once, tracked in storyteller._migrations.

-- Language of a shared city story (drives the TTS voice/lang). Default 'en'.
ALTER TABLE storyteller.city_stories
    ADD COLUMN IF NOT EXISTS lang text NOT NULL DEFAULT 'en';

-- ---------------------------------------------------------------------------
-- chapter_audio: one row per (story_id, chapter_index, lang) TTS synth result.
-- status: pending | ready | failed. `byte_len` = MP3 size, `duration_sec` =
-- estimated spoken length. Written by the worker's pre-warm pass; read by the
-- status/get-story endpoints. The proxy does NOT depend on this row existing.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS storyteller.chapter_audio (
    story_id       uuid NOT NULL REFERENCES storyteller.city_stories(id) ON DELETE CASCADE,
    chapter_index  int  NOT NULL,
    lang           text NOT NULL DEFAULT 'en',
    byte_len       int,                     -- size of the produced MP3 in bytes
    duration_sec   int,                     -- estimated spoken duration
    status         text NOT NULL DEFAULT 'pending',  -- pending|ready|failed
    error          text,
    updated_at     timestamptz DEFAULT now(),
    PRIMARY KEY (story_id, chapter_index, lang)
);

CREATE INDEX IF NOT EXISTS idx_chapter_audio_story
    ON storyteller.chapter_audio (story_id);
