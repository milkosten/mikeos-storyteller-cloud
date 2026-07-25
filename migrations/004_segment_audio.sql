-- mikeos-storyteller-cloud: per-SEGMENT spoken AUDIO layer (road-trip stories).
--
-- The road-trip analogue of 003's chapter_audio. Road-trip stories live in the
-- per-user `stories` table (segments jsonb, not chapters), so their TTS synth
-- state cannot go in chapter_audio (whose FK targets city_stories). This adds a
-- parallel table keyed on (story_id, segment_order, lang), FK -> stories.
--
-- Same model as chapter_audio: the MP3 bytes are NOT stored here — tts.osmike.com
-- caches by sha256(text+voice) on unlimited disk and the audio proxy re-fetches
-- (cache hit -> near-instant). We only track synth STATE + size/duration so the
-- app can report audio_ready. The proxy does NOT depend on this row existing;
-- it synthesizes on demand and writes the row best-effort.
--
-- House rules honoured:
--   - No reserved SQL keyword is used as a column name (`bytes`->`byte_len`,
--     `order`->`segment_order`).
--   - Idempotent (IF NOT EXISTS).
--   - Timestamps are timestamptz.
--   - Applied once, tracked in storyteller._migrations.

CREATE TABLE IF NOT EXISTS storyteller.segment_audio (
    story_id       uuid NOT NULL REFERENCES storyteller.stories(id) ON DELETE CASCADE,
    segment_order  int  NOT NULL,
    lang           text NOT NULL DEFAULT 'en',
    byte_len       int,                     -- size of the produced MP3 in bytes
    duration_sec   int,                     -- estimated spoken duration
    status         text NOT NULL DEFAULT 'pending',  -- pending|ready|failed
    error          text,
    updated_at     timestamptz DEFAULT now(),
    PRIMARY KEY (story_id, segment_order, lang)
);

CREATE INDEX IF NOT EXISTS idx_segment_audio_story
    ON storyteller.segment_audio (story_id);
