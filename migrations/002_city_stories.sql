-- mikeos-storyteller-cloud: ASYNC deep city-story engine (MikeStoryteller app backend).
--
-- Adds an async job queue + a SHARED cache of deep, fact-grounded CITY STORIES.
-- A city story is generated ONCE per (city_key, theme, minutes) via a
-- multi-source, outline->chapter pipeline grounded in wiki.osmike.com, and then
-- reused across ALL users (city_stories is NOT user-scoped). Jobs ARE
-- user-scoped (who asked), so a user's library feed can be built.
--
-- Applied idempotently on startup (tracked in storyteller._migrations).
--
-- House rules honoured:
--   - No reserved SQL keyword is used as a column name.
--   - Idempotent (IF NOT EXISTS).
--   - Timestamps are timestamptz; clients send/receive ISO-8601.

-- ---------------------------------------------------------------------------
-- city_stories: the SHARED, cached deep story for a (city, theme, length).
-- Generated once, reused across users. `chapters` holds the ordered chapters:
--   [{index, title, text, est_seconds, source_url}]
-- `sources` is the list of wiki.osmike.com article URLs it was grounded in.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS storyteller.city_stories (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    city         text NOT NULL,          -- human city name as requested
    city_key     text NOT NULL,          -- normalized key (lowercased, accent-stripped)
    theme        text NOT NULL,          -- people|history|art|architecture|nature|food|complete
    minutes      int  NOT NULL,          -- target spoken length in minutes
    title        text,                   -- the story's title
    chapters     jsonb NOT NULL,         -- [{index, title, text, est_seconds, source_url}]
    total_words  int,                    -- total words across all chapters
    sources      jsonb,                  -- [urls] the story was grounded in
    source_book  text,                   -- the wiki book/slug used (provenance)
    model        text,                   -- the model that generated it
    created_at   timestamptz DEFAULT now()
);

-- One shared story per (city_key, theme, minutes) — the cache key.
CREATE UNIQUE INDEX IF NOT EXISTS idx_city_stories_key
    ON storyteller.city_stories (city_key, theme, minutes);
CREATE INDEX IF NOT EXISTS idx_city_stories_city
    ON storyteller.city_stories (city_key);
CREATE INDEX IF NOT EXISTS idx_city_stories_created
    ON storyteller.city_stories (created_at DESC);

-- ---------------------------------------------------------------------------
-- story_jobs: the async job queue. One row per request to generate/fetch a
-- city story. status: queued|generating|ready|failed. On ready, story_id links
-- the produced (or cache-hit) city_stories row.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS storyteller.story_jobs (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      text NOT NULL,          -- who requested it (for the library feed)
    city         text NOT NULL,
    city_key     text NOT NULL,
    theme        text NOT NULL,
    minutes      int  NOT NULL,
    lat          double precision,
    lon          double precision,
    status       text NOT NULL DEFAULT 'queued',  -- queued|generating|ready|failed
    story_id     uuid,                   -- -> city_stories.id once ready
    error        text,
    created_at   timestamptz DEFAULT now(),
    updated_at   timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_story_jobs_status
    ON storyteller.story_jobs (status, created_at);
CREATE INDEX IF NOT EXISTS idx_story_jobs_user
    ON storyteller.story_jobs (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_story_jobs_key
    ON storyteller.story_jobs (city_key, theme, minutes);
