-- mikeos-storyteller-cloud: core schema (per-user road-trip story library).
-- Applied idempotently on startup (tracked in storyteller._migrations).
--
-- A per-user library of duration-matched, spoken-word ROAD-TRIP STORIES. Each
-- story is generated on the FREE GPU (qwen3:8b) from an ORDERED list of POIs
-- (the route.pois MikeGuide produces along a route) and paced to a target
-- spoken length (minutes * ~150 words/min), with the per-POI word budget
-- weighted by each POI's dwell. The `segments` jsonb holds one narrated segment
-- per POI, in drive-past order.
--
-- Every row is scoped to the user_id resolved from the caller's hive agent key
-- (X-API-KEY).
--
-- NOTE: no reserved SQL keyword is used as a column name.

-- ---------------------------------------------------------------------------
-- stories: one generated road-trip story for a trip.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS storyteller.stories (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      text NOT NULL,
    trip_id      text,                 -- the MikeMaps/MikeGuide trip this narrates (nullable)
    dest         text,                 -- human destination name, if known
    minutes      int NOT NULL,         -- target spoken length in minutes
    poi_count    int,                  -- number of POIs narrated
    segments     jsonb NOT NULL,       -- [{order, poi_name, text, est_seconds}]
    total_words  int,                  -- total words across all segments
    model        text,                 -- the model that generated it ('qwen3:8b' or 'fallback')
    created_at   timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_stories_user_created
    ON storyteller.stories (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_stories_trip
    ON storyteller.stories (trip_id);
