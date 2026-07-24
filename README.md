# mikeos-storyteller-cloud

The MikeOS **storyteller** cloud (MikeStoryteller's brain): it turns an ordered
list of **POIs** — the `route.pois` MikeGuide produces along a route — into a
**duration-matched, spoken-word road-trip story** generated on the **FREE GPU**
(Ollama `qwen3:8b`), and keeps a **per-user library** of the stories it tells.

FastAPI + asyncpg + Postgres, nixpacks-only, self-migrating. Every story is
scoped to the `user_id` resolved from the caller's hive agent key (`X-API-KEY`)
via mikeoscomputers (the IdP).

## How it paces a story

- Target spoken length = `minutes` at ~**150 words/min** → `total_words = minutes*150`.
- The per-POI word budget is distributed **proportional to each POI's `dwell`**
  weight — a `dwell=3.0` landmark gets ~3× the words of a `dwell=1.0` one — with
  a small intro + outro reserve and a per-POI floor so nothing is silent.
- The narration is generated on the GPU in **chunks** (a few POIs per call) so no
  single call is huge (a 30-min story is ~4500 words), and returned as
  **segments**: `[{order, poi_name, text, est_seconds}]` where
  `est_seconds ≈ words/150*60`, one per POI in drive-past order.
- **Never-trust-the-GPU:** an empty/unparseable GPU reply is retried once, then
  falls back to a deterministic template (marked `model:"fallback"`). The
  endpoint never returns an empty `segments` array with 200.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET  | `/api/health` | no | `{status, db, gpu, version}` |
| POST | `/api/story` | key | generate (or cache-hit) a story |
| GET  | `/api/story/{id}` | key | one story (scoped to the user) |
| GET  | `/api/stories?limit=20` | key | the user's story library |

`POST /api/story` body:
```json
{"trip_id": "abc", "dest": "Nice", "minutes": 10,
 "pois": [{"name": "...", "cat": "...", "order": 0, "dwell": 2.0, "blurb": "..."}]}
```
Returns `{story_id, trip_id, dest, minutes, poi_count, total_words, model, segments, cached}`.
When `trip_id` is given and a story for `(user_id, trip_id, minutes)` already
exists, it is returned with `"cached": true`.

## Run locally

```bash
pip install -r requirements.txt
export DATABASE_URL=postgresql://user:pass@localhost:5432/mikeos_storyteller
export OLLAMA_GPU_URL=ollama://user:pass@host:port   # free GPU; else fallback template
uvicorn server.http_server:app --host 0.0.0.0 --port 8000
```

## Deploy

Railway (workspace SaaSRyan), nixpacks-only. See `DEPLOY.md`.

## House rules honoured

Parameterised SQL only; tolerant timestamp parsing; idempotent migrations; no
reserved-keyword columns; **no paid APIs** (free GPU only, deterministic fallback
otherwise); never-trust-200 / never-trust-GPU (segments are never empty on a 200).
