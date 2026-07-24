# Deploying mikeos-storyteller-cloud

FastAPI + asyncpg + Postgres, **nixpacks-only** (no Dockerfile/Procfile, never
hardcode `$PORT`). Self-migrating from `migrations/*.sql`.

## Railway (workspace SaaSRyan)

```bash
cd mikeos-storyteller-cloud
railway init -n mikeos-storyteller-cloud
railway add --database postgres --json          # note the "Postgres-XXXX" service name
railway add -s mikeos-storyteller-cloud -r milkosten/mikeos-storyteller-cloud --branch main --json
# grab the free-GPU cred from a service that already has it:
railway variables -s mikeos-photos-cloud -e production | grep OLLAMA_GPU_URL
railway variables --set 'DATABASE_URL=${{Postgres-XXXX.DATABASE_URL}}' --set 'OLLAMA_GPU_URL=<value>' -s mikeos-storyteller-cloud -e production
railway domain -s mikeos-storyteller-cloud -e production   # public URL
```

Auto-deploys from GitHub + on the var change. Health comes up ~2-4 min later;
poll `GET {url}/api/health` until 200.

## Env vars

- `DATABASE_URL` (required) — Railway Postgres.
- `OLLAMA_GPU_URL` (required for GPU generation) — the free shared GPU
  (`ollama://user:pass@host:port`). Without it, stories use the deterministic
  fallback template (never a paid API).
- `OLLAMA_TEXT_MODEL` (optional) — defaults to `qwen3:8b`.
- `MIKEOSCOMPUTERS_URL` (optional) — IdP; defaults to production.
- `STORYTELLER_SCHEMA` (optional) — defaults to `storyteller`.
