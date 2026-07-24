"""PostgreSQL storage manager for mikeos-storyteller-cloud.

Self-provisioning: reads a single DATABASE_URL env var (Railway auto-provides
it), creates the schema + a _migrations tracking table, and applies
migrations/*.sql idempotently on startup. No admin key required.

Holds the per-user library of generated road-trip STORIES, scoped by user_id.
Each row keeps the paced `segments` (one narrated segment per POI, in drive-past
order) as jsonb, plus the pacing metadata (minutes, poi_count, total_words) and
which model produced it.

House rules honoured:
  - Timestamps are parsed TOLERANTLY (a helper is provided) so a client that
    ever sends a raw Long never silently drops.
  - Parameterised queries only; never string-interpolate values into SQL.
  - No reserved keyword is used as a column name.
  - Never trust 200: writes RETURNING the row / a real id.
"""
import glob
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import asyncpg

logger = logging.getLogger(__name__)

# Schema everything lives under (keeps the public schema clean).
SCHEMA = os.environ.get("STORYTELLER_SCHEMA", "storyteller")


def _parse_ts(value: Union[str, int, float, None]) -> Optional[datetime]:
    """Tolerantly parse a timestamp into an aware UTC datetime, or None.

    Accepts None, epoch int/float (>=1e12 treated as ms), ISO-8601 strings
    (trailing 'Z' tolerated), and numeric strings. Anything unparseable returns
    None so the caller can decide rather than dropping the whole write.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        num: Optional[float] = float(value)
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            num = float(s)
        except ValueError:
            num = None
        if num is None:
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError:
                logger.warning("unparseable ts %r; skipping", value)
                return None
    else:
        return None
    if num >= 1e12:
        num = num / 1000.0
    try:
        return datetime.fromtimestamp(num, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        logger.warning("out-of-range ts %r; skipping", value)
        return None


class PostgreSQLManager:
    """Manages all PostgreSQL operations for the story library."""

    def __init__(self, schema: str = SCHEMA):
        self.schema = schema
        self._pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize the connection pool from DATABASE_URL."""
        dsn = os.environ.get("DATABASE_URL")
        if not dsn:
            raise RuntimeError(
                "DATABASE_URL is not set. mikeos-storyteller-cloud needs a Postgres "
                "database at runtime. On Railway: add a Postgres plugin and set "
                "DATABASE_URL=${{Postgres.DATABASE_URL}}."
            )
        logger.info("PostgreSQL: connecting via DATABASE_URL")
        self._pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=10,
            command_timeout=60,
            server_settings={"search_path": f"{self.schema},public"},
        )
        async with self._pool.acquire() as conn:
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
        logger.info("Database connection pool initialized (schema: %s)", self.schema)

    async def run_migrations(self):
        """Apply migrations/*.sql in order, tracked in <schema>._migrations."""
        mig_dir = os.path.join(os.path.dirname(__file__), "..", "..", "migrations")
        files = sorted(glob.glob(os.path.join(mig_dir, "*.sql")))
        if not files:
            logger.warning("No migration files found at %s", mig_dir)
            return
        async with self._pool.acquire() as conn:
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            await conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.schema}._migrations "
                "(filename text PRIMARY KEY, applied_at timestamptz DEFAULT now())"
            )
            applied = {
                r["filename"]
                for r in await conn.fetch(
                    f"SELECT filename FROM {self.schema}._migrations"
                )
            }
            for path in files:
                name = os.path.basename(path)
                if name in applied:
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    sql = f.read()
                async with conn.transaction():
                    await conn.execute(sql)
                    await conn.execute(
                        f"INSERT INTO {self.schema}._migrations(filename) VALUES($1)",
                        name,
                    )
                logger.info("migration applied: %s", name)

    async def get_connection(self) -> asyncpg.Pool:
        if not self._pool:
            await self.initialize()
        return self._pool

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection pool closed")

    async def ping(self) -> bool:
        try:
            pool = await self.get_connection()
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error("Database ping failed: %s", e)
            return False

    # =========================================================================
    # Stories
    # =========================================================================

    async def find_cached_story(
        self, user_id: str, trip_id: str, minutes: int
    ) -> Optional[Dict[str, Any]]:
        """Most recent story for (user_id, trip_id, minutes), or None."""
        pool = await self.get_connection()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT * FROM {self.schema}.stories
                WHERE user_id = $1 AND trip_id = $2 AND minutes = $3
                ORDER BY created_at DESC
                LIMIT 1
                """,
                user_id,
                trip_id,
                int(minutes),
            )
        return self._row_to_story(row) if row else None

    async def create_story(
        self,
        user_id: str,
        *,
        trip_id: Optional[str],
        dest: Optional[str],
        minutes: int,
        poi_count: int,
        segments: List[Dict[str, Any]],
        total_words: int,
        model: str,
    ) -> Dict[str, Any]:
        """Insert a generated story; returns the saved row (never-trust-200)."""
        pool = await self.get_connection()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                INSERT INTO {self.schema}.stories
                    (user_id, trip_id, dest, minutes, poi_count,
                     segments, total_words, model)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8)
                RETURNING *
                """,
                user_id,
                trip_id,
                dest,
                int(minutes),
                int(poi_count),
                json.dumps(segments),
                int(total_words),
                model,
            )
        return self._row_to_story(row)

    async def get_story(
        self, user_id: str, story_id: str
    ) -> Optional[Dict[str, Any]]:
        """One story (scoped to the user), or None."""
        pool = await self.get_connection()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self.schema}.stories "
                "WHERE id = $1::uuid AND user_id = $2",
                story_id,
                user_id,
            )
        return self._row_to_story(row) if row else None

    async def list_stories(
        self, user_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """The user's story library (lightweight rows), newest first."""
        pool = await self.get_connection()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, dest, trip_id, minutes, poi_count, total_words,
                       model, created_at
                FROM {self.schema}.stories
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                user_id,
                max(1, min(limit, 200)),
            )
        return [
            {
                "id": str(r["id"]),
                "dest": r["dest"],
                "trip_id": r["trip_id"],
                "minutes": r["minutes"],
                "poi_count": r["poi_count"],
                "total_words": r["total_words"],
                "model": r["model"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]

    # =========================================================================
    # Row mappers
    # =========================================================================

    @staticmethod
    def _row_to_story(r) -> Optional[Dict[str, Any]]:
        if r is None:
            return None
        segments = r["segments"]
        if isinstance(segments, str):
            try:
                segments = json.loads(segments)
            except Exception:  # noqa: BLE001
                segments = []
        return {
            "id": str(r["id"]),
            "user_id": r["user_id"],
            "trip_id": r["trip_id"],
            "dest": r["dest"],
            "minutes": r["minutes"],
            "poi_count": r["poi_count"],
            "segments": segments,
            "total_words": r["total_words"],
            "model": r["model"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }


# Singleton instance
postgres_manager: Optional[PostgreSQLManager] = None


async def get_postgres_manager() -> PostgreSQLManager:
    """Get or create the PostgreSQL manager singleton."""
    global postgres_manager
    if postgres_manager is None:
        postgres_manager = PostgreSQLManager()
        await postgres_manager.initialize()
    return postgres_manager
