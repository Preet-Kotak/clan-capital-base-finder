"""
Step 2 of pipeline: Upload layouts.json → Supabase

1. Adds layout_json and processed_at columns to `bases` table if they don't exist yet.
2. Reads data/layouts/layouts.json and bulk-updates each row by base ID.

Usage: python scripts/upload_layouts.py
Resume safe — skips rows that already have layout_json set.
"""

import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import os

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# ─────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

TABLE_NAME   = "bases"
LAYOUTS_FILE = Path("data/layouts/layouts.json")
BATCH_SIZE   = 50
# ─────────────────────────────────────────────


def add_columns_if_missing(supabase: Client) -> None:
    """
    Run ALTER TABLE ... ADD COLUMN IF NOT EXISTS for layout_json and processed_at.
    Supabase exposes a /rest/v1/rpc endpoint — we use the postgres function approach
    via raw SQL through the Supabase Management API isn't available on anon key,
    so we do a lightweight probe instead: try selecting the columns.
    If the select raises a 'column does not exist' error we surface a clear message.
    The actual ALTER TABLE must be run once in the Supabase SQL editor:

        ALTER TABLE bases ADD COLUMN IF NOT EXISTS layout_json    jsonb;
        ALTER TABLE bases ADD COLUMN IF NOT EXISTS processed_at   timestamptz;

    This function checks whether that has been done and exits early if not.
    """
    print("Checking Supabase columns...")
    try:
        supabase.table(TABLE_NAME).select("id, layout_json, processed_at").limit(1).execute()
        print("  ✓ Columns layout_json and processed_at are present.")
    except Exception as e:
        err = str(e)
        if "layout_json" in err or "processed_at" in err or "column" in err.lower():
            print("\n  ✗ Columns are missing. Run this SQL in the Supabase SQL editor:\n")
            print("      ALTER TABLE bases ADD COLUMN IF NOT EXISTS layout_json  jsonb;")
            print("      ALTER TABLE bases ADD COLUMN IF NOT EXISTS processed_at timestamptz;\n")
            sys.exit(1)
        # unexpected error — re-raise
        raise


def fetch_already_uploaded(supabase: Client) -> set[int]:
    """Return set of base IDs that already have layout_json populated."""
    print("Fetching already-uploaded IDs...")
    uploaded: set[int] = set()
    page_size = 1000
    offset    = 0

    while True:
        resp = (
            supabase.table(TABLE_NAME)
            .select("id")
            .not_.is_("layout_json", "null")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data
        if not rows:
            break
        for r in rows:
            uploaded.add(int(r["id"]))
        if len(rows) < page_size:
            break
        offset += page_size

    print(f"  Already uploaded: {len(uploaded)} rows")
    return uploaded


def upload_batch(supabase: Client, batch: list[dict]) -> int:
    """Update layout_json and processed_at for each row by id. Returns number of rows written."""
    for row in batch:
        supabase.table(TABLE_NAME).update({
            "layout_json":  row["layout_json"],
            "processed_at": row["processed_at"],
        }).eq("id", row["id"]).execute()
    return len(batch)


def main():
    print("=" * 55)
    print("  Upload Layouts → Supabase")
    print("=" * 55)

    # ── load layouts.json ──────────────────────────────
    if not LAYOUTS_FILE.exists():
        print(f"\n✗ {LAYOUTS_FILE} not found. Run scripts/extract_all.py first.")
        sys.exit(1)

    with open(LAYOUTS_FILE, "r") as f:
        layouts: dict = json.load(f)

    print(f"\nLoaded {len(layouts)} layouts from {LAYOUTS_FILE}")

    # ── connect ────────────────────────────────────────
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # ── check columns exist ────────────────────────────
    add_columns_if_missing(supabase)

    # ── skip already-uploaded ──────────────────────────
    already_done = fetch_already_uploaded(supabase)

    to_upload = {
        base_id: data
        for base_id, data in layouts.items()
        if int(base_id) not in already_done
    }

    print(f"\nTo upload:   {len(to_upload)}")
    print(f"Already done:{len(already_done)}")

    if not to_upload:
        print("\nAll layouts already in Supabase. Nothing to do.")
        return

    # ── build rows and upload in batches ───────────────
    now_ts   = datetime.now(timezone.utc).isoformat()
    rows     = list(to_upload.items())
    total    = len(rows)
    uploaded = 0
    failed   = 0
    start    = time.time()

    print(f"\nUploading in batches of {BATCH_SIZE}...\n")

    for batch_start in range(0, total, BATCH_SIZE):
        batch_items = rows[batch_start : batch_start + BATCH_SIZE]

        batch_payload = []
        for base_id, data in batch_items:
            # store only type/x/y — drop conf to keep DB size small
            buildings_clean = [
                {"type": b["type"], "x": b["x"], "y": b["y"]}
                for b in data.get("buildings", [])
            ]
            batch_payload.append({
                "id":           int(base_id),
                "layout_json":  buildings_clean,
                "processed_at": now_ts,
            })

        try:
            upload_batch(supabase, batch_payload)
            uploaded += len(batch_payload)
            elapsed   = time.time() - start
            per_row   = elapsed / uploaded
            eta_secs  = per_row * (total - uploaded)
            print(
                f"  [{uploaded}/{total}]  "
                f"batch {batch_start // BATCH_SIZE + 1}  |  "
                f"ETA: {eta_secs:.0f}s"
            )
        except Exception as e:
            print(f"  ✗ Batch {batch_start // BATCH_SIZE + 1} failed: {e}")
            failed += len(batch_payload)

    elapsed = time.time() - start
    print(f"\n{'=' * 55}")
    print(f"Done!")
    print(f"  Uploaded: {uploaded}")
    print(f"  Failed:   {failed}")
    print(f"  Time:     {elapsed:.1f}s")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
