"""
Download all base screenshots from Supabase + Cloudinary
Organizes them into folders by district, then zips for Roboflow upload.

FILL IN the config section below before running.
"""

import os
import zipfile
import requests
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# ─────────────────────────────────────────────
SUPABASE_URL     = os.getenv("SUPABASE_URL")
SUPABASE_KEY     = os.getenv("SUPABASE_KEY")

TABLE_NAME       = "bases"          # your table name
URL_COLUMN       = "screenshot"      # column with the Cloudinary URL
DISTRICT_COLUMN  = "district_number"       # column with district name e.g. "capital_peak"
ID_COLUMN        = "id"             # any unique identifier column for filename

OUTPUT_DIR = Path("data/screenshots/by_district")  # where to save images
ZIP_OUTPUT = Path("data/screenshots/all_bases.zip") # final zip path
# ─────────────────────────────────────────────

DISTRICT_MAP = {
    0: "capital_peak",
    1: "barbarian_camp",
    2: "wizard_valley",
    3: "balloon_lagoon",
    4: "builders_workshop",
    5: "dragon_cliffs",
    6: "golem_quarry",
    7: "skeleton_park",
    8: "goblin_mines",
}


def fetch_all_rows(supabase: Client) -> list[dict]:
    """Fetch every row from the table, handling Supabase's 1000-row page limit."""
    all_rows = []
    page_size = 1000
    offset = 0

    print(f"Fetching rows from '{TABLE_NAME}'...")
    while True:
        response = (
            supabase.table(TABLE_NAME)
            .select(f"{ID_COLUMN}, {URL_COLUMN}, {DISTRICT_COLUMN}")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = response.data
        if not rows:
            break
        all_rows.extend(rows)
        print(f"  Fetched {len(all_rows)} rows so far...")
        if len(rows) < page_size:
            break  # last page
        offset += page_size

    print(f"Total rows fetched: {len(all_rows)}")
    return all_rows


def download_image(url: str, save_path: Path) -> bool:
    """Download a single image from a URL. Returns True on success."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
        return True
    except requests.RequestException as e:
        print(f"  ✗ Failed to download {url}: {e}")
        return False


def zip_directory(source_dir: Path, zip_path: Path):
    """Zip the entire output directory for Roboflow upload."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nZipping {source_dir} → {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in source_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(source_dir))
    print(f"✓ Zip created: {zip_path}  ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    # connect to Supabase
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    rows = fetch_all_rows(supabase)
    if not rows:
        print("No rows found. Check your TABLE_NAME and column names.")
        return

    # summary before downloading
    district_counts: dict[str, int] = {}
    for row in rows:
        raw = row.get(DISTRICT_COLUMN)
        d = DISTRICT_MAP.get(int(raw), f"district_{raw}") if raw is not None else "unknown"
        district_counts[d] = district_counts.get(d, 0) + 1

    print("\nDistrict breakdown:")
    for district, count in sorted(district_counts.items()):
        print(f"  {district}: {count} images")

    print(f"\nStarting downloads to '{OUTPUT_DIR}' ...")
    success, failed = 0, 0

    for i, row in enumerate(rows, 1):
        url      = row.get(URL_COLUMN)
        raw      = row.get(DISTRICT_COLUMN)
        district = DISTRICT_MAP.get(int(raw), f"district_{raw}") if raw is not None else "unknown"
        row_id   = row.get(ID_COLUMN, i)

        if not url:
            print(f"  ✗ Row {row_id} has no URL, skipping.")
            failed += 1
            continue

        # determine file extension from URL, default to .jpg
        ext = Path(url.split("?")[0]).suffix or ".jpg"
        filename = f"{district}_{row_id}{ext}"
        save_path = OUTPUT_DIR / district / filename

        # skip if already downloaded (resume support)
        if save_path.exists():
            print(f"  → [{i}/{len(rows)}] Already exists, skipping: {filename}")
            success += 1
            continue

        print(f"  ↓ [{i}/{len(rows)}] {filename}")
        if download_image(url, save_path):
            success += 1
        else:
            failed += 1

    print(f"\n✓ Downloaded: {success}  ✗ Failed: {failed}")

    # zip everything up
    if success > 0:
        zip_directory(OUTPUT_DIR, ZIP_OUTPUT)
        print(f"\nAll done! Upload this file to Roboflow:\n  {ZIP_OUTPUT.resolve()}")
    else:
        print("\nNo images downloaded. Check your credentials and column names.")


if __name__ == "__main__":
    main()
