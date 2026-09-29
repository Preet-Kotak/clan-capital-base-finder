"""
Script A: Given a screenshot, find top N similar bases from layouts.json

Usage:
    python scripts/find_matches.py --image path/to/image.jpg --district 0
    python scripts/find_matches.py --image path/to/image.jpg --district 0 --top 5 --threshold 80 --fuzzy
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.extractor import Extractor, DISTRICT_MAP
from src.matcher import match_layouts

LAYOUTS_FILE = Path("data/layouts/layouts.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     required=True,       help="Path to screenshot")
    parser.add_argument("--district",  required=True, type=int, help="District number 0-8")
    parser.add_argument("--top",       default=5,     type=int, help="Number of results")
    parser.add_argument("--threshold", default=80,    type=float, help="Min match % (default 80)")
    parser.add_argument("--fuzzy",     action="store_true", help="Use fuzzy type matching")
    args = parser.parse_args()

    # validate
    if not Path(args.image).exists():
        print(f"Image not found: {args.image}")
        sys.exit(1)
    if args.district not in DISTRICT_MAP:
        print(f"Invalid district: {args.district}. Must be 0-8.")
        sys.exit(1)

    district_name = DISTRICT_MAP[args.district]
    mode          = "fuzzy" if args.fuzzy else "strict"

    print(f"\nDistrict:  {district_name} ({args.district})")
    print(f"Mode:      {mode}")
    print(f"Threshold: {args.threshold}%")
    print(f"Top N:     {args.top}")

    # extract input base
    print(f"\nExtracting layout from input image...")
    extractor    = Extractor()
    input_result = extractor.extract(args.image, district=args.district)
    input_layout = input_result["buildings"]

    print(f"  Detections: {input_result['total_detections']}")
    print(f"  Anchor:     {'found' if input_result['anchor_found'] else 'not found — using center'}")
    print(f"  Conf used:  {input_result['conf_used']}")
    print(f"  Runs:       {input_result['runs']}")

    # load layouts
    print(f"\nLoading layouts from {LAYOUTS_FILE}...")
    with open(LAYOUTS_FILE) as f:
        all_layouts = json.load(f)

    # filter by district
    district_layouts = {
        base_id: data
        for base_id, data in all_layouts.items()
        if data["district"] == args.district
    }
    print(f"  Bases in district {args.district}: {len(district_layouts)}")

    # compare against all
    print(f"\nComparing against {len(district_layouts)} bases...")
    results = []

    for base_id, data in district_layouts.items():
        db_layout = data["buildings"]
        result    = match_layouts(input_layout, db_layout, fuzzy=args.fuzzy)
        results.append({
            "base_id":   base_id,
            "match_pct": result["match_pct"],
            "matched":   result["matched"],
            "total":     result["total"],
        })

    # sort by match %
    results.sort(key=lambda x: x["match_pct"], reverse=True)

    # filter by threshold
    above_threshold = [r for r in results if r["match_pct"] >= args.threshold]

    print(f"\n{'=' * 50}")
    if not above_threshold:
        print(f"No strong match found (threshold: {args.threshold}%)")
        print(f"\nBest result was: {results[0]['match_pct']}% (base_id: {results[0]['base_id']})")
    else:
        print(f"Top {min(args.top, len(above_threshold))} matches ≥ {args.threshold}%:")
        print(f"{'=' * 50}")
        for i, r in enumerate(above_threshold[:args.top], 1):
            print(f"  #{i}  base_id: {r['base_id']:>6}  |  {r['match_pct']:>5.1f}%  |  {r['matched']}/{r['total']} buildings matched")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
