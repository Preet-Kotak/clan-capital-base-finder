"""
Debug script for compare_two — shows exactly which buildings matched and which didn't
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.extractor import Extractor
from src.matcher import match_layouts

IMAGE1 = "data/screenshots/by_district/barbarian_camp/barbarian_camp_626.jpg"
IMAGE2 = "data/screenshots/by_district/barbarian_camp/barbarian_camp_656.png"

ext = Extractor()

print("Extracting base 1...")
r1 = ext.extract(IMAGE1, district=1)
print(f"  Total: {r1['total_detections']}, anchor: {r1['anchor_found']}, conf: {r1['conf_used']}, runs: {r1['runs']}")

print("Extracting base 2...")
r2 = ext.extract(IMAGE2, district=1)
print(f"  Total: {r2['total_detections']}, anchor: {r2['anchor_found']}, conf: {r2['conf_used']}, runs: {r2['runs']}")

# count per class in each base
print("\n--- Base 1 class counts ---")
counts1 = defaultdict(int)
for b in r1["buildings"]:
    counts1[b["type"]] += 1
for t, c in sorted(counts1.items()):
    print(f"  {t:20s}: {c}")

print("\n--- Base 2 class counts ---")
counts2 = defaultdict(int)
for b in r2["buildings"]:
    counts2[b["type"]] += 1
for t, c in sorted(counts2.items()):
    print(f"  {t:20s}: {c}")

# run matcher
print("\n--- Matching ---")
result = match_layouts(r1["buildings"], r2["buildings"], fuzzy=False)
print(f"Match: {result['match_pct']}%")
print(f"Matched: {result['matched']} / {result['total']}")
print(f"Unmatched in base1: {len(result['unmatched_a'])}")
print(f"Unmatched in base2: {len(result['unmatched_b'])}")

print("\n--- Unmatched in Base 1 ---")
um1_counts = defaultdict(int)
for b in result["unmatched_a"]:
    um1_counts[b["type"]] += 1
for t, c in sorted(um1_counts.items()):
    print(f"  {t:20s}: {c}")

print("\n--- Unmatched in Base 2 ---")
um2_counts = defaultdict(int)
for b in result["unmatched_b"]:
    um2_counts[b["type"]] += 1
for t, c in sorted(um2_counts.items()):
    print(f"  {t:20s}: {c}")

print("\n--- Matched pairs (type, distance) ---")
match_counts = defaultdict(list)
for m in result["matched_buildings"]:
    match_counts[m["type"]].append(m["distance"])
for t, dists in sorted(match_counts.items()):
    avg = sum(dists)/len(dists)
    print(f"  {t:20s}: {len(dists)} matched, avg dist={avg:.4f}, max dist={max(dists):.4f}")

print("\n--- Per class summary (base1 vs base2) ---")
all_types = sorted(set(list(counts1.keys()) + list(counts2.keys())))
print(f"  {'Type':20s} {'B1':>4} {'B2':>4} {'Matched':>8} {'Unmatched B1':>13} {'Unmatched B2':>13}")
print(f"  {'-'*70}")
for t in all_types:
    b1    = counts1.get(t, 0)
    b2    = counts2.get(t, 0)
    mat   = len(match_counts.get(t, []))
    um1   = um1_counts.get(t, 0)
    um2   = um2_counts.get(t, 0)
    print(f"  {t:20s} {b1:>4} {b2:>4} {mat:>8} {um1:>13} {um2:>13}")
