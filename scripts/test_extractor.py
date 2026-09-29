import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# force reload
import importlib
import src.extractor
importlib.reload(src.extractor)
from src.extractor import Extractor

ext = Extractor()
result = ext.extract("data/screenshots/by_district/capital_peak/capital_peak_469.png", district=0)

print(f"Anchor found:      {result['anchor_found']}")
print(f"Total detections:  {result['total_detections']}")
print(f"Conf used:         {result['conf_used']}")
print(f"Runs:              {result['runs']}")
print(f"Calibrated:        {result['calibrated']}")
print("\nFirst 5 buildings:")
for b in result["buildings"][:5]:
    print(f"  {b['type']:20s}  x={b['x']:+.4f}  y={b['y']:+.4f}  conf={b['conf']}")
