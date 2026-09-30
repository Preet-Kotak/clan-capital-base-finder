"""
Match defense arrangements using relative geometry, without a hall anchor.
Supports strict mode (exact type match) and fuzzy mode (grouped types).
"""

from dataclasses import dataclass
from math import ceil
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from src.geometry import Building, Geometry, HALL_TYPES, SETTINGS, distinct, geometry, proposals, shortlist


class MatchedBuilding(TypedDict):
    type: str
    a: Building
    b: Building
    distance: float


class MatchResult(TypedDict):
    match_pct: float
    matched: int
    total: int
    matched_buildings: list[MatchedBuilding]
    unmatched_a: list[Building]
    unmatched_b: list[Building]


@dataclass(frozen=True, slots=True)
class Fit:
    transform: NDArray[np.float64]
    query_indices: NDArray[np.int64]
    candidate_indices: NDArray[np.int64]
    residual: float
    spacing: float

    @property
    def order(self) -> tuple:
        return (-len(self.query_indices), self.residual, *self.transform)


def assign(layouts: tuple[Geometry, Geometry], transform: NDArray[np.float64]) -> Fit:
    query, candidate = layouts
    scale, translation = transform[:2], transform[2:]
    spacing = min(query.spacing, scale.min() * candidate.spacing)
    tolerance = SETTINGS["tolerance"] * spacing
    qi, ci, residuals = [], [], []
    for name in sorted(query.groups.keys() & candidate.groups.keys()):
        q, c = query.groups[name], candidate.groups[name]
        distances = cdist(query.points[q], scale * candidate.points[c] + translation)
        costs = np.where(distances <= tolerance, distances / tolerance, min(len(q), len(c)) + 1)
        rows, columns = linear_sum_assignment(costs)
        keep = distances[rows, columns] <= tolerance
        qi.extend(q[rows[keep]])
        ci.extend(c[columns[keep]])
        residuals.extend(distances[rows[keep], columns[keep]])
    return Fit(transform, np.array(qi, dtype=int), np.array(ci, dtype=int),
               float(np.median(residuals)) if residuals else float("inf"), spacing)


def refine(layouts: tuple[Geometry, Geometry], fit: Fit) -> NDArray[np.float64] | None:
    if len(fit.query_indices) < 2:
        return None
    query, candidate = layouts
    q, c = query.points[fit.query_indices], candidate.points[fit.candidate_indices]
    qm, cm = q.mean(axis=0), c.mean(axis=0)
    variance = ((c - cm) ** 2).sum(axis=0)
    if np.any(variance <= 1e-24):
        return None
    scale = ((q - qm) * (c - cm)).sum(axis=0) / variance
    if not np.isfinite(scale).all() or np.any((scale < SETTINGS["scale_min"]) | (scale > SETTINGS["scale_max"])):
        return None
    return np.concatenate((scale, qm - scale * cm))


def supported(layouts: tuple[Geometry, Geometry], fit: Fit) -> bool:
    query, candidate = layouts
    if (len(fit.query_indices) < SETTINGS["support"] or fit.residual > SETTINGS["median_residual"] * fit.spacing or
            len({query.classes[index] for index in fit.query_indices}) < SETTINGS["classes"]):
        return False
    for layout, indices in ((query, fit.query_indices), (candidate, fit.candidate_indices)):
        inliers = layout.points[indices]
        extent = np.ptp(np.quantile(layout.points, [.05, .95], axis=0), axis=0)
        if np.any(extent <= 1e-12) or np.any(np.ptp(inliers, axis=0) / extent < SETTINGS["coverage"]):
            return False
        eigenvalues = np.linalg.eigvalsh(np.cov(inliers.T))
        if eigenvalues[-1] <= 0 or eigenvalues[0] / eigenvalues[-1] < SETTINGS["eigenvalue_ratio"]:
            return False
    return True


def match_layouts(layout_a: list[Building], layout_b: list[Building], fuzzy: bool = False) -> MatchResult:
    """Align defenses by type and relative positions, returning the original building records."""
    defenses_a = [building for building in layout_a if building["type"] not in HALL_TYPES]
    defenses_b = [building for building in layout_b if building["type"] not in HALL_TYPES]
    total = max(len(defenses_a), len(defenses_b))
    rejected: MatchResult = {"match_pct": 0., "matched": 0, "total": total, "matched_buildings": [],
                             "unmatched_a": defenses_a, "unmatched_b": defenses_b}
    first, second = geometry(defenses_a, fuzzy), geometry(defenses_b, fuzzy)
    for item in (first, second):
        if (len(item.points) < SETTINGS["support"] or not np.isfinite(item.points).all() or
                not np.isfinite(item.spacing) or item.spacing <= 1e-12 or np.any(np.ptp(item.points, axis=0) <= 1e-12)):
            return rejected
    reversed_pair = (first.classes, tuple(first.points.flat)) > (second.classes, tuple(second.points.flat))
    if reversed_pair:
        first, second = second, first
    common = first.groups.keys() & second.groups.keys()
    if (len(common) < SETTINGS["classes"] or
            sum(min(len(first.groups[key]), len(second.groups[key])) for key in common) < SETTINGS["support"]):
        return rejected
    layouts = first, second
    transforms = proposals(layouts)
    if not len(transforms):
        return rejected
    selected = shortlist(distinct(transforms, layouts), layouts)
    fits = []
    for transform in selected:
        fit = assign(layouts, transform)
        best = fit
        for _ in range(SETTINGS["refinements"]):
            refined = refine(layouts, fit)
            if refined is None:
                break
            fit = assign(layouts, refined)
            if fit.order < best.order:
                best = fit
        if supported(layouts, best):
            fits.append(best)
    if not fits:
        return rejected
    fits.sort(key=lambda fit: fit.order)
    best = fits[0]
    allowance = max(2, ceil(.05 * total))
    if any(len(best.query_indices) - len(fit.query_indices) <= allowance and
           np.linalg.norm((fit.transform[:2] - best.transform[:2]) * second.points +
                          fit.transform[2:] - best.transform[2:], axis=1).max() >
           2 * SETTINGS["tolerance"] * best.spacing for fit in fits[1:]):
        return rejected
    matched_buildings: list[MatchedBuilding] = []
    for qi, ci in zip(best.query_indices, best.candidate_indices, strict=True):
        a, b = first.buildings[qi], second.buildings[ci]
        distance = float(np.linalg.norm(first.points[qi] - (best.transform[:2] * second.points[ci] + best.transform[2:])))
        matched_buildings.append({"type": first.classes[qi], "a": b if reversed_pair else a,
                                  "b": a if reversed_pair else b, "distance": round(distance, 4)})
    matched_a = {first.building_indices[index] for index in best.query_indices}
    matched_b = {second.building_indices[index] for index in best.candidate_indices}
    if reversed_pair:
        matched_a, matched_b = matched_b, matched_a
    matched = len(matched_buildings)
    return {"match_pct": round(100 * matched / total, 1), "matched": matched, "total": total,
            "matched_buildings": matched_buildings,
            "unmatched_a": [building for index, building in enumerate(defenses_a) if index not in matched_a],
            "unmatched_b": [building for index, building in enumerate(defenses_b) if index not in matched_b]}
