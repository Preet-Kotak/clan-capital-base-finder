from dataclasses import dataclass
from itertools import combinations
from typing import Final, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


class _Position(TypedDict):
    type: str
    x: float
    y: float


class Building(_Position, total=False):
    conf: float


HALL_TYPES: Final = frozenset({"capital_peak", "district_hall"})
FUZZY_GROUPS: Final = {"cannon": "cannon_spear", "spear": "cannon_spear"}
SETTINGS: Final = {
    "tolerance": .35, "support": 20, "classes": 4, "median_residual": .15,
    "coverage": .7, "eigenvalue_ratio": .05, "scale_min": .5, "scale_max": 2.,
    "class_pairs": 8, "edges_per_pair": 32, "edge_spacing": 3,
    "verified_proposals": 32, "refinements": 2, "dedup_tolerance": .25,
}


@dataclass(frozen=True, slots=True)
class Geometry:
    buildings: tuple[Building, ...]
    building_indices: tuple[int, ...]
    points: NDArray[np.float64]
    classes: tuple[str, ...]
    groups: dict[str, NDArray[np.int64]]
    spacing: float


def geometry(layout: list[Building], fuzzy: bool) -> Geometry:
    indices = tuple(sorted(range(len(layout)), key=lambda i: (
        FUZZY_GROUPS.get(layout[i]["type"], layout[i]["type"]) if fuzzy else layout[i]["type"],
        layout[i]["x"], layout[i]["y"])))
    buildings = tuple(layout[index] for index in indices)
    classes = tuple(FUZZY_GROUPS.get(b["type"], b["type"]) if fuzzy else b["type"] for b in buildings)
    points = np.array([(b["x"], b["y"]) for b in buildings], dtype=float).reshape(-1, 2)
    groups = {name: np.flatnonzero(np.array(classes) == name) for name in sorted(set(classes))}
    spacing = 0.
    if len(points) > 1 and np.isfinite(points).all():
        points -= np.median(points, axis=0)
        distances = cdist(points, points)
        distances[distances <= 1e-12] = np.inf
        spacing = float(np.median(distances.min(axis=1)))
    return Geometry(buildings, indices, points, classes, groups, spacing)


def edges(layout: Geometry, pair: tuple[str, str]) -> NDArray[np.float64]:
    starts, ends = layout.points[layout.groups[pair[0]]], layout.points[layout.groups[pair[1]]]
    first, second = np.repeat(starts, len(ends), axis=0), np.tile(ends, (len(starts), 1))
    vectors = second - first
    lengths = np.linalg.norm(vectors, axis=1)
    keep = (lengths >= SETTINGS["edge_spacing"] * layout.spacing) & (np.abs(vectors) > 1e-12).all(axis=1)
    first, second, vectors, lengths = first[keep], second[keep], vectors[keep], lengths[keep]
    order = np.lexsort((second[:, 1], second[:, 0], first[:, 1], first[:, 0],
                        np.arctan2(vectors[:, 1], vectors[:, 0]), lengths))
    if len(order) > SETTINGS["edges_per_pair"]:
        order = order[np.linspace(0, len(order) - 1, SETTINGS["edges_per_pair"], dtype=int)]
    return np.stack((first[order], second[order]), axis=1)


def proposals(layouts: tuple[Geometry, Geometry]) -> NDArray[np.float64]:
    query, candidate = layouts
    pairs = sorted(combinations(sorted(query.groups.keys() & candidate.groups.keys()), 2), key=lambda pair: (
        len(query.groups[pair[0]]) * len(query.groups[pair[1]]) *
        len(candidate.groups[pair[0]]) * len(candidate.groups[pair[1]]), pair))
    transforms = []
    for pair in pairs[:SETTINGS["class_pairs"]]:
        qe, ce = edges(query, pair), edges(candidate, pair)
        if not len(qe) or not len(ce):
            continue
        q, c = np.repeat(qe, len(ce), axis=0), np.tile(ce, (len(qe), 1, 1))
        scale = (q[:, 1] - q[:, 0]) / (c[:, 1] - c[:, 0])
        keep = ((scale >= SETTINGS["scale_min"]) & (scale <= SETTINGS["scale_max"])).all(axis=1)
        translation = (q[keep] - scale[keep, None] * c[keep]).mean(axis=1)
        transforms.extend(np.column_stack((scale[keep], translation)))
    return np.unique(np.array(transforms).reshape(-1, 4), axis=0)


def distinct(transforms: NDArray[np.float64], layouts: tuple[Geometry, Geometry]) -> NDArray[np.float64]:
    query, candidate = layouts
    ends = np.array([candidate.points.min(axis=0), candidate.points.max(axis=0)])
    embedded = (transforms[:, None, :2] * ends + transforms[:, None, 2:]).reshape(-1, 4)
    tree = cKDTree(embedded)
    removed = np.zeros(len(transforms), dtype=bool)
    selected = []
    for index, transform in enumerate(transforms):
        if removed[index]:
            continue
        selected.append(index)
        tolerance = SETTINGS["dedup_tolerance"] * SETTINGS["tolerance"] * min(
            query.spacing, transform[:2].min() * candidate.spacing)
        neighbors = np.asarray(tree.query_ball_point(embedded[index], np.sqrt(2) * tolerance), dtype=int)
        neighbors = neighbors[(neighbors > index) & ~removed[neighbors]]
        for start in range(0, len(neighbors), 64):
            indices = neighbors[start:start + 64]
            delta = transforms[indices] - transform
            displacement = delta[:, None, :2] * candidate.points + delta[:, None, 2:]
            removed[indices[np.linalg.norm(displacement, axis=2).max(axis=1) <= tolerance]] = True
    return transforms[selected]


def shortlist(transforms: NDArray[np.float64], layouts: tuple[Geometry, Geometry]) -> NDArray[np.float64]:
    query, candidate = layouts
    supports, residuals = np.zeros(len(transforms), dtype=int), np.zeros(len(transforms))
    for name in sorted(query.groups.keys() & candidate.groups.keys()):
        q, c = query.points[query.groups[name]], candidate.points[candidate.groups[name]]
        qt, ct = cKDTree(q), cKDTree(c)
        for start in range(0, len(transforms), 64):
            chunk = transforms[start:start + 64]
            scale, translation = chunk[:, :2], chunk[:, 2:]
            tolerance = SETTINGS["tolerance"] * np.minimum(query.spacing, scale.min(axis=1) * candidate.spacing)
            forward = qt.query(scale[:, None] * c + translation[:, None])[0]
            reverse = ct.query((q - translation[:, None]) / scale[:, None])[0] * scale.min(axis=1)[:, None]
            supports[start:start + len(chunk)] += np.minimum(
                (forward <= tolerance[:, None]).sum(axis=1), (reverse <= tolerance[:, None]).sum(axis=1))
            residuals[start:start + len(chunk)] += np.minimum(forward / tolerance[:, None], 1).sum(axis=1)
            residuals[start:start + len(chunk)] += np.minimum(reverse / tolerance[:, None], 1).sum(axis=1)
    order = np.lexsort((transforms[:, 3], transforms[:, 2], transforms[:, 1], transforms[:, 0], residuals, -supports))
    return transforms[order[:SETTINGS["verified_proposals"]]]
