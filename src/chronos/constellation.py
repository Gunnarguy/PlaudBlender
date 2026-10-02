"""Entity Constellation: the precomputed map behind GET /api/v1/graph/constellation.

From the exported entity graph this module derives
- communities: Louvain, where a stated relationship weighs 3 co-mentions;
- a 2D layout inside the unit disc (so in [-1, 1]): communities are packed as discs (linked
  ones pull together), each laid out inside with Fruchterman-Reingold in plain numpy (there
  is no scipy on the Pi); entities with no link at all fill a ring around them;
- weekly activity per entity ({"YYYY-Www": moments}) and entity_index.json (entity counts
  per recording and per day).

Stability: the previous entity_graph.json seeds the build. Entities already on the map keep
their position and their community exactly; a new entity joins the community of its
strongest placed neighbour and settles next to it; groups with no link to the map get
their own communities and are packed at its edge. If the map then reaches past radius 1,
everything shrinks by one factor. Communities and map are only computed from scratch (the
map then turned, scaled and shifted onto the old one) when the map has doubled since its
last full layout, when most of its entities are new, or when LAYOUT_VERSION changes.
Identical inputs give identical output: fixed seeds, sorted iteration, CRC32-derived angles
(never Python's per-process salted hash()).
"""

from __future__ import annotations

import json
import math
import time
import zlib
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import networkx as nx
import numpy as np

LAYOUT_VERSION = 1
STATED_WEIGHT = 3.0  # a stated relationship ("works_with") counts like 3 co-mentions
LABEL_TYPES = ("person", "organization", "project", "location")  # preferred in labels
RELAYOUT_GROWTH = 2.0  # full layout again once the map has this many times its base nodes
MARGIN = 0.9  # a full layout fills the disc of radius 0.9: room to grow before anything shrinks
PAIR_BUDGET = 30_000_000  # n^2 * iterations for one community's force layout (~2 s on a Pi 4)
BLOCK = 512  # rows per block of the dense repulsion: <= 512 x n float32 arrays in memory
_GOLDEN = math.pi * (3.0 - math.sqrt(5.0))


def _unit_hash(text: str) -> float:
    """A stable number in [0, 1) for a string."""
    return zlib.crc32(text.encode("utf-8")) / 2**32


# --------------------------------------------------------------------------- graph


def weighted_graph(node_ids: Iterable[str], edges: Iterable[dict[str, Any]]) -> nx.Graph:
    """Undirected graph of the exported entities; parallel edges summed, stated ones x3.

    Nodes and edges are inserted in sorted order, so Louvain's seeded shuffle sees the same
    sequence for the same input.
    """
    ids = set(node_ids)
    weights: dict[tuple[str, str], float] = defaultdict(float)
    for edge in edges:
        a, b = edge.get("source"), edge.get("target")
        if a == b or a not in ids or b not in ids:
            continue
        factor = 1.0 if edge.get("type") == "co_mentioned" else STATED_WEIGHT
        weights[(a, b) if a < b else (b, a)] += factor * float(edge.get("weight") or 1.0)
    graph = nx.Graph()
    graph.add_nodes_from(sorted(ids))
    graph.add_weighted_edges_from((a, b, w) for (a, b), w in sorted(weights.items()) if w > 0)
    return graph


def _by_size(groups: Iterable[Iterable[str]], mentions: dict[str, int]) -> list[list[str]]:
    """Largest first (then most mentioned, then smallest id); members sorted by id."""
    out = [sorted(group) for group in groups if group]
    out.sort(key=lambda m: (-len(m), -sum(int(mentions.get(n) or 0) for n in m), m[0]))
    return out


def find_communities(graph: nx.Graph, mentions: dict[str, int]) -> list[list[str]]:
    """Louvain communities (seed 42), largest first; members sorted by id."""
    if graph.number_of_nodes() == 0:
        return []
    return _by_size(nx.community.louvain_communities(graph, weight="weight", seed=42), mentions)


def extend_communities(graph: nx.Graph, previous: dict[str, int], mentions: dict[str, int]) -> list[list[str]]:
    """Last build's communities, grown, for an incremental build.

    An entity in ``previous`` keeps its community; a new one joins the community of its
    strongest already-assigned neighbour (in rounds; ties go to the smaller id, as in the
    layout's placement), and entities with no path to an assigned one get Louvain
    communities of their own. Fresh Louvain on a grown graph reshuffles membership, which
    would scatter regions across a map whose entities stay put.
    """
    label: dict[str, tuple[int, int]] = {n: (0, int(c)) for n, c in previous.items() if graph.has_node(n)}
    pending = [n for n in graph.nodes if n not in label]
    while pending:
        batch = {}
        for node in pending:
            best: Optional[tuple[float, str]] = None
            for other, data in graph[node].items():
                if other in label:
                    weight = float(data.get("weight", 1.0))
                    if best is None or weight > best[0] or (weight == best[0] and other < best[1]):
                        best = (weight, other)
            if best is not None:
                batch[node] = label[best[1]]
        if not batch:
            break
        label.update(batch)
        pending = [n for n in pending if n not in label]
    if pending:
        parts = nx.community.louvain_communities(graph.subgraph(pending), weight="weight", seed=42)
        for i, part in enumerate(sorted(sorted(part) for part in parts)):
            for node in part:
                label[node] = (1, i)
    groups: dict[tuple[int, int], list[str]] = defaultdict(list)
    for node, key in label.items():
        groups[key].append(node)
    return _by_size(groups.values(), mentions)


def _label_rank(node_id: str, nodes_by_id: dict[str, dict]):
    """People, organizations, projects and places first, then topics, then actions and the
    rest; more mentions first within each."""
    node = nodes_by_id.get(node_id) or {}
    kind = node.get("type")
    tier = 0 if kind in LABEL_TYPES else 1 if kind == "topic" else 2
    return (tier, -int(node.get("mentions") or 0), str(node.get("name") or ""), node_id)


def community_label(members: list[str], nodes_by_id: dict[str, dict], top: int = 3) -> tuple[str, list[str]]:
    """Up to ``top`` member names by mentions; people, organizations, projects and places
    before topics, topics before actions and the rest."""
    best = sorted(members, key=lambda node_id: _label_rank(node_id, nodes_by_id))[:top]
    return " · ".join(str((nodes_by_id.get(n) or {}).get("name") or n) for n in best), best


# --------------------------------------------------------------------------- time


def iso_week(day: str | date) -> str:
    """'2026-10-02' -> '2026-W40' (ISO year and week)."""
    when = date.fromisoformat(day[:10]) if isinstance(day, str) else day
    year, week, _ = when.isocalendar()
    return f"{year}-W{week:02d}"


def local_day(utc_naive: datetime, tz) -> str:
    """A stored naive-UTC timestamp as the local 'YYYY-MM-DD' the timeline shows it under."""
    aware = utc_naive.replace(tzinfo=timezone.utc) if utc_naive.tzinfo is None else utc_naive
    return aware.astimezone(tz).strftime("%Y-%m-%d")


def recording_days_from_db(session, recording_ids: Iterable[str]) -> dict[str, str]:
    """Recording id -> the day /api/v1/timeline/days lists it under: the local date of
    chronos_recordings.created_at (naive UTC, shown in the machine's time zone)."""
    from src.config import get_local_timezone
    from src.database.models import ChronosRecording

    tz = get_local_timezone()
    wanted = sorted({str(r) for r in recording_ids if r})
    days: dict[str, str] = {}
    for start in range(0, len(wanted), 500):
        rows = (
            session.query(ChronosRecording.recording_id, ChronosRecording.created_at)
            .filter(ChronosRecording.recording_id.in_(wanted[start : start + 500]))
            .all()
        )
        for recording_id, created_at in rows or []:
            if isinstance(created_at, datetime):
                days[str(recording_id)] = local_day(created_at, tz)
    return days


def moment_days(
    moments: dict[str, tuple[str, str]], recording_days: Optional[dict[str, str]] = None
) -> dict[str, str]:
    """event_id -> 'YYYY-MM-DD' of the day its recording is listed under in the timeline.

    moments: event_id -> (recording_id, start time ISO; event times are local wall time).
    A recording missing from ``recording_days`` falls back to its earliest moment's date
    (the same date for 803 of 804 recordings on 2026-10-02).
    """
    recording_days = recording_days or {}
    earliest: dict[str, str] = {}
    for recording_id, when in moments.values():
        if recording_id and when and (recording_id not in earliest or when < earliest[recording_id]):
            earliest[recording_id] = when
    days = {}
    for event_id, (recording_id, when) in moments.items():
        day = recording_days.get(recording_id) or (earliest.get(recording_id) or when or "")[:10]
        if len(day) == 10:
            days[event_id] = day
    return days


def build_activity(
    kept: Iterable[str],
    entity_events: dict[str, list[tuple[str, str]]],
    moments: dict[str, tuple[str, str]],
    recording_days: Optional[dict[str, str]] = None,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, dict[str, int]]]]:
    """Weekly moment counts per entity, and the recording/day index.

    entity_events: entity_id -> [(event_id, start ISO)], one entry per moment.
    Returns (weeks, index) where weeks[entity] = {"YYYY-Www": moments} and
    index = {"recordings": {recording_id: {entity: moments}}, "days": {day: {entity: moments}}}.
    A moment's day is its recording's timeline day, so the days of a week add up to it.
    """
    days = moment_days(moments, recording_days)
    weeks: dict[str, dict[str, int]] = {}
    by_recording: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_day: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    week_of: dict[str, str] = {}
    for entity_id in sorted(set(kept)):
        counts: dict[str, int] = defaultdict(int)
        for event_id, when in entity_events.get(entity_id, []):
            recording_id = (moments.get(event_id) or ("", ""))[0]
            if recording_id:
                by_recording[recording_id][entity_id] += 1
            day = days.get(event_id) or (when or "")[:10]
            if len(day) != 10:
                continue
            by_day[day][entity_id] += 1
            if day not in week_of:
                week_of[day] = iso_week(day)
            counts[week_of[day]] += 1
        weeks[entity_id] = dict(sorted(counts.items()))
    index = {
        "recordings": {r: dict(sorted(c.items())) for r, c in sorted(by_recording.items())},
        "days": {d: dict(sorted(c.items())) for d, c in sorted(by_day.items())},
    }
    return weeks, index


# --------------------------------------------------------------------------- forces


def _repel(cols: np.ndarray, rows: np.ndarray, k2: float, col_weight: Optional[np.ndarray] = None):
    """Fruchterman-Reingold repulsion k^2/d on each row point from every column point.

    Exact and dense, computed in float32 blocks of BLOCK rows; a point never pushes itself
    (its difference vector is zero).
    """
    fx = np.zeros(len(rows))
    fy = np.zeros(len(rows))
    if len(rows) == 0 or len(cols) == 0:
        return fx, fy
    cx = cols[:, 0].astype(np.float32)
    cy = cols[:, 1].astype(np.float32)
    eps = np.float32(1e-6 * k2)
    weight = None if col_weight is None else col_weight.astype(np.float32)[None, :]
    for start in range(0, len(rows), BLOCK):
        stop = start + BLOCK
        dx = np.subtract.outer(rows[start:stop, 0].astype(np.float32), cx)
        dy = np.subtract.outer(rows[start:stop, 1].astype(np.float32), cy)
        inv = dx * dx
        inv += dy * dy
        inv += eps
        np.reciprocal(inv, out=inv)
        if weight is not None:
            inv *= weight
        fx[start:stop] = np.einsum("ij,ij->i", dx, inv)
        fy[start:stop] = np.einsum("ij,ij->i", dy, inv)
    return k2 * fx, k2 * fy


def _attract(pos: np.ndarray, src: np.ndarray, dst: np.ndarray, weight: np.ndarray, k: float):
    """FR attraction w*d^2/k along each edge, summed per node."""
    n = len(pos)
    if len(src) == 0:
        return np.zeros(n), np.zeros(n)
    dx = pos[dst, 0] - pos[src, 0]
    dy = pos[dst, 1] - pos[src, 1]
    f = weight * np.sqrt(dx * dx + dy * dy) / k
    dx *= f
    dy *= f
    fx = np.bincount(src, dx, n) - np.bincount(dst, dx, n)
    fy = np.bincount(src, dy, n) - np.bincount(dst, dy, n)
    return fx, fy


def _limit(fx: np.ndarray, fy: np.ndarray, cap: float):
    """Scale each displacement down to at most ``cap`` long."""
    scale = np.minimum(1.0, cap / (np.sqrt(fx * fx + fy * fy) + 1e-12))
    return fx * scale, fy * scale


def _layout_group(size: int, src: np.ndarray, dst: np.ndarray, weight: np.ndarray, seed: str) -> np.ndarray:
    """Force layout of one community with unit link length, centred on (0, 0).

    Iterations shrink with size (PAIR_BUDGET / n^2, clamped to 25..150): a community of
    1,000 runs 30, one of 3,000 runs 25, so the cost of a very large one stays bounded.
    """
    if size == 1:
        return np.zeros((1, 2))
    angle = 2 * math.pi * _unit_hash(seed)
    if size == 2:
        return 0.5 * np.array([[math.cos(angle), math.sin(angle)], [-math.cos(angle), -math.sin(angle)]])
    rng = np.random.default_rng(zlib.crc32(seed.encode("utf-8")))
    radius = 0.6 * math.sqrt(size)
    r = radius * np.sqrt(rng.random(size))
    theta = 2 * math.pi * rng.random(size)
    pos = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    iterations = int(min(150, max(25, PAIR_BUDGET // (size * size))))
    t0, t1 = max(1.0, radius / 3), 0.02
    gravity = 1.5 / math.sqrt(size)  # keeps long tails in; weak next to the links
    for it in range(iterations):
        fx, fy = _repel(pos, pos, 1.0)
        ax, ay = _attract(pos, src, dst, weight, 1.0)
        fx += ax - gravity * pos[:, 0]
        fy += ay - gravity * pos[:, 1]
        fx, fy = _limit(fx, fy, t1 + (t0 - t1) * (1 - it / iterations))
        pos[:, 0] += fx
        pos[:, 1] += fy
    return pos - pos.mean(axis=0)


def _collide(
    pos: np.ndarray, radius: np.ndarray, reach2: np.ndarray, mass: np.ndarray, fixed: np.ndarray, strength: float = 0.7
) -> None:
    """Push overlapping discs apart; the lighter one moves more and a fixed one never moves.

    reach2[i, j] = (radius[i] + radius[j])^2 as float32 (0 on the diagonal). Only the
    movable discs are tested against all others (one dense float32 distance test), and only
    overlapping pairs are worked on: in an incremental build most discs are fixed.
    """
    rows = np.flatnonzero(~fixed)
    if not len(rows):
        return
    x = pos[:, 0].astype(np.float32)
    y = pos[:, 1].astype(np.float32)
    dx = np.subtract.outer(x[rows], x)
    dy = np.subtract.outer(y[rows], y)
    d2 = dx * dx
    d2 += dy * dy
    hit, j = np.nonzero(d2 < reach2[rows])
    if not len(j):
        return
    i = rows[hit]
    ddx = pos[i, 0] - pos[j, 0]
    ddy = pos[i, 1] - pos[j, 1]
    dist = np.sqrt(ddx * ddx + ddy * ddy)
    share = np.where(fixed[j], 1.0, mass[j] / (mass[i] + mass[j]))  # i's part of the overlap
    push = (radius[i] + radius[j] - dist) / np.maximum(dist, 1e-9) * share * strength
    pos[:, 0] += np.bincount(i, ddx * push, len(pos))
    pos[:, 1] += np.bincount(i, ddy * push, len(pos))


def _pack(
    center: np.ndarray,
    radius: np.ndarray,
    links: tuple[np.ndarray, np.ndarray, np.ndarray],
    fixed: np.ndarray,
    gap: float,
    ticks: int = 300,
) -> np.ndarray:
    """Place community discs: linked discs pull toward touching, all drift to the centre,
    none overlap (a d3-force style simulation over the small community graph)."""
    pos = center.astype(float).copy()
    count = len(radius)
    if count < 2 or fixed.all():
        return pos
    vel = np.zeros_like(pos)
    li, lj, lw = links
    if len(li):
        degree = np.bincount(li, minlength=count) + np.bincount(lj, minlength=count)
        strongest = np.zeros(count)
        np.maximum.at(strongest, li, lw)
        np.maximum.at(strongest, lj, lw)
        relative = lw / np.maximum(np.maximum(strongest[li], strongest[lj]), 1e-12)
        strength = (0.3 + 0.7 * relative) / np.sqrt(np.minimum(degree[li], degree[lj]))
        bias = degree[li] / (degree[li] + degree[lj])
        target = radius[li] + radius[lj] + gap
    mass = np.maximum(radius, gap) ** 2
    reach = radius + gap / 2
    reach2 = (np.add.outer(reach, reach) ** 2).astype(np.float32)
    np.fill_diagonal(reach2, 0.0)
    pull = 0.05
    alpha, decay = 1.0, 1 - 0.001 ** (1 / ticks)
    for _ in range(ticks):
        if len(li):
            dx = pos[lj, 0] + vel[lj, 0] - pos[li, 0] - vel[li, 0]
            dy = pos[lj, 1] + vel[lj, 1] - pos[li, 1] - vel[li, 1]
            length = np.sqrt(dx * dx + dy * dy) + 1e-9
            f = (length - target) / length * alpha * strength
            dx *= f
            dy *= f
            vel[:, 0] += np.bincount(li, dx * (1 - bias), count) - np.bincount(lj, dx * bias, count)
            vel[:, 1] += np.bincount(li, dy * (1 - bias), count) - np.bincount(lj, dy * bias, count)
        vel -= pos * (pull * alpha)
        vel *= 0.6
        vel[fixed] = 0.0
        pos += vel
        _collide(pos, reach, reach2, mass, fixed)
        alpha -= alpha * decay
    for _ in range(50):  # settle any overlap the last ticks left
        before = pos.copy()
        _collide(pos, reach, reach2, mass, fixed, strength=1.0)
        if np.abs(pos - before).max() < 1e-6 * gap:
            break
    return pos


def _ring(count: int, inner: float, spacing: float) -> np.ndarray:
    """``count`` evenly spread points filling an annulus that starts at radius ``inner``
    (a sunflower: golden-angle steps, one ``spacing``-sized cell per point)."""
    i = np.arange(count, dtype=float)
    r = np.sqrt(inner * inner + (i + 0.5) * spacing * spacing / math.pi)
    theta = _GOLDEN * i
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


# --------------------------------------------------------------------------- layout


class _Graph:
    """Index arrays for one layout run."""

    def __init__(self, graph: nx.Graph, groups: list[list[str]]):
        self.ids = sorted(graph.nodes)
        self.index = {node_id: i for i, node_id in enumerate(self.ids)}
        self.n = len(self.ids)
        self.comm = np.zeros(self.n, dtype=np.int64)
        self.members = []
        for c, group in enumerate(groups):
            members = np.array(sorted(self.index[m] for m in group), dtype=np.int64)
            self.comm[members] = c
            self.members.append(members)
        rows = sorted((self.index[a], self.index[b], float(w)) for a, b, w in graph.edges(data="weight", default=1.0))
        self.src = np.array([r[0] for r in rows], dtype=np.int64)
        self.dst = np.array([r[1] for r in rows], dtype=np.int64)
        self.raw = np.array([r[2] for r in rows], dtype=float)
        self.weight = 1.0 + np.log(np.maximum(self.raw, 1.0))  # hubs pull, but not 50x harder
        self.degree = np.bincount(self.src, minlength=self.n) + np.bincount(self.dst, minlength=self.n)
        self.tier = np.zeros(self.n)  # set by layout(): ring order for isolated entities

    def isolated(self, communities: Iterable[int]) -> list[int]:
        """The single-entity communities with no link at all, best known first."""
        alone = [c for c in communities if len(self.members[c]) == 1 and self.degree[self.members[c][0]] == 0]
        return sorted(alone, key=lambda c: (self.tier[self.members[c][0]], self.ids[self.members[c][0]]))

    def community_links(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Summed edge weight between each pair of communities."""
        ci, cj = self.comm[self.src], self.comm[self.dst]
        cross = ci != cj
        if not cross.any():
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, np.zeros(0)
        lo, hi = np.minimum(ci[cross], cj[cross]), np.maximum(ci[cross], cj[cross])
        keys, inverse = np.unique(lo * len(self.members) + hi, return_inverse=True)
        weight = np.bincount(inverse, self.raw[cross])
        return keys // len(self.members), keys % len(self.members), np.log1p(weight)

    def local_layout(self, c: int) -> np.ndarray:
        """Unit-link-length layout of community c (rows follow self.members[c])."""
        members = self.members[c]
        local = np.full(self.n, -1, dtype=np.int64)
        local[members] = np.arange(len(members))
        inside = (local[self.src] >= 0) & (local[self.dst] >= 0)
        return _layout_group(
            len(members), local[self.src[inside]], local[self.dst[inside]], self.weight[inside], self.ids[members[0]]
        )


def _spiral(radius: np.ndarray, gap: float) -> np.ndarray:
    """Start positions: a golden-angle spiral from (0, 0), largest first, spaced by the
    area of the discs before it."""
    area = np.concatenate([[0.0], np.cumsum((radius + gap) ** 2)[:-1]])
    r = np.sqrt(area) * 1.1
    theta = _GOLDEN * np.arange(len(radius))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _seed_centers(radius: np.ndarray, links: tuple[np.ndarray, np.ndarray, np.ndarray], gap: float) -> np.ndarray:
    """Start positions for the discs: a weighted force layout of the community graph, so
    strongly linked communities start side by side (from a spiral; bigger discs push harder)."""
    pos = _spiral(radius, gap)
    li, lj, lw = links
    count = len(radius)
    if count < 3 or not len(li):
        return pos
    size = radius + gap
    span = float(np.sqrt((size**2).sum()))
    k = span / math.sqrt(count)
    weight = 3.0 * lw / lw.max()
    iterations = 200 if count <= 400 else 120
    for it in range(iterations):
        fx, fy = _repel(pos, pos, k * k, size / size.mean())
        ax, ay = _attract(pos, li, lj, weight, k)
        fx += ax - 0.02 * pos[:, 0]
        fy += ay - 0.02 * pos[:, 1]
        fx, fy = _limit(fx, fy, 0.1 * span * (1 - it / iterations) + 0.01 * k)
        pos[:, 0] += fx
        pos[:, 1] += fy
    return pos


def _cold(g: _Graph) -> np.ndarray:
    """Full layout: communities as discs packed by their links, each laid out inside;
    entities with no link at all fill a ring around them."""
    pos = np.zeros((g.n, 2))
    alone = g.isolated(range(len(g.members)))
    lone = set(alone)
    packed = [c for c in range(len(g.members)) if c not in lone]
    slot = np.full(len(g.members), -1, dtype=np.int64)
    slot[packed] = np.arange(len(packed))
    radius = np.zeros(len(packed))
    for c in packed:
        local = g.local_layout(c)
        pos[g.members[c]] = local
        radius[slot[c]] = float(np.sqrt((local**2).sum(axis=1)).max()) + 0.5
    li, lj, lw = g.community_links()  # isolated entities have no links
    links = (slot[li], slot[lj], lw)
    reach = 0.0
    if packed:
        centers = _pack(_seed_centers(radius, links, 1.0), radius, links, np.zeros(len(packed), dtype=bool), gap=1.0)
        for c in packed:
            pos[g.members[c]] += centers[slot[c]]
        reach = float((np.sqrt((centers**2).sum(axis=1)) + radius).max()) + 1.0
    if alone:
        pos[[g.members[c][0] for c in alone]] = _ring(len(alone), reach, 1.5)
    return pos


def _normalize(pos: np.ndarray, limit: float = MARGIN) -> tuple[np.ndarray, float]:
    """Scale uniformly about (0, 0), the centre the discs were packed around and the ring
    drawn around, into the disc of radius ``limit``; returns the scale. (The map is round,
    so a disc and not the square sets the scale; a radius <= 1 keeps x and y in [-1, 1].)"""
    span = _reach(pos)
    scale = limit / span if span > 0 else 1.0
    return pos * scale, scale


def _reach(pos: np.ndarray) -> float:
    """Largest distance from (0, 0)."""
    return float(np.sqrt((pos**2).sum(axis=1)).max()) if len(pos) else 0.0


def _align(new: np.ndarray, old: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Least-squares rotation/reflection, scale and shift taking ``new`` onto ``old`` (Procrustes)."""
    mean_new, mean_old = new.mean(axis=0), old.mean(axis=0)
    a, b = new - mean_new, old - mean_old
    u, s, vt = np.linalg.svd(a.T @ b)
    norm = float((a * a).sum())
    scale = float(s.sum()) / norm if norm > 0 else 1.0
    return u @ vt, scale, mean_new, mean_old


def _refine(g: _Graph, pos: np.ndarray, movable: np.ndarray, k: float, iterations: int = 50) -> None:
    """Let the movable nodes settle among their community and links; the rest stay put.
    Each travels at most ~5 link lengths, so a far-off second link can't drag it away from
    the neighbour (and community) it was placed by."""
    moving = np.flatnonzero(movable)
    if not len(moving):
        return
    groups = [(g.members[c], g.members[c][movable[g.members[c]]]) for c in sorted(set(g.comm[moving].tolist()))]
    k2 = k * k
    for it in range(iterations):
        fx, fy = _attract(pos, g.src, g.dst, g.weight, k)
        for members, rows in groups:
            rx, ry = _repel(pos[members], pos[rows], k2)
            fx[rows] += rx
            fy[rows] += ry
        sx, sy = _limit(fx[moving], fy[moving], 0.15 * k * (1 - it / iterations) + 0.02 * k)
        pos[moving, 0] += sx
        pos[moving, 1] += sy


def _warm(g: _Graph, pos: np.ndarray, known: np.ndarray, unit: float) -> tuple[np.ndarray, int, int]:
    """Keep the known nodes, attach new ones to the map, pack unlinked groups at its edge."""
    placed = known.copy()
    neighbors: list[list[tuple[float, int]]] = [[] for _ in range(g.n)]
    for a, b, w in zip(g.src.tolist(), g.dst.tolist(), g.raw.tolist()):
        neighbors[a].append((w, b))
        neighbors[b].append((w, a))
    pending = [i for i in range(g.n) if not placed[i]]
    attached = 0
    while pending:  # rounds, so the result doesn't depend on the order within a round
        batch = []
        for i in pending:
            best = max(((w, -j) for w, j in neighbors[i] if placed[j]), default=None)
            if best is not None:
                batch.append((i, -best[1]))
        if not batch:
            break
        for i, j in batch:
            angle = 2 * math.pi * _unit_hash(g.ids[i])
            pos[i] = pos[j] + unit * np.array([math.cos(angle), math.sin(angle)])
            placed[i] = True
        attached += len(batch)
        pending = [i for i in pending if not placed[i]]

    # Whatever is left has no path to the map (Louvain communities never span components):
    # whole new communities, packed as discs around the fixed footprint of the old ones,
    # then new entities with no link at all on a ring outside everything.
    fresh = sorted(set(g.comm[~placed].tolist()))
    alone = g.isolated(fresh)
    lone = set(alone)
    fresh = [c for c in fresh if c not in lone]
    if fresh:
        discs, centers, radius, fixed, local = [], [], [], [], {}
        for c in sorted(set(g.comm[placed].tolist())):
            members = g.members[c][placed[g.members[c]]]
            middle = pos[members].mean(axis=0)
            discs.append(c)
            centers.append(middle)
            radius.append(float(np.sqrt(((pos[members] - middle) ** 2).sum(axis=1)).max()) + 0.5 * unit)
            fixed.append(True)
        reach = float(np.sqrt((pos[placed] ** 2).sum(axis=1)).max()) if placed.any() else 0.0
        for j, c in enumerate(fresh):
            local[c] = g.local_layout(c) * unit
            r = float(np.sqrt((local[c] ** 2).sum(axis=1)).max()) + 0.5 * unit
            angle = _GOLDEN * j + 2 * math.pi * _unit_hash(g.ids[g.members[c][0]])
            discs.append(c)
            centers.append((reach + r + unit) * np.array([math.cos(angle), math.sin(angle)]))
            radius.append(r)
            fixed.append(False)
        slot = {c: i for i, c in enumerate(discs)}
        li, lj, lw = g.community_links()
        keep = np.array([a in slot and b in slot for a, b in zip(li.tolist(), lj.tolist())], dtype=bool)
        links = (
            np.array([slot[a] for a in li[keep].tolist()], dtype=np.int64),
            np.array([slot[b] for b in lj[keep].tolist()], dtype=np.int64),
            lw[keep],
        )
        packed = _pack(np.array(centers), np.array(radius), links, np.array(fixed), gap=unit)
        for c in fresh:
            pos[g.members[c]] = packed[slot[c]] + local[c]
            placed[g.members[c]] = True
    if alone:
        reach = float(np.sqrt((pos[placed] ** 2).sum(axis=1)).max()) + unit if placed.any() else 0.0
        pos[[g.members[c][0] for c in alone]] = _ring(len(alone), reach, 1.5 * unit)
    _refine(g, pos, ~known, unit)
    return pos, attached, sum(len(g.members[c]) for c in fresh) + len(alone)


def read_previous_layout(path: Path) -> Optional[dict[str, Any]]:
    """Positions, communities and layout facts from an existing entity_graph.json (None if
    it has no layout)."""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    meta = payload.get("layout") or {}
    positions, community = {}, {}
    for node in payload.get("nodes", []):
        x, y = node.get("x"), node.get("y")
        if node.get("id") and isinstance(x, (int, float)) and isinstance(y, (int, float)):
            positions[node["id"]] = (float(x), float(y))
            if isinstance(node.get("community"), int):
                community[node["id"]] = node["community"]
    if not positions:
        return None
    edges = payload.get("edges", [])
    linked = {e.get("source") for e in edges} | {e.get("target") for e in edges}
    return {
        "version": meta.get("version"),
        "unit": meta.get("unit"),
        "base_nodes": meta.get("base_nodes"),
        "positions": positions,
        "community": community,
        "isolated": set(positions) - linked,  # on the outer ring last time
    }


def plan_layout(graph: nx.Graph, previous: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Decide how this build lays out the map.

    "incremental" when the previous map has this layout version, holds at least half of
    today's entities and the map has less than doubled since its last full layout; then
    ``known`` keeps position and community. An entity that sat on the ring of unlinked
    entities and has gained a link is in ``rejoin`` instead: it is placed again. Otherwise
    "relayout" (a full layout turned onto the previous map) when >= 3 entities carry over,
    else "full".
    """
    prev = previous if previous and previous.get("version") == LAYOUT_VERSION else None
    old = (prev or {}).get("positions") or {}
    community = (prev or {}).get("community") or {}
    kept = {node for node in graph.nodes if node in old}
    base = int((prev or {}).get("base_nodes") or len(kept))
    unit = float((prev or {}).get("unit") or 0.0)
    n = graph.number_of_nodes()
    if (
        kept
        and 2 * len(kept) >= n
        and n < RELAYOUT_GROWTH * max(base, 1)
        and unit > 0
        and all(node in community for node in kept)
    ):
        alone = (prev or {}).get("isolated") or set()
        rejoin = {node for node in kept if node in alone and graph.degree(node) > 0}
        known = kept - rejoin
        return {"mode": "incremental", "known": known, "rejoin": rejoin, "base_nodes": base, "unit": unit,
                "positions": {node: old[node] for node in known},
                "community": {node: community[node] for node in known}}
    return {"mode": "relayout" if len(kept) >= 3 else "full", "known": kept, "rejoin": set(),
            "base_nodes": n, "unit": 0.0, "positions": {node: old[node] for node in kept}, "community": {}}


def layout(
    graph: nx.Graph,
    groups: list[list[str]],
    plan: Optional[dict[str, Any]] = None,
    prominence: Optional[dict[str, int]] = None,
):
    """Positions in the unit disc (so in [-1, 1]) for every node, and how they were made.

    plan: plan_layout() (None = full layout). prominence: rank per node (0 = most
    prominent), which orders the ring of unlinked entities.
    Returns ({node_id: (x, y)}, meta), meta = {version, mode, unit, base_nodes, kept,
    rejoined, attached, packed, seconds}.
    """
    started = time.perf_counter()
    plan = plan or {"mode": "full", "known": set(), "rejoin": set(), "positions": {}}
    g = _Graph(graph, groups)
    for node_id, rank in (prominence or {}).items():
        if node_id in g.index:
            g.tier[g.index[node_id]] = rank
    meta: dict[str, Any] = {"version": LAYOUT_VERSION, "mode": plan["mode"], "unit": 0.0,
                            "base_nodes": int(plan.get("base_nodes") or g.n), "kept": 0, "rejoined": 0,
                            "attached": 0, "packed": 0}
    if g.n == 0:
        meta.update(mode="full", seconds=0.0)
        return {}, meta
    old = plan["positions"]
    known = np.array([node_id in old for node_id in g.ids], dtype=bool)
    if plan["mode"] == "incremental":
        unit = float(plan["unit"])
        pos = np.zeros((g.n, 2))
        pos[known] = np.array([old[node_id] for node_id in g.ids if node_id in old])
        pos, attached, packed = _warm(g, pos, known, unit)
        span = _reach(pos)
        if span > 1.0:  # grew past the frame: shrink everything by one factor, about the centre
            pos *= MARGIN / span
            unit *= MARGIN / span
        meta.update(kept=int(known.sum()), rejoined=len(plan["rejoin"]), attached=attached, packed=packed)
    else:
        pos, unit = _normalize(_cold(g))
        if plan["mode"] == "relayout":  # turn, scale and shift the new map onto the old one
            rotation, scale, mean_new, mean_old = _align(
                pos[known], np.array([old[node_id] for node_id in g.ids if node_id in old])
            )
            pos = (pos - mean_new) @ rotation * scale + mean_old
            unit *= scale
            span = _reach(pos)
            if span > 1.0:
                pos *= MARGIN / span
                unit *= MARGIN / span
            meta["kept"] = int(known.sum())
    meta["unit"] = round(unit, 6)
    meta["seconds"] = round(time.perf_counter() - started, 2)
    return {node_id: (float(pos[i, 0]), float(pos[i, 1])) for i, node_id in enumerate(g.ids)}, meta


# --------------------------------------------------------------------------- export


def build(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    previous: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Communities and layout for the exported nodes and edges.

    Returns {"positions": {id: (x, y)}, "community": {id: int}, "communities": [...],
    "layout": meta}. Each community: id, label, size, x, y (centroid), radius (distance
    from the centroid that holds 90% of its members) and top (the labelled entity ids).
    """
    nodes_by_id = {node["id"]: node for node in nodes}
    graph = weighted_graph(nodes_by_id, edges)
    mentions = {node_id: int(node.get("mentions") or 0) for node_id, node in nodes_by_id.items()}
    plan = plan_layout(graph, previous)
    if plan["mode"] == "incremental":
        groups = extend_communities(graph, plan["community"], mentions)
    else:
        groups = find_communities(graph, mentions)
    order = sorted(nodes_by_id, key=lambda i: _label_rank(i, nodes_by_id))
    positions, meta = layout(graph, groups, plan, {node_id: rank for rank, node_id in enumerate(order)})
    community: dict[str, int] = {}
    communities = []
    for c, members in enumerate(groups):
        xy = np.array([positions[m] for m in members])
        middle = xy.mean(axis=0)
        spread = np.sqrt(((xy - middle) ** 2).sum(axis=1))
        label, top = community_label(members, nodes_by_id)
        for m in members:
            community[m] = c
        communities.append({
            "id": c,
            "label": label,
            "size": len(members),
            "x": round(float(middle[0]), 4),
            "y": round(float(middle[1]), 4),
            "radius": round(float(np.quantile(spread, 0.9)), 4),
            "top": top,
        })
    return {"positions": positions, "community": community, "communities": communities, "layout": meta}
