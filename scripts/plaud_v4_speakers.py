"""Who is speaking, across the whole library.

Every 4.0 recording carries one 256-dim voice embedding per speaker, and the
account keeps a registry of known voices with their own embeddings. Plaud's
web app tags a file's speaker as a known voice when the cosine distance to
that voice is under 0.3 (its `matchReplaceThreshold`). This script does the
same across every recording at once, then clusters the voices that matched
nobody, so each distinct person appears once with the recordings they are in
and a few things they said -- a lineup that can be named in one pass.

    venv/bin/python scripts/plaud_v4_speakers.py           # fetch + match + report
    venv/bin/python scripts/plaud_v4_speakers.py --refresh # re-fetch embeddings

Writes data/artifacts/<id>/EMBEDDINGS.json per recording, data/speaker_map.json,
and data/voice_identities.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plaud_v4 import PlaudV4Client, classic_id  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data" / "artifacts"
MATCH_COS = 0.70      # 1 - matchReplaceThreshold(0.3): Plaud's own auto-tag bar
CLUSTER_COS = 0.70    # same bar for grouping unknown voices with each other


def cos(a, b) -> float:
    n = sum(x * y for x, y in zip(a, b))
    d = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return n / d if d else 0.0


def registry_vector(sp: dict):
    """The vector Plaud weights highest: 'me' for the owner, else 'mark', else 'auto'."""
    emb = sp.get("embeddings") or {}
    for kind in ("me", "mark", "auto"):
        v = emb.get(kind)
        if isinstance(v, list) and v:
            return v, kind
    return None, None


def load_or_fetch(client, native_id: str, cid: str, refresh: bool):
    p = ART / cid / "EMBEDDINGS.json"
    if p.exists() and not refresh:
        try:
            return json.loads(p.read_text())
        except ValueError:
            pass
    det = client.file_detail(native_id)
    emb = det.get("embeddings") or []
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(emb))
    return emb


def plaud_label_for(cid: str, label: str) -> str:
    """What Plaud's own tagging called this local speaker in the transcript."""
    p = ART / cid / "TRANSCRIPT.json"
    if not p.exists():
        return ""
    try:
        segs = json.loads(p.read_text())
    except ValueError:
        return ""
    names = {}
    for sg in segs:
        if isinstance(sg, dict) and sg.get("original_speaker") == label and sg.get("speaker"):
            names[sg["speaker"]] = names.get(sg["speaker"], 0) + 1
    return max(names, key=names.get) if names else ""


def quotes_for(cid: str, label: str, n: int = 2):
    p = ART / cid / "TRANSCRIPT.json"
    if not p.exists():
        return [], 0.0
    try:
        segs = json.loads(p.read_text())
    except ValueError:
        return [], 0.0
    mine = [s for s in segs if isinstance(s, dict) and (s.get("original_speaker") or s.get("speaker")) == label]
    talk = sum((s.get("end_time", 0) - s.get("start_time", 0)) for s in mine) / 1000.0
    best = sorted(mine, key=lambda s: -len(s.get("content") or ""))[:n]
    return [(s.get("start_time", 0) // 1000, (s.get("content") or "").strip()) for s in best], talk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    client = PlaudV4Client()
    registry = client._request("GET", "/speaker/list")["data"]["speakers"]
    reg_vecs = []
    for sp in registry:
        v, kind = registry_vector(sp)
        if v:
            reg_vecs.append((sp["speaker_id"], sp["speaker_name"], v, kind))
    print(f"registry: {len(registry)} voices, {len(reg_vecs)} with vectors", flush=True)

    started = time.time()
    files = []  # (cid, native, title, start_time)
    for rec in client.iter_recordings():
        native = str(rec.get("id") or rec.get("file_id") or "")
        if not native:
            continue
        files.append((classic_id(native), native, rec.get("title") or rec.get("name") or "", rec.get("start_time") or rec.get("start_at") or 0))
        if args.limit and len(files) >= args.limit:
            break
    print(f"recordings listed: {len(files)}", flush=True)

    assignments = []   # (cid, label, voice_key, score)
    unknown = []       # (cid, label, vec)
    fetched = failed = 0
    for i, (cid, native, title, st) in enumerate(files, 1):
        try:
            emb = load_or_fetch(client, native, cid, args.refresh)
            fetched += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            continue
        for item in emb:
            vec = item.get("embedding")
            label = item.get("label") or item.get("id")
            if not (isinstance(vec, list) and label):
                continue
            best = max(((cos(vec, rv), sid, name) for sid, name, rv, _ in reg_vecs), default=(0, None, None))
            if best[0] >= MATCH_COS:
                assignments.append((cid, label, best[1], best[0]))
            else:
                unknown.append((cid, label, vec))
        if i % 50 == 0:
            print(f"  {i}/{len(files)} · {time.time()-started:.0f}s", flush=True)
    print(f"fetched {fetched} · failed {failed} · matched {len(assignments)} speaker slots · unmatched {len(unknown)}", flush=True)

    # Cluster the unmatched voices greedily against running centroids.
    clusters = []  # [centroid_sum, count, members]
    for cid, label, vec in unknown:
        best_i, best_s = None, 0.0
        for idx, (csum, cnt, _) in enumerate(clusters):
            cen = [x / cnt for x in csum]
            s = cos(vec, cen)
            if s > best_s:
                best_i, best_s = idx, s
        if best_i is not None and best_s >= CLUSTER_COS:
            csum, cnt, members = clusters[best_i]
            clusters[best_i] = ([a + b for a, b in zip(csum, vec)], cnt + 1, members + [(cid, label)])
        else:
            clusters.append((list(vec), 1, [(cid, label)]))
    print(f"unknown voices grouped into {len(clusters)} clusters", flush=True)

    # Build the report.
    titles = {cid: (title, st) for cid, _, title, st in files}
    # Dates and devices come from the broker's rows: the listing carries neither reliably.
    try:
        import sqlite3
        from src.database.engine import DB_PATH
        con = sqlite3.connect(str(DB_PATH))
        for rid, title, created, dev in con.execute("select recording_id, title, created_at, device_id from chronos_recordings"):
            if rid in titles:
                titles[rid] = (titles[rid][0] or title or "", str(created)[:10], {"888": "Note", "860": "One"}.get(str(dev), str(dev)))
    except Exception as exc:  # noqa: BLE001
        print("db lookup failed:", exc)
    voices = {}
    reg_names = {sid: name for sid, name, _, _ in reg_vecs}
    for cid, label, sid, score in assignments:
        voices.setdefault(sid, {"name": reg_names[sid], "kind": "registry", "members": []})["members"].append((cid, label, score))
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    cl_sorted = sorted(clusters, key=lambda c: -c[1])
    for csum, cnt, members in cl_sorted:
        key = "cluster-" + next(letters, str(len(voices)))
        voices[key] = {"name": f"Unknown {key[-1]}", "kind": "cluster", "members": [(cid, label, None) for cid, label in members]}

    rows = []
    for key, v in voices.items():
        talk = 0.0; recs = set(); quotes = []; dates = []; devs = {}; plaud_says = {}
        for cid, label, _ in v["members"]:
            q, t = quotes_for(cid, label)
            talk += t; recs.add(cid)
            if len(quotes) < 4 and q:
                quotes.extend((cid, *x) for x in q[:1])
            info = titles.get(cid, ("", "", ""))
            dates.append(str(info[1])[:10] if len(info) > 1 else "")
            dev = info[2] if len(info) > 2 else "?"
            devs[dev] = devs.get(dev, 0) + 1
            pl = plaud_label_for(cid, label)
            if pl: plaud_says[pl] = plaud_says.get(pl, 0) + 1
        v["devices"] = devs; v["plaud_says"] = plaud_says
        rows.append((key, v["name"], v["kind"], len(recs), talk, sorted(d for d in dates if d), quotes, devs, plaud_says))
    rows.sort(key=lambda r: -r[4])
    fmt = lambda d: ", ".join(f"{k} ×{n}" for k, n in sorted(d.items(), key=lambda kv: -kv[1])) or "—"

    out = ["# Voices across the library", "",
           f"{len(files)} recordings · {len(assignments)} speaker slots matched a known voice · {len(unknown)} grouped into {len(clusters)} unknown voices",
           "", "Match bar: cosine ≥ 0.70, the same threshold Plaud's app uses to auto-tag a speaker.", "",
           "| voice | kind | recordings | talk time | first–last seen | device | Plaud's own label |", "|---|---|---|---|---|---|---|"]
    for key, name, kind, nrec, talk, dates, _, devs, plaud_says in rows:
        span = f"{dates[0]} – {dates[-1]}" if dates else ""
        out.append(f"| **{name}** | {kind} | {nrec} | {talk/3600:.1f} h | {span} | {fmt(devs)} | {fmt(plaud_says)} |")
    out += ["", "## Who they sound like", ""]
    for key, name, kind, nrec, talk, dates, quotes, devs, plaud_says in rows:
        if name == "Gunnar":
            continue
        out.append(f"### {name}  ·  {nrec} recordings · {talk/60:.0f} min")
        for cid, t, text in quotes[:4]:
            title = (titles.get(cid, ("", ""))[0] or cid)[:56]
            out.append(f"- *{title}* `{t//3600:02d}:{t%3600//60:02d}:{t%60:02d}` — “{text[:170]}”")
        out.append("")
    (ROOT / "data" / "voice_identities.md").write_text("\n".join(out))
    json.dump({k: {"name": v["name"], "kind": v["kind"], "members": [{"recording_id": c, "label": l, "score": s} for c, l, s in v["members"]]} for k, v in voices.items()},
              open(ROOT / "data" / "speaker_map.json", "w"), indent=1)
    print(f"\nvoices: {len(rows)} · report data/voice_identities.md · map data/speaker_map.json · {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
