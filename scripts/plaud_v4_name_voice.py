"""Name a voice once, everywhere.

Given `Speaker 2 = Kristen` (a registry voice) or `Unknown A = Richard` (a
cluster from plaud_v4_speakers.py), this:

  1. tells Plaud -- renames the registry voice, or creates one from the
     cluster's mean embedding -- through /speaker/sync, the same call the web
     app makes, so every future recording is auto-tagged with the name;
  2. rewrites the `speaker` label on every matching segment in the broker's
     held transcripts, so the app shows the name on all covered recordings
     immediately; and
  3. records the name in data/speaker_map.json.

    venv/bin/python scripts/plaud_v4_name_voice.py "Speaker 2=Kristen" "Unknown A=Richard"           # dry run
    venv/bin/python scripts/plaud_v4_name_voice.py "Speaker 2=Kristen" "Unknown A=Richard" --apply
    venv/bin/python scripts/plaud_v4_name_voice.py --merge "Unknown 7" --into Gunnar --apply           # same person, fold in

Dry run by default. Nothing is sent or rewritten without --apply.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plaud_v4 import PlaudV4Client  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data" / "artifacts"
MAP = ROOT / "data" / "speaker_map.json"


def load_map():
    return json.loads(MAP.read_text())


def find_voice(smap, label: str):
    for key, v in smap.items():
        if v["name"].lower() == label.lower():
            return key, v
    raise SystemExit(f"no voice named {label!r} in {MAP}; names are in data/voice_identities.md")


def slot_vectors(v) -> list[list[float]]:
    out = []
    for m in v["members"]:
        p = ART / m["recording_id"] / "EMBEDDINGS.json"
        if not p.exists():
            continue
        for item in json.loads(p.read_text()):
            if (item.get("label") or item.get("id")) == m["label"] and isinstance(item.get("embedding"), list):
                out.append(item["embedding"])
    return out


def rewrite_transcripts(v, new_name: str, apply: bool) -> tuple[int, int]:
    files = segs = 0
    for m in v["members"]:
        p = ART / m["recording_id"] / "TRANSCRIPT.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        changed = 0
        for sg in rows:
            if isinstance(sg, dict) and (sg.get("original_speaker") or sg.get("speaker")) == m["label"] and sg.get("speaker") != new_name:
                if apply:
                    sg["speaker"] = new_name
                changed += 1
        if changed:
            files += 1; segs += changed
            if apply:
                p.write_text(json.dumps(rows, ensure_ascii=False))
    return files, segs


def registry_object(client, name: str):
    for sp in client._request("GET", "/speaker/list")["data"]["speakers"]:
        if sp["speaker_name"] == name:
            return sp
    return None


def sync(client, speakers: list[dict], apply: bool):
    if not apply:
        print(f"   would POST /speaker/sync with {len(speakers)} speaker object(s): " + ", ".join(f"{s['speaker_name']} ({s['speaker_id'][:8]})" for s in speakers))
        return
    r = client._request("POST", "/speaker/sync", json={"speakers": speakers})
    print("   /speaker/sync ->", json.dumps(r)[:160])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="*", help='"Voice label=New name"')
    ap.add_argument("--merge", help="fold this voice into --into (same person)")
    ap.add_argument("--into")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    client = PlaudV4Client()
    smap = load_map()
    mode = "APPLY" if args.apply else "DRY RUN"

    if args.merge:
        if not args.into:
            raise SystemExit("--merge needs --into")
        key, v = find_voice(smap, args.merge)
        tkey, tv = find_voice(smap, args.into)
        files, segs = rewrite_transcripts(v, tv["name"], args.apply)
        print(f"[{mode}] fold {v['name']} into {tv['name']}: {segs} segments in {files} transcripts")
        if args.apply:
            tv["members"].extend(v["members"]); del smap[key]
            MAP.write_text(json.dumps(smap, indent=1))
        return 0

    to_sync = []
    for pair in args.pairs:
        if "=" not in pair:
            raise SystemExit(f"expected 'Voice=Name', got {pair!r}")
        label, new_name = (x.strip() for x in pair.split("=", 1))
        key, v = find_voice(smap, label)
        files, segs = rewrite_transcripts(v, new_name, args.apply)
        print(f"[{mode}] {label} -> {new_name}: {segs} segments in {files} transcripts")
        if v["kind"] == "registry":
            obj = registry_object(client, label)
            if obj is None:
                print(f"   registry voice {label!r} not found on Plaud; skipping sync")
            else:
                obj = {**obj, "speaker_name": new_name, "updated_at": int(time.time() * 1000), "need_sync": True}
                to_sync.append(obj)
        else:
            vecs = slot_vectors(v)
            if vecs:
                centroid = [sum(col) / len(vecs) for col in zip(*vecs)]
                to_sync.append({"speaker_id": uuid.uuid4().hex, "speaker_name": new_name, "speaker_type": 2,
                                "embeddings": {"mark": centroid}, "sample_counts": {"mark": len(vecs)},
                                "updated_at": int(time.time() * 1000), "need_sync": True})
                print(f"   new registry voice from {len(vecs)} embeddings")
        if args.apply:
            v["name"] = new_name
    if to_sync:
        sync(client, to_sync, args.apply)
    if args.apply:
        MAP.write_text(json.dumps(smap, indent=1))
        print("speaker_map.json updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
