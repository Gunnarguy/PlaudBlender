"""Wrap raw Opus frames in an Ogg container, losslessly.

The Note uploads its audio as a bare stream of constant-bitrate Opus packets:
80-byte, 20 ms, mono, fullband frames (TOC byte 0xb8) with no container at
all, so nothing can open the file even though every frame is intact. This
puts the standard Ogg Opus framing around those packets -- OpusHead, OpusTags,
then pages of packets with a 48 kHz granule position -- without touching a
single audio byte.

    venv/bin/python scripts/opus_raw_to_ogg.py in.opus out.ogg
    venv/bin/python scripts/opus_raw_to_ogg.py --all        # every raw master under data/raw
    venv/bin/python scripts/opus_raw_to_ogg.py --all --dry-run
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

FRAME_BYTES = 80          # 32 kbps CBR at 20 ms
SAMPLES_PER_FRAME = 960   # 20 ms at Opus's 48 kHz clock
PACKETS_PER_PAGE = 50     # one second per page

_CRC_TABLE = []
for _i in range(256):
    _r = _i << 24
    for _ in range(8):
        _r = ((_r << 1) ^ 0x04C11DB7) if _r & 0x80000000 else (_r << 1)
    _CRC_TABLE.append(_r & 0xFFFFFFFF)


def _crc_reference(data: bytes) -> int:
    """Ogg's CRC-32: polynomial 0x04C11DB7, non-reflected, init 0, no final xor."""
    crc = 0
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _CRC_TABLE[((crc >> 24) & 0xFF) ^ byte]
    return crc


_REV8 = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))


def _crc(data: bytes) -> int:
    """The same CRC at C speed. zlib's CRC-32 is the bit-reflected form of the
    same polynomial, so reflecting the input bytes and the 32-bit result turns
    zlib.crc32 (with init/xor-out cancelled) into Ogg's checksum."""
    reflected = zlib.crc32(data.translate(_REV8), 0xFFFFFFFF) ^ 0xFFFFFFFF
    return int(f"{reflected:032b}"[::-1], 2)


assert _crc(b"OggS test page") == _crc_reference(b"OggS test page")


def _page(serial: int, seq: int, granule: int, packets: list[bytes], flags: int = 0) -> bytes:
    segments = bytearray()
    for p in packets:
        n = len(p)
        while n >= 255:
            segments.append(255)
            n -= 255
        segments.append(n)
    header = bytearray(b"OggS") + struct.pack("<BBqIIIB", 0, flags, granule, serial, seq, 0, len(segments)) + segments
    body = b"".join(packets)
    page = bytes(header) + body
    crc = _crc(page)
    return page[:22] + struct.pack("<I", crc) + page[26:]


def looks_raw_opus(data: bytes) -> bool:
    """Constant TOC byte at every frame boundary, and a size that is whole frames."""
    if len(data) < FRAME_BYTES * 4 or len(data) % FRAME_BYTES:
        return False
    if data[:4] == b"OggS":
        return False
    toc = data[0]
    return all(data[i] == toc for i in range(0, min(len(data), FRAME_BYTES * 200), FRAME_BYTES))


def wrap(raw: bytes, *, channels: int = 1, pre_skip: int = 0, input_rate: int = 16000, serial: int = 0x5049) -> bytes:
    # pre_skip 0 / 16 kHz mono mirror the OpusHead the recorder itself writes.
    head = b"OpusHead" + struct.pack("<BBHIhB", 1, channels, pre_skip, input_rate, 0, 0)
    vendor = b"plaudblender raw-opus wrap"
    tags = b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", 0)
    out = bytearray()
    out += _page(serial, 0, 0, [head], flags=0x02)   # beginning of stream
    out += _page(serial, 1, 0, [tags])
    frames = [raw[i:i + FRAME_BYTES] for i in range(0, len(raw), FRAME_BYTES)]
    seq = 2
    granule = 0
    for start in range(0, len(frames), PACKETS_PER_PAGE):
        chunk = frames[start:start + PACKETS_PER_PAGE]
        granule += SAMPLES_PER_FRAME * len(chunk)
        last = start + PACKETS_PER_PAGE >= len(frames)
        out += _page(serial, seq, granule, chunk, flags=0x04 if last else 0)
        seq += 1
    return bytes(out)


def verify_crc_against(path: Path) -> bool:
    """Recompute the CRC of a real Ogg file's first page and compare to what it stores."""
    b = path.read_bytes()[:65536]
    if b[:4] != b"OggS":
        return False
    nseg = b[26]
    body_len = sum(b[27:27 + nseg])
    page = b[:27 + nseg + body_len]
    stored = struct.unpack("<I", page[22:26])[0]
    zeroed = page[:22] + b"\0\0\0\0" + page[26:]
    return _crc(zeroed) == stored


def probe_seconds(path: Path) -> float | None:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=60)
        return float(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def convert(src: Path, dst: Path) -> tuple[bool, str]:
    raw = src.read_bytes()
    if not looks_raw_opus(raw):
        return False, "not raw opus"
    expected = len(raw) / FRAME_BYTES * 0.02
    dst.write_bytes(wrap(raw))
    got = probe_seconds(dst)
    if got is None or abs(got - expected) > max(2.0, expected * 0.01):
        dst.unlink(missing_ok=True)
        return False, f"probe {got} vs expected {expected:.1f}s"
    return True, f"{expected/3600:.2f} h"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?"); ap.add_argument("dst", nargs="?")
    ap.add_argument("--all", action="store_true", help="convert every raw master under data/raw and repoint the database")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-raw", action="store_true", help="do not delete the raw file after a verified conversion")
    ap.add_argument("--verify-crc", help="recompute the first-page CRC of a genuine .ogg and compare")
    args = ap.parse_args()
    if args.verify_crc:
        ok = verify_crc_against(Path(args.verify_crc)); print("crc matches genuine page:", ok); return 0 if ok else 1
    if args.src and args.dst:
        ok, why = convert(Path(args.src), Path(args.dst)); print(("ok " if ok else "FAIL ") + why); return 0 if ok else 1
    if not args.all:
        ap.print_help(); return 2

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sqlalchemy import text as sqltext
    from src.database import SessionLocal
    session = SessionLocal()
    rows = session.execute(sqltext("select recording_id, local_audio_path from chronos_recordings where local_audio_path like '%.opus'")).all()
    print(f"{len(rows)} rows point at .opus files")
    done = skipped = failed = 0
    freed = 0
    for rid, path in rows:
        src = Path(path)
        if not src.exists():
            skipped += 1; continue
        dst = src.with_suffix(".ogg")
        if dst.exists() and probe_seconds(dst):
            ok, why = True, "already wrapped"
        elif args.dry_run:
            ok, why = looks_raw_opus(src.read_bytes()[:FRAME_BYTES * 200 + 1] + b"\0" * ((FRAME_BYTES - len(src.read_bytes()) % FRAME_BYTES) % FRAME_BYTES)), "would wrap"
            print(f"  {'raw opus' if ok else 'other   '}  {src.name}"); continue
        else:
            ok, why = convert(src, dst)
        if ok:
            session.execute(sqltext("update chronos_recordings set local_audio_path=:p where recording_id=:i"), {"p": str(dst), "i": rid})
            if not args.keep_raw and src.exists() and dst.exists():
                freed += src.stat().st_size; src.unlink()
            done += 1
        else:
            failed += 1; print(f"  FAIL {src.name}: {why}")
        if (done + failed) % 25 == 0:
            session.commit(); print(f"  {done} wrapped · {failed} failed · {freed/1e9:.1f} GB raw removed", flush=True)
    session.commit()
    print(f"\nwrapped {done} · failed {failed} · missing {skipped} · raw removed {freed/1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
