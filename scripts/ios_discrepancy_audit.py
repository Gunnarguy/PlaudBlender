#!/usr/bin/env python3
"""Audit drift between monorepo iOS sources and a standalone backup repo.

This script is intentionally read-only: it reports differences but does not
copy, delete, or overwrite files.
"""

from __future__ import annotations

import argparse
import filecmp
from pathlib import Path


def _relative_file_set(root: Path) -> set[Path]:
    files: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_file():
            files.add(path.relative_to(root))
    return files


def audit(monorepo_ios_root: Path, backup_ios_root: Path) -> int:
    mono_files = _relative_file_set(monorepo_ios_root)
    backup_files = _relative_file_set(backup_ios_root)

    only_in_monorepo = sorted(mono_files - backup_files)
    only_in_backup = sorted(backup_files - mono_files)

    changed: list[Path] = []
    for rel in sorted(mono_files & backup_files):
        left = monorepo_ios_root / rel
        right = backup_ios_root / rel
        if not filecmp.cmp(left, right, shallow=False):
            changed.append(rel)

    print("PlaudBlender iOS discrepancy audit")
    print(f"monorepo: {monorepo_ios_root}")
    print(f"backup:   {backup_ios_root}")
    print()
    print(f"changed files: {len(changed)}")
    print(f"only in monorepo: {len(only_in_monorepo)}")
    print(f"only in backup: {len(only_in_backup)}")

    if changed:
        print("\nChanged files:")
        for rel in changed:
            print(f"  - {rel}")

    if only_in_monorepo:
        print("\nOnly in monorepo:")
        for rel in only_in_monorepo:
            print(f"  - {rel}")

    if only_in_backup:
        print("\nOnly in backup:")
        for rel in only_in_backup:
            print(f"  - {rel}")

    # Non-zero when drift exists so callers can gate on this in automation.
    return 1 if (changed or only_in_monorepo or only_in_backup) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit iOS source drift vs backup repo")
    parser.add_argument(
        "--monorepo-ios-root",
        default="PlaudBlenderiOS/PlaudBlenderiOS",
        help="Path to iOS app sources inside monorepo (default: PlaudBlenderiOS/PlaudBlenderiOS)",
    )
    parser.add_argument(
        "--backup-ios-root",
        required=True,
        help="Path to iOS app sources inside backup repo (example: ../PlaudBlenderiOS.backup-before-monorepo-20260603-162251/PlaudBlenderiOS)",
    )
    args = parser.parse_args()

    monorepo_ios_root = Path(args.monorepo_ios_root).expanduser().resolve()
    backup_ios_root = Path(args.backup_ios_root).expanduser().resolve()

    if not monorepo_ios_root.exists() or not monorepo_ios_root.is_dir():
        raise SystemExit(f"Monorepo iOS root does not exist: {monorepo_ios_root}")
    if not backup_ios_root.exists() or not backup_ios_root.is_dir():
        raise SystemExit(f"Backup iOS root does not exist: {backup_ios_root}")

    return audit(monorepo_ios_root, backup_ios_root)


if __name__ == "__main__":
    raise SystemExit(main())
