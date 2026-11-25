"""Safely move a Qdrant volume root into a child directory for staged promotion.

Run with no flags first to inspect the proposed move:
    python scripts/migrate_qdrant_volume.py --root /app/qdrant_db

Only run the apply command while the application is stopped and after taking a
volume backup:
    python scripts/migrate_qdrant_volume.py --root /app/qdrant_db --apply --confirm-service-stopped
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def plan_migration(root: str | Path, target_name: str = "active") -> tuple[Path, Path, list[Path]]:
    volume_root = Path(root).resolve()
    if not volume_root.is_dir():
        raise FileNotFoundError(f"Volume root does not exist: {volume_root}")

    target = volume_root / target_name
    if target.exists():
        if any(target.iterdir()):
            raise RuntimeError(
                f"Target directory is not empty: {target}. Refusing to merge or overwrite data."
            )

    required = ("nodes.pkl", "embedding_config.json")
    missing = [name for name in required if not (volume_root / name).is_file()]
    if missing:
        raise RuntimeError(
            "Volume root is missing expected index files: " + ", ".join(missing)
        )

    candidates = [
        path for path in sorted(volume_root.iterdir(), key=lambda item: item.name)
        if path.name != target_name and path.name != "lost+found"
    ]
    if not candidates:
        raise RuntimeError("No volume-root contents were found to migrate.")
    return volume_root, target, candidates


def migrate_volume(
    root: str | Path,
    *,
    apply: bool = False,
    service_stopped: bool = False,
    target_name: str = "active",
) -> list[Path]:
    volume_root, target, candidates = plan_migration(root, target_name)
    if not apply:
        return candidates
    if not service_stopped:
        raise RuntimeError(
            "Refusing to move a live Qdrant directory. Stop the application and pass "
            "--confirm-service-stopped after taking a persistent-volume backup."
        )

    target.mkdir(exist_ok=True)
    moved: list[Path] = []
    try:
        for source in candidates:
            destination = target / source.name
            if destination.exists():
                raise FileExistsError(f"Migration destination already exists: {destination}")
            os.replace(source, destination)
            moved.append(destination)
    except Exception:
        for destination in reversed(moved):
            original = volume_root / destination.name
            if destination.exists() and not original.exists():
                os.replace(destination, original)
        if target.exists() and not any(target.iterdir()):
            target.rmdir()
        raise
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/app/qdrant_db", help="Persistent volume mount root")
    parser.add_argument("--target-name", default="active", help="Child directory for QDRANT_DIR")
    parser.add_argument("--apply", action="store_true", help="Perform the migration (default is dry-run)")
    parser.add_argument(
        "--confirm-service-stopped",
        action="store_true",
        help="Confirm the application is stopped and the volume has been backed up",
    )
    args = parser.parse_args()

    planned = migrate_volume(
        args.root,
        apply=args.apply,
        service_stopped=args.confirm_service_stopped,
        target_name=args.target_name,
    )
    if args.apply:
        print(f"Moved {len(planned)} entries into {Path(args.root) / args.target_name}")
    else:
        print("Dry run only; no files were moved.")
        for path in planned:
            print(f"  {path.name}")
        print(
            f"After backup and stopping the service, rerun with --apply "
            f"--confirm-service-stopped, then set QDRANT_DIR={Path(args.root) / args.target_name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
