"""Build the memory graph for one or more UIDs.

    properagent-build-memory --uid b1000_cleaning_laundry_0001
    properagent-build-memory --all --workers 4

Construction is the expensive, one-off stage: a year-long history is ~900 events
and each needs three LLM calls. Already-built UIDs are skipped unless --force.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..backends import build_chat_backend, build_embedding_backend
from ..config import load_config, section
from ..data import Dataset
from ..memory.build import BuildConfig, build_memory
from ..memory.store import MemoryPaths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="path to a YAML config (defaults to configs/default.yaml)")
    parser.add_argument("--set", action="append", dest="overrides", default=[], metavar="KEY=VALUE")
    parser.add_argument("--uid", action="append", default=[], help="repeatable; omit with --all")
    parser.add_argument("--all", action="store_true", help="build every UID in the eval set")
    parser.add_argument("--force", action="store_true", help="rebuild UIDs that already exist")
    parser.add_argument("--workers", type=int, default=None, help="concurrent events per UID")
    args = parser.parse_args(argv)

    config = load_config(args.config, args.overrides)
    dataset = Dataset(config["data"]["root"], config["data"].get("uid_list"))

    uids = list(args.uid)
    if args.all:
        uids = list(dataset.uids())
    if not uids:
        parser.error("pass --uid UID (repeatable) or --all")

    memory_root = Path(config["memory"]["root"])
    build_config = BuildConfig()
    if args.workers:
        build_config = BuildConfig(**{**build_config.__dict__, "workers": args.workers})

    extractor = build_chat_backend(section(config, "gate_backend"))
    embedder = build_embedding_backend(section(config, "embedding"))

    for position, uid in enumerate(uids, start=1):
        paths = MemoryPaths.for_uid(memory_root, uid)
        if not args.force and not paths.missing():
            print(f"[{position}/{len(uids)}] {uid}: already built, skipping")
            continue

        def progress(done: int, total: int, uid: str = uid, position: int = position) -> None:
            if done % 25 == 0 or done == total:
                print(f"[{position}/{len(uids)}] {uid}: {done}/{total} events", flush=True)

        captions = dataset.captions(uid)
        print(f"[{position}/{len(uids)}] {uid}: {len(captions)} captions", flush=True)
        manifest = build_memory(
            uid,
            captions,
            memory_root,
            extractor=extractor,
            embedder=embedder,
            config=build_config,
            progress=progress,
        )
        counts = manifest["counts"]
        note = (
            f" ({counts['failed_events']} events failed)" if counts["failed_events"] else ""
        )
        print(
            f"[{position}/{len(uids)}] {uid}: {counts['events']} events, "
            f"{counts['triples']} triples, {counts['semantic']} standing facts{note}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
