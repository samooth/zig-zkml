#!/usr/bin/env python3
"""
verify_llama_adapter.py — cross-check the llama.cpp adapter's attestation
root against the independent auditor (tools/verify_weights.py).

Thin integration wrapper: reads the artifacts emitted by
`zkml_llama_test <workdir>` (root.hex + manifest.json) and delegates to
verify_weights.py's manifest mode. Kept separate so the adapter gate has a
single, documented entry point (PLAN_MULTI_ENGINE Stage 1).

Usage:
  verify_llama_adapter.py <workdir>          # root.hex + manifest.json
  verify_llama_adapter.py --root HEX --manifest path.json

Exit 0 = verified, 1 = mismatch, 2 = usage/missing artifacts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    # tools/integration/ -> repo root is two levels up.
    return Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n")[1] if __doc__ else "verify llama adapter"
    )
    p.add_argument("workdir", nargs="?", help="directory with root.hex + manifest.json")
    p.add_argument("--root", help="expected root (hex); overrides workdir/root.hex")
    p.add_argument("--manifest", help="manifest.json path; overrides workdir/manifest.json")
    args = p.parse_args()

    root_hex = args.root
    manifest = args.manifest

    if args.workdir:
        wd = Path(args.workdir)
        if root_hex is None:
            rf = wd / "root.hex"
            if not rf.is_file():
                print(f"error: missing {rf}", file=sys.stderr)
                return 2
            root_hex = rf.read_text().strip()
        if manifest is None:
            mf = wd / "manifest.json"
            if not mf.is_file():
                print(f"error: missing {mf}", file=sys.stderr)
                return 2
            manifest = str(mf)

    if not root_hex or not manifest:
        p.error("need either a workdir with root.hex+manifest.json, or --root and --manifest")

    verifier = _repo_root() / "tools" / "verify_weights.py"
    if not verifier.is_file():
        print(f"error: missing {verifier}", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        str(verifier),
        "manifest",
        "--root",
        root_hex.removeprefix("0x"),
        manifest,
    ]
    print(f"+ {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
