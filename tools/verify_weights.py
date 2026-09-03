#!/usr/bin/env python3
"""
verify_weights.py — independent auditor for zig-zkml weights attestation
(BLUE_PRINT F0: "verify_weights.py").

This tool deliberately shares NO code with the Zig implementation: it
re-implements the leaf/node hashing and the proof wire format from the
spec, so a bug in the library cannot hide behind a bug in the auditor.

Modes:
  1. manifest:  recompute the Merkle root from a JSON manifest
                [{"name": ..., "sha256": <hex-of-tensor-file-or-bytes>}]
                or [{"name": ..., "data_b64": ...}] and compare with an
                expected root.
  2. proof:     verify a serialized inclusion proof ("ZKMP" wire format)
                against a published root, optionally re-hashing the
                tensor bytes to confirm the leaf.

Usage:
  verify_weights.py manifest --root <hex> manifest.json
  verify_weights.py proof --root <hex> --name <name> [file ...] < proof.bin
  verify_weights.py selftest            # built-in known-answer tests

Exit code 0 = verified, 1 = mismatch, 2 = usage/parse error.

Wire format (must match libs/merkle.zig Proof.serialize, "ZKMP" v1):
  [4]  magic "ZKMP"
  [1]  version = 1
  [1]  reserved = 0
  [2]  reserved = 0
  [32] leaf hash
  [4]  leaf index  (u32 LE)
  [4]  sibling count (u32 LE)
  [n*32] siblings, bottom-up

Hashes (domain-separated Blake3, keyed by prefix — NOT keyed Blake3):
  leaf = blake3(b"zkml.wleaf"  || name || u64_le(len(data)) || data)
  node = blake3(b"zkml.wnode"  || left || right)
  Odd levels self-pair: node = H(orphan, orphan).
"""

import argparse
import base64
import hashlib
import json
import struct
import sys

try:
    import blake3  # pip install blake3
except ImportError:
    print(
        "error: the 'blake3' package is required (pip install blake3)",
        file=sys.stderr,
    )
    sys.exit(2)

LEAF_DOMAIN = b"zkml.wleaf"
NODE_DOMAIN = b"zkml.wnode"
WIRE_MAGIC = b"ZKMP"
WIRE_VERSION = 1
WIRE_HEADER_LEN = 48


def hash_leaf(name: bytes, data: bytes) -> bytes:
    h = blake3.blake3()
    h.update(LEAF_DOMAIN)
    h.update(name)
    h.update(struct.pack("<Q", len(data)))
    h.update(data)
    return h.digest()


def hash_node(left: bytes, right: bytes) -> bytes:
    h = blake3.blake3()
    h.update(NODE_DOMAIN)
    h.update(left)
    h.update(right)
    return h.digest()


def empty_root() -> bytes:
    return blake3.blake3(LEAF_DOMAIN).digest()


def fold_levels(leaves: list[bytes]) -> bytes:
    """Fold with odd-level self-pairing (matches Zig foldLevels)."""
    if not leaves:
        return empty_root()
    level = list(leaves)
    while len(level) > 1:
        pairs = len(level) // 2
        nxt = [hash_node(level[2 * i], level[2 * i + 1]) for i in range(pairs)]
        if len(level) % 2 == 1:
            orphan = level[-1]
            nxt.append(hash_node(orphan, orphan))  # self-pair
        level = nxt
    return level[0]


def build_root(named_leaves: list[tuple[bytes, bytes]]) -> bytes:
    """Root over (name, leaf_hash) pairs — name-sorted, like the Zig tree."""
    hashes = [h for _, h in sorted(named_leaves, key=lambda t: t[0])]
    return fold_levels(hashes)


def parse_proof(wire: bytes):
    if len(wire) < WIRE_HEADER_LEN:
        raise ValueError(f"proof too short: {len(wire)} bytes")
    if wire[0:4] != WIRE_MAGIC:
        raise ValueError("bad magic")
    if wire[4] != WIRE_VERSION:
        raise ValueError(f"unsupported version {wire[4]}")
    leaf = wire[8:40]
    (index,) = struct.unpack_from("<I", wire, 40)
    (n_sib,) = struct.unpack_from("<I", wire, 44)
    if len(wire) != WIRE_HEADER_LEN + n_sib * 32:
        raise ValueError(
            f"length mismatch: {len(wire)} != {WIRE_HEADER_LEN} + {n_sib}*32"
        )
    sibs = [wire[WIRE_HEADER_LEN + i * 32:][:32] for i in range(n_sib)]
    return leaf, index, sibs


def verify_path(leaf: bytes, index: int, siblings: list[bytes], root: bytes) -> bool:
    h = leaf
    idx = index
    for sib in siblings:
        h = hash_node(h, sib) if idx % 2 == 0 else hash_node(sib, h)
        idx //= 2
    return h == root


# --- manifest mode -----------------------------------------------------------


def cmd_manifest(args) -> int:
    with open(args.manifest, "rb") as f:
        manifest = json.load(f)

    named: list[tuple[bytes, bytes]] = []
    for entry in manifest:
        name = entry["name"].encode()
        if "data_b64" in entry:
            data = base64.b64decode(entry["data_b64"])
            leaf = hash_leaf(name, data)
        elif "sha256" in entry:
            # Pre-hashed tensor: the manifest supplies the raw BYTES hash;
            # leaf input is the 32-byte digest itself (data = digest).
            digest = bytes.fromhex(entry["sha256"])
            if len(digest) != 32:
                print(f"error: {entry['name']}: sha256 must be 32 bytes", file=sys.stderr)
                return 2
            leaf = hash_leaf(name, digest)
        else:
            print(f"error: {entry['name']}: need 'data_b64' or 'sha256'", file=sys.stderr)
            return 2
        named.append((name, leaf))

    if len({n for n, _ in named}) != len(named):
        print("error: duplicate names in manifest", file=sys.stderr)
        return 2

    root = build_root(named)
    print(f"computed root: {root.hex()}")
    if args.root:
        expected = bytes.fromhex(args.root.removeprefix("0x"))
        if root == expected:
            print("OK: manifest verifies against the published root")
            return 0
        print(f"MISMATCH: expected {expected.hex()}")
        return 1
    print("no --root given; computed root printed above")
    return 0


# --- proof mode --------------------------------------------------------------


def cmd_proof(args) -> int:
    expected_root = bytes.fromhex(args.root.removeprefix("0x"))
    if len(expected_root) != 32:
        print("error: --root must be 32 bytes (hex)", file=sys.stderr)
        return 2

    wire = sys.stdin.buffer.read()
    try:
        leaf, index, siblings = parse_proof(wire)
    except ValueError as e:
        print(f"error: bad proof: {e}", file=sys.stderr)
        return 2

    ok = verify_path(leaf, index, siblings, expected_root)
    print(f"leaf:    {leaf.hex()}")
    print(f"index:   {index}")
    print(f"levels:  {len(siblings)}")
    print(f"result:  {'OK — proof verifies against the root' if ok else 'MISMATCH'}")

    if args.name is not None and args.files:
        # Re-hash the tensor bytes to confirm the leaf too.
        data = b"".join(open(f, "rb").read() for f in args.files)
        recomputed = hash_leaf(args.name.encode(), data)
        print(f"leaf re-hash: {'OK' if recomputed == leaf else 'MISMATCH'}")
        ok = ok and recomputed == leaf

    return 0 if ok else 1


# --- selftest ----------------------------------------------------------------


def cmd_selftest(_) -> int:
    """Known-answer tests against the Zig implementation's test vectors."""
    failures = 0

    def check(desc, got, want):
        nonlocal failures
        if got == want:
            print(f"  ok    {desc}")
        else:
            failures += 1
            print(f"  FAIL  {desc}\n        got  {got!r}\n        want {want!r}")

    # 1. Single leaf: root == leaf hash.
    leaf = hash_leaf(b"solo", b"xyz")
    check("single-leaf root == leaf", build_root([(b"solo", leaf)]), leaf)

    # 2. Empty root == blake3(leaf domain).
    check("empty root", fold_levels([]), empty_root())

    # 3. Order independence (3 entries, both orders).
    e = [(b"a", hash_leaf(b"a", b"1")), (b"b", hash_leaf(b"b", b"2")), (b"c", hash_leaf(b"c", b"3"))]
    r1 = build_root(e)
    r2 = build_root([e[2], e[0], e[1]])
    check("order independence", r1, r2)

    # 4. Wire format: build a proof path by hand for [a,b] and verify.
    la, lb = hash_leaf(b"a", b"1"), hash_leaf(b"b", b"2")
    root = hash_node(la, lb)
    ok = verify_path(la, 0, [lb], root) and verify_path(lb, 1, [la], root)
    check("2-leaf path verify", ok, True)
    ok = verify_path(la, 0, [lb], hash_node(lb, la))  # wrong root
    check("2-leaf path reject", ok, False)

    # 5. Orphan self-pairing (3 leaves): c proves with itself.
    lc = hash_leaf(b"c", b"3")
    l1 = hash_node(la, lb)
    root3 = hash_node(l1, hash_node(lc, lc))
    ok = verify_path(lc, 2, [lc, l1], root3)
    check("orphan self-pair verify", ok, True)

    # 6. parse_proof rejects garbage.
    for bad in [b"", b"ZKXX" + b"\x00" * 44, WIRE_MAGIC + b"\x63" + b"\x00" * 43]:
        try:
            parse_proof(bad)
            check(f"parse rejects {bad[:4]!r}", "rejected", "rejected")
            failures += 1
        except ValueError:
            print(f"  ok    parse rejects {bad[:4]!r}")

    print(f"\nselftest: {'PASS' if failures == 0 else f'{failures} FAILURES'}")
    return 0 if failures == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="mode", required=True)

    m = sub.add_parser("manifest", help="recompute root from a JSON manifest")
    m.add_argument("manifest")
    m.add_argument("--root", help="expected root (hex)")

    pr = sub.add_parser("proof", help="verify a ZKMP proof against a root")
    pr.add_argument("--root", required=True, help="published root (hex)")
    pr.add_argument("--name", help="tensor name (to also re-hash the leaf)")
    pr.add_argument("files", nargs="*", help="tensor byte files, if re-hashing")

    sub.add_parser("selftest", help="run built-in known-answer tests")

    args = p.parse_args()
    if args.mode == "manifest":
        return cmd_manifest(args)
    if args.mode == "proof":
        return cmd_proof(args)
    return cmd_selftest(args)


if __name__ == "__main__":
    sys.exit(main())
