#!/bin/sh
# Differential: libs/field.zig against the pinned zig-algebra M61.
# Kept as a script so the measurement can be repeated: a number measured with
# a tool that does not exist tomorrow is not a number anybody can check.
#
# The pin version is read out of build.zig.zon, NOT globbed. An earlier
# version of this script took `ls zig-pkg/zig_algebra-* | head -1`, which
# resolved to v0.3.0 — a different dependency than the one the repo builds
# against. Measuring the wrong version is the same error as citing the right
# version and inferring the content.
set -e
ZIG=${ZIG:-/home/t0m4s/.zvm/0.16.0/zig}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HASH=$(sed -n 's/.*\.hash = "zig_algebra-\(.*\)".*/\1/p' "$ROOT/build.zig.zon")
PKG="$ROOT/zig-pkg/zig_algebra-$HASH"
if [ ! -d "$PKG" ]; then echo "falta el pin $PKG" >&2; exit 1; fi
echo "pin leido de build.zig.zon: $(basename "$PKG")"
mkdir -p "$ROOT/.zig-cache/field_diff"
"$ZIG" build-exe -OReleaseSafe --dep field_mine --dep zig-field \
  -Mroot="$ROOT/tools/field_diff.zig" \
  -Mfield_mine="$ROOT/libs/field.zig" \
  -Mzig-field="$PKG/libs/field/src/lib.zig" \
  --cache-dir "$ROOT/.zig-cache" --global-cache-dir "$HOME/.cache/zig" \
  -femit-bin="$ROOT/.zig-cache/field_diff/field_diff" || exit 1
# The earlier version of this script ended in `exec`, so it built the
# differential and never ran it, and printed nothing and exited 0. A
# measurement that produces no output is not a measurement.
"$ROOT/.zig-cache/field_diff/field_diff"
