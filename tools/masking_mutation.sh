#!/usr/bin/env bash
# Mutation gate for the masking scaffolding.
#
# The tests in libs/stark/masking.zig claim a *negative* result: masking a
# committed column by a multiple of Z_H does not hide it, because the verifier
# recovers the column as `g mod Z_H`. A claim like that is exactly the kind that
# can be true by accident of one implementation, so each mutation below breaks
# one step of the argument and the gate is green only when the tests go RED.
#
# A gate that has never been shown to fail is not a gate, so this script fails if
# a mutation lands and the suite still passes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source tools/zig_bin.sh
ZIG=$(zig_bin)

TARGET="libs/stark/masking.zig"
BACKUP="$(mktemp)"
cp "$TARGET" "$BACKUP"
restore() { cp "$BACKUP" "$TARGET"; rm -f "$BACKUP"; }
trap restore EXIT

FOLD='out[i % n] = out[i % n].add(c);'

mutate() {
  local name="$1" from="$2" to="$3"
  # Each mutation starts from the committed file, so they cannot mask each other.
  cp "$BACKUP" "$TARGET"
  python3 - "$TARGET" "$from" "$to" <<'PY'
import sys
path, frm, to = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding='utf8').read()
if s.count(frm) != 1:
    sys.exit(f"ancla no unica ({s.count(frm)}): {frm}")
open(path, 'w', encoding='utf8').write(s.replace(frm, to))
PY
  # The pattern must be present after the edit, or we measured nothing.
  if ! grep -qF "$to" "$TARGET"; then
    echo "  MUTACION $name: el patron no quedo" >&2
    exit 1
  fi
}

expect_red() {
  local name="$1"
  if "$ZIG" build test --summary all >/dev/null 2>&1; then
    echo "  $name: la suite SIGUE EN VERDE -- la mutacion no muerde" >&2
    exit 1
  fi
  echo "  $name: la suite cae, como debe"
}

expect_green() {
  local name="$1"
  if ! "$ZIG" build test --summary all >/dev/null 2>&1; then
    echo "  $name: la suite cae sin mutacion" >&2
    exit 1
  fi
  echo "  $name: verde sin mutacion"
}

expect_green "sin mutacion"

# 1. Fold to the wrong exponent class. The reduction still happens, but the
#    remainder comes back rotated -- so the column is not recovered and the two
#    falsification tests must notice.
mutate "fold-desplazado" \
  "$FOLD" \
  'out[(i + 1) % n] = out[(i + 1) % n].add(c);'
expect_red "fold-desplazado"

# 2. Do not reduce at all. This is the mistake a reader is most likely to make
#    when the claim is "the remainder is f": return the dividend.
mutate "sin-reducir" \
  "$FOLD" \
  'out[i % n] = out[i % n].add(c);'
python3 - "$TARGET" <<'PY'
import sys
p = sys.argv[1]
s = open(p, encoding='utf8').read()
old = "    for (poly, 0..) |c, i| {\n        out[i % n] = out[i % n].add(c);\n    }\n    return out;"
new = "    _ = poly;\n    return out;"
assert s.count(old) == 1
open(p, 'w', encoding='utf8').write(s.replace(old, new))
PY
grep -qF "_ = poly;" "$TARGET" || { echo "  sin-reducir: no quedo" >&2; exit 1; }
expect_red "sin-reducir"

restore
trap - EXIT

echo "  masking-mutation: las tres comprobaciones se comportan como deben"