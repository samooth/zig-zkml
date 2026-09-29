#!/bin/sh
# Differential of the Fp2 -> Torus61 conversion that the FRI differential needs.
# The pin version is read out of build.zig.zon, NOT globbed: measuring against
# a different dependency than the one the repo builds on is the error that
# cost four rewrites today.
#
# `fp2.zig` imports `../field.zig`, which is outside its own module path, so
# the build needs the pair staged side by side with the import rewritten. That
# means the code under test is a COPY, and a copy that silently drifts is a
# measurement of the wrong thing — so the script diffs the staged copy against
# the original and fails unless the import line is the only difference.
set -e
ZIG=${ZIG:-/home/t0m4s/.zvm/0.16.0/zig}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HASH=$(sed -n 's/.*\.hash = "zig_algebra-\(.*\)".*/\1/p' "$ROOT/build.zig.zon")
PKG="$ROOT/zig-pkg/zig_algebra-$HASH"
[ -d "$PKG" ] || { echo "falta el pin $PKG" >&2; exit 1; }
echo "pin leido de build.zig.zon: $(basename "$PKG")"

STAGE="$ROOT/.zig-cache/fri_conv"
mkdir -p "$STAGE"
cp "$ROOT/libs/field.zig" "$STAGE/field.zig"
sed 's|@import("../field.zig")|@import("./field.zig")|' \
  "$ROOT/libs/fri/fp2.zig" > "$STAGE/fp2.zig"

# La copia solo puede diferir en la linea del import.
sed 's|@import("./field.zig")|@import("../field.zig")|' "$STAGE/fp2.zig" > "$STAGE/fp2.renorm.zig"
sed 's|@import("./field.zig")|@import("../field.zig")|' "$STAGE/field.zig" > "$STAGE/field.renorm.zig"
if ! diff "$STAGE/fp2.renorm.zig" "$ROOT/libs/fri/fp2.zig" > /dev/null; then
  echo "ABORTO: la copia de fp2.zig difiere del original mas alla del import" >&2
  exit 1
fi
diff "$STAGE/field.renorm.zig" "$ROOT/libs/field.zig" > /dev/null || { echo "ABORTO: field.zig no es copia integra" >&2; exit 1; }
echo "fidelidad verificada: las copias solo difieren en la linea del import"

"$ZIG" build-exe -OReleaseSafe --dep fp2_mine --dep zig-field \
  -Mroot="$ROOT/tools/fri_conv_diff.zig" \
  -Mfp2_mine="$STAGE/fp2.zig" \
  -Mzig-field="$PKG/libs/field/src/lib.zig" \
  --cache-dir "$ROOT/.zig-cache" --global-cache-dir "$HOME/.cache/zig" \
  -femit-bin="$STAGE/fri_conv_diff" || exit 1
"$STAGE/fri_conv_diff"
