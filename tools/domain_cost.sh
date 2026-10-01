#!/bin/sh
# Measure what a domain costs, and gate `max_log_domain` on the measurement.
#
# `domain.max_log_domain = 30` is a RESOURCE limit. It was a judgement with
# nothing behind it, and the test beside it asserted a constant against itself,
# which is a tautology wearing the costume of a measurement. This script is the
# thing that makes the number a gate rather than a claim.
#
# See tools/domain_cost.zig for what is measured and why the claim is the weak
# one ("2^30 is far outside what is payable") rather than the strong one ("2^30
# is fast"), which is false.
set -e
ZIG=${ZIG:-/home/t0m4s/.zvm/0.16.0/zig}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/tools/pin_dir.sh"
PKG=$(pin_resolve)
echo "pin leido de build.zig.zon: $(basename "$PKG")"

STAGE="$ROOT/.zig-cache/domain_cost"
. "$ROOT/tools/stage_fri.sh"
stage_fri "$STAGE"

# Los mismos modulos que fri_diff.sh, por la misma razon: `libs/fri` cuelga de
# merkle y, a traves de el, del resto del paquete. Copiar la invocacion que ya
# funciona es mejor que descubrir que mas falta, aqui y otra vez.
"$ZIG" build-exe -OReleaseFast --dep fri --dep transcript --dep zig-field \
    --dep zig-algebra-traits --dep zig-hash --dep zig-bigint \
    -Mroot="$ROOT/tools/domain_cost.zig" \
    --dep transcript -Mtranscript="$ROOT/libs/transcript.zig" \
    --dep zig-merkle -Mfri="$STAGE/root.zig" \
    -Mzig-field="$PKG/libs/field/src/lib.zig" \
    -Mzig-merkle="$PKG/libs/merkle/src/root.zig" \
    -Mzig-algebra-traits="$PKG/libs/algebra-traits/src/root.zig" \
    -Mzig-hash="$PKG/libs/hash/src/root.zig" \
    -Mzig-bigint="$PKG/libs/bigint/src/root.zig" \
    --cache-dir "$ROOT/.zig-cache" --global-cache-dir "$HOME/.cache/zig" \
    -femit-bin="$STAGE/domain_cost" || exit 1

"$STAGE/domain_cost"