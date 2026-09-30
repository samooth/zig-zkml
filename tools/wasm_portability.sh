#!/bin/bash
# Gate: what fails to compile for wasm32, pinned.
#
# `zig build` runs for one target and the seven gates all pass, so "this
# compiles" had no destination in it. This step supplies one. It does NOT
# assert that wasm compiles — wasm does not, and pretending otherwise would
# be a door that lies. It asserts the CURRENT failure, by file, against
# tools/wasm_expected.txt.
#
# Two ways it can fail, and both mean the world moved:
#   - the error set SHRINKS: something got fixed, so the list is stale
#   - the error set GROWS or CHANGES: a new portability problem appeared
#
# Either way the gate fails and names the difference. That is the point: a
# limitation nobody can reproduce is a rumour, and fixing one of these files
# turns the gate red instead of silently contradicting this file.
set -e
ZIG=${ZIG:-/home/t0m4s/.zvm/0.16.0/zig}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TARGET=${1:-wasm32-freestanding}
LOG="$ROOT/.zig-cache/wasm_portability.log"
EXPECTED="$ROOT/tools/wasm_expected.txt"
mkdir -p "$ROOT/.zig-cache"

# El fallo ES lo esperado, asi que se invierte el codigo de salida.
set +e
(cd "$ROOT" && "$ZIG" build test -Dtarget="$TARGET") > "$LOG" 2>&1
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "WASM: $TARGET COMPILA. La lista de tools/wasm_expected.txt esta vieja:" >&2
  echo "  si se arreglo de verdad, borra el fichero y quita este gate;" >&2
  echo "  si no deberia compilar, este gate no describe el fallo real." >&2
  exit 1
fi

# Ordenar DESPUES de quitar el prefijo, en los dos lados. Ordenar antes en uno
# y despues en otro hace que dos listas iguales se digan distintas: al quitar
# `/home/.../lib/std/` cambia la colacion de las mayusculas. Es el mismo fallo
# que comparar una codificacion de 8 bytes con otra de 16 — comparar dos
# cosas iguales por una diferencia de formato.
got=$(grep -E '^[^ ].*\.zig:[0-9]+:[0-9]+: error:' "$LOG" \
      | grep -oE '^[^(:]*\.zig' \
      | sed -E 's|.*/lib/std/||; s|^'"$ROOT"'/||' | sort -u || true)
want=$(sed -E 's|.*/lib/std/||; s|^'"$ROOT"'/||' "$EXPECTED" | sort -u)

if [ "$got" = "$want" ]; then
  n=$(printf '%s\n' "$want" | grep -c . )
  echo "WASM $TARGET: falla como esta previsto en $n ficheros."
  printf '%s\n' "$want" | sed 's/^/  /'
  exit 0
fi

echo "WASM $TARGET: el conjunto de errores CAMBIO respecto a $EXPECTED" >&2
echo "--- esperado ($(printf '%s\n' "$want" | grep -c . ) ficheros) ---" >&2
printf '%s\n' "$want" | sed 's/^/  /' >&2
echo "--- obtained ($(printf '%s\n' "$got" | grep -c . ) ficheros) ---" >&2
printf '%s\n' "$got" | sed 's/^/  /' >&2
echo "--- solo en lo obtenido (nuevo, o arreglado) ---" >&2
comm -13 <(printf '%s\n' "$want") <(printf '%s\n' "$got") 2>/dev/null | sed 's/^/  /' >&2
echo "--- solo en lo esperado (dejo de fallar) ---" >&2
comm -23 <(printf '%s\n' "$want") <(printf '%s\n' "$got") 2>/dev/null | sed 's/^/  /' >&2
exit 1
