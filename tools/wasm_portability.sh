#!/bin/bash
# Gate: what fails to compile for wasm32, pinned.
#
# `zig build` runs for one target and the seven gates all pass, so "this
# compiles" had no destination in it. This step supplies one. It does NOT
# assert that wasm compiles — wasm does not, and pretending otherwise would
# be a door that lies. It asserts the CURRENT failure, by file, against
# tools/wasm_test_sweep_expected.txt.
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
EXPECTED="$ROOT/tools/wasm_test_sweep_expected.txt"
mkdir -p "$ROOT/.zig-cache"

# El fallo ES lo esperado, asi que se invierte el codigo de salida.
set +e
(cd "$ROOT" && "$ZIG" build test -Dtarget="$TARGET") > "$LOG" 2>&1
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "WASM: $TARGET COMPILA. La lista de tools/wasm_test_sweep_expected.txt esta vieja:" >&2
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
# El fichero esperado puede explicar cada entrada: se ignoran los comentarios
# y se lee solo el primer campo de cada linea. Una lista que no puede decir por
# que contiene lo que contiene es la mitad de una lista de tickets.
want=$(grep -vE '^[[:space:]]*(#|$)' "$EXPECTED" \
      | sed -E 's/[[:space:]].*$//' \
      | sed -E 's|.*/lib/std/||; s|^'"$ROOT"'/||' | sort -u)

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
# Las dos direcciones significan cosas DISTINTAS y hay que nombrarlas bien.
# "Falla y no estaba en la lista" NO es "nuevo": tambien es "la lista esta
# incompleta", y con la etiqueta equivocada invita a regenerar la lista sin
# mirar, que es la version silenciosa de mover la cifra para que la puerta pase.
extra=$(comm -13 <(printf '%s\n' "$want") <(printf '%s\n' "$got"))
fixed=$(comm -23 <(printf '%s\n' "$want") <(printf '%s\n' "$got"))
echo "--- falla y NO estaba en la lista (problema nuevo, o lista incompleta) ---" >&2
[ -n "$extra" ] && printf '%s\n' "$extra" | sed 's/^/  /' >&2 || echo "  (ninguno)" >&2
echo "--- estaba en la lista y DEJO de fallar (alguien lo arreglo) ---" >&2
[ -n "$fixed" ] && printf '%s\n' "$fixed" | sed 's/^/  /' >&2 || echo "  (ninguno)" >&2
echo >&2
echo "Si algo DEJO de fallar, actualiza tools/wasm_test_sweep_expected.txt en el MISMO" >&2
echo "commit que lo arreglo, y el mensaje dice que salio y por que. Una" >&2
echo "lista que cambia de nueve a ocho sin explicacion es una afirmacion" >&2
echo "que se ha movido sola." >&2
exit 1
