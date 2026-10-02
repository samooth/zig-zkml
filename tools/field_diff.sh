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
ROOT=$(cd "$(dirname "$0")/.." && pwd)

# El binario de Zig se resuelve UNA vez, en tools/zig_bin.sh. Estaba escrito
# a mano en seis guiones con la ruta de una sola maquina, y en CI —donde
# setup-zig lo pone en el PATH— todos ellos morian al instante.
. "$ROOT/tools/zig_bin.sh"
ZIG=$(zig_bin)
. "$ROOT/tools/pin_dir.sh"
PKG=$(pin_resolve)
echo "pin leido de build.zig.zon: $(basename "$PKG")"

STAGE="$ROOT/.zig-cache/field_diff"
mkdir -p "$STAGE"

# MI LADO SE COPIA A POR ETAPAS. Antes compilaba directamente contra
# libs/field.zig, y por eso no habia ganchos de mutacion: mutar en vivo
# ensucia el arbol y, si el proceso muere a mitad, se deja el repositorio
# modificado. fri_diff.sh ya copiaba por etapas por el motivo del pin; aqui el
# motivo era el propio repositorio.
cp "$ROOT/libs/field.zig" "$STAGE/field_mine.zig"

# fidelidad: la copia solo puede diferir si una mutacion esta activa, y entonces
# la fidelidad ya no aplica y se dice.
mutated=0
if [ -n "${FIELD_DIFF_MUTATE:-}" ]; then
  echo "MUTACION activa en MI field.zig: $FIELD_DIFF_MUTATE"
  mutated=1
  # Cada mutacion comprueba que su patron ATTERRIZO, con grep, y aborta si no.
  # Un print que anuncia una sustitucion que no ocurrio es el fallo mas caro
  # que hay: el 0 que devuelve el diferencial se lee como una medicion. Asi se
  # detecto el gancho de MI lado en fri_diff.sh, que estaba vacio.
  case "$FIELD_DIFF_MUTATE" in
    add-sin-reducir)
      # La reduccion desaparece. Es la mutacion mas antigua del campo y la que
      # mas se ha medido: sin ella, add devuelve un rep >= p.
      perl -0pi -e 's/return \.\{ \.rep = if \(s >= p\) s - p else s \};\s*$/return .{ .rep = s }; \/\/ MUTADO/m' "$STAGE/field_mine.zig"
      grep -q MUTADO "$STAGE/field_mine.zig" || { echo "ABORTO: la mutacion no aplico" >&2; exit 2; } ;;
    sub-invertido)
      # a - b pasa a ser b - a. El signo cambia en elreduction, asi que la mitad
      # del dominio coincide por simetria y la otra mitad no.
      perl -0pi -e 's/const d = a\.rep \+ p - b\.rep;/const d = b.rep + p - a.rep; \/\/ MUTADO/' "$STAGE/field_mine.zig"
      grep -q MUTADO "$STAGE/field_mine.zig" || { echo "ABORTO: la mutacion no aplico" >&2; exit 2; } ;;
    eql-reflexivo)
      # Compara cada operando consigo mismo: siempre verdadero. La forma usa
      # los dos parametros, porque `return true` deja parametros sin usar y el
      # fallo que salta es de compilacion, no del diferencial.
      perl -0pi -e 's/return a\.rep == b\.rep;/return a.rep == a.rep or b.rep == b.rep; \/\/ MUTADO/' "$STAGE/field_mine.zig"
      grep -q MUTADO "$STAGE/field_mine.zig" || { echo "ABORTO: la mutacion no aplico" >&2; exit 2; } ;;
    mul-shift-60)
      # El plegado de 61 a 60 bits. La de mas superficie: toca todos los
      # productos. La marca va al final de la linea, porque en linea se come el
      # parentesis de cierre de @intCast y el fallo que sale es de sintaxis, no
      # del diferencial.
      perl -0pi -e 's/(prod >> )61(\);\s*)$/$1 60\); \/\/ MUTADO/m' "$STAGE/field_mine.zig"
      grep -q MUTADO "$STAGE/field_mine.zig" || { echo "ABORTO: la mutacion no aplico" >&2; exit 2; } ;;
    inv-p-menos-3)
      # El exponente del inverso: p-2 pasa a p-3, que no es el inverso de nada.
      perl -0pi -e 's/return a\.pow\(p - 2\);/return a.pow(p - 3); \/\/ MUTADO/' "$STAGE/field_mine.zig"
      grep -q MUTADO "$STAGE/field_mine.zig" || { echo "ABORTO: la mutacion no aplico" >&2; exit 2; } ;;
    neg-mas-uno)
      perl -0pi -e 's/else p - a\.rep \};/else p - a.rep + 1 }; \/\/ MUTADO/' "$STAGE/field_mine.zig"
      grep -q MUTADO "$STAGE/field_mine.zig" || { echo "ABORTO: la mutacion no aplico" >&2; exit 2; } ;;
    *) echo "mutacion desconocida: $FIELD_DIFF_MUTATE" >&2; exit 2 ;;
  esac
else
  # Sin mutacion, la copia TIENE que ser identica. Un fallo aqui significa que
  # el staged quedo sucio de una corrida anterior.
  if ! diff "$STAGE/field_mine.zig" "$ROOT/libs/field.zig" > /dev/null; then
    echo "ABORTO: la copia de field.zig difiere del original" >&2
    exit 1
  fi
fi

"$ZIG" build-exe -OReleaseSafe --dep field_mine --dep zig-field \
    -Mroot="$ROOT/tools/field_diff.zig" \
    -Mfield_mine="$STAGE/field_mine.zig" \
    -Mzig-field="$PKG/libs/field/src/lib.zig" \
    --cache-dir "$ROOT/.zig-cache" --global-cache-dir "$HOME/.cache/zig" \
    -femit-bin="$STAGE/field_diff" || exit 1
# The earlier version of this script ended in `exec`, so it built the
# differential and never ran it, and printed nothing and exited 0. A
# measurement that produces no output is not a measurement.
"$STAGE/field_diff"