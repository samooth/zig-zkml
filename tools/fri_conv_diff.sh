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
  "$ROOT/libs/torus/fp2.zig" > "$STAGE/fp2.zig"

# La copia solo puede diferir en la linea del import.
sed 's|@import("./field.zig")|@import("../field.zig")|' "$STAGE/fp2.zig" > "$STAGE/fp2.renorm.zig"
sed 's|@import("./field.zig")|@import("../field.zig")|' "$STAGE/field.zig" > "$STAGE/field.renorm.zig"
if ! diff "$STAGE/fp2.renorm.zig" "$ROOT/libs/torus/fp2.zig" > /dev/null; then
  echo "ABORTO: la copia de fp2.zig difiere del original mas alla del import" >&2
  exit 1
fi
diff "$STAGE/field.renorm.zig" "$ROOT/libs/field.zig" > /dev/null || { echo "ABORTO: field.zig no es copia integra" >&2; exit 1; }
  # Ganchos de mutacion. El mismo diseno que field_diff.sh y fri_diff.sh: cada
  # patron se comprueba con grep, y si no cae el script ABORTA en vez de
  # seguir. Sin esa comprobacion un gancho puede no mutar nada y devolver un
  # 0 que se lee como una medicion.
  if [ -n "${FRI_CONV_DIFF_MUTATE:-}" ]; then
    echo "MUTACION activa en MI fp2.zig: $FRI_CONV_DIFF_MUTATE"
    case "$FRI_CONV_DIFF_MUTATE" in
      toBytes-coords-intercambiadas)
        # a y b se escriben en el orden contrario. Es la mutacion de mas
        # superficie de la conversion, y el layout de bytes es justo lo que
        # este instrumento comprueba.
        perl -0pi -e 's/x\.a\.toBytes\(out\[0\.\.8\]\);\n(\s*)x\.b\.toBytes\(out\[8\.\.16\]\);/x.b.toBytes(out[0..8]);\n$1x.a.toBytes(out[8..16]); \/\/ MUTADO/m' "$STAGE/fp2.zig"
        grep -q MUTADO "$STAGE/fp2.zig" || { echo "ABORTO: la mutacion no aplico" >&2; exit 2; } ;;
      fromBytes-sin-canonico)
        # La guarda de canonicidad pasa a comprobar SOLO a. Sigue usando las
        # dos variables, porque quitarlas deja ra y rb sin usar y el fallo que
        # sale es de compilacion, no del diferencial. El comentario del propio
        # codigo dice que la guarda hace falta para que el muestreo por rechazo
        # sea uniforme, asi que es la mutacion que mas directamente toca el
        # transcript: a podria llegar no canonico y el parser lo aceptaria.
        # El signo de la comparacion de b pasa de >= a >, de modo que rb == p
        # deja de rechazarse. p es exactamente el primo: una codificacion con
        # b == p es NO canonica, y el comentario del propio codigo dice que la
        # guarda existe para que el muestreo por rechazo sea uniforme. Sin la
        # guarda, un valor no canonico pasaria al transcript.
        #
        # El patron no lleva grupo de captura a proposito: escribirlo con
        # parentesis escapados abre un grupo que el \) literal no cierra, y perl
        # responde "Unmatched (" sin tocar el fichero y sin salir con codigo de
        # error en el sitio que se mira. Por eso el guard comprueba la marca y
        # no solo el codigo de salida.
        perl -0pi -e 's{rb >= Goldilocks\.p\) return error\.OutOfField;}{rb > Goldilocks.p) return error.OutOfField; // MUTADO}' "$STAGE/fp2.zig"
        grep -q MUTADO "$STAGE/fp2.zig" || { echo "ABORTO: la mutacion no aplico" >&2; exit 2; } ;;
      *) echo "mutacion desconocida: $FRI_CONV_DIFF_MUTATE" >&2; exit 2 ;;
    esac
  else
    echo "fidelidad verificada: las copias solo difieren en la linea del import"
  fi

"$ZIG" build-exe -OReleaseSafe --dep fp2_mine --dep zig-field \
  -Mroot="$ROOT/tools/fri_conv_diff.zig" \
  -Mfp2_mine="$STAGE/fp2.zig" \
  -Mzig-field="$PKG/libs/field/src/lib.zig" \
  --cache-dir "$ROOT/.zig-cache" --global-cache-dir "$HOME/.cache/zig" \
  -femit-bin="$STAGE/fri_conv_diff" || exit 1
"$STAGE/fri_conv_diff"
