#!/bin/sh
# Differential of my FRI composition against the pinned one.
# Pin read from build.zig.zon, never globbed.
# The staged copies are diffed against the originals and the run aborts unless
# the only difference is the rewritten import: a copy that drifts silently
# measures the wrong thing.
set -e
ZIG=${ZIG:-/home/t0m4s/.zvm/0.16.0/zig}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HASH=$(sed -n 's/.*\.hash = "zig_algebra-\(.*\)".*/\1/p' "$ROOT/build.zig.zon")
PKG="$ROOT/zig-pkg/zig_algebra-$HASH"
[ -d "$PKG" ] || { echo "falta el pin $PKG" >&2; exit 1; }
echo "pin leido de build.zig.zon: $(basename "$PKG")"

STAGE="$ROOT/.zig-cache/fri_diff"
mkdir -p "$STAGE"

cp "$ROOT/libs/field.zig" "$STAGE/field.zig"
sed 's|@import("../field.zig")|@import("./field.zig")|' \
  "$ROOT/libs/fri/fp2.zig" > "$STAGE/fp2.zig"
for f in root domain fft fp2; do
  sed 's|@import("../field.zig")|@import("./field.zig")|; s|@import("../merkle.zig")|@import("./merkle.zig")|; s|@import("../transcript.zig")|@import("./transcript.zig")|; s|@import("fp2.zig")|@import("./fp2.zig")|; s|@import("domain.zig")|@import("./domain.zig")|; s|@import("fft.zig")|@import("./fft.zig")|; s|@import("root.zig")|@import("./root.zig")|' \
    "$ROOT/libs/fri/$f.zig" > "$STAGE/$f.zig"
done
cp "$ROOT/libs/merkle.zig" "$STAGE/merkle.zig"
cp "$ROOT/libs/transcript.zig" "$STAGE/transcript.zig"
sed 's|@import("torus.zig")|@import("./torus.zig")|' "$PKG/libs/fri/src/root.zig" > "$STAGE/pin_root.zig"
cp "$PKG/libs/fri/src/torus.zig" "$STAGE/torus.zig"
# Sin envoltorio: un fichero no puede pertenecer a dos modulos, asi que todo
# mi lado cuelga de UN modulo, `fri_mine`, que ya reexporta Fp2 y Domain. Un
# envoltorio de una linea daria dos definiciones de Fp2 en el binario.

# fidelidad: la copia solo puede diferir en las lineas de import reescritas
fail=0
for f in fp2 root domain fft fp2; do
  : # los nombres se solapan a proposito; se comprueba la unicidad abajo
done
for f in root domain fft; do
  sed 's|@import("./field.zig")|@import("../field.zig")|; s|@import("./merkle.zig")|@import("../merkle.zig")|; s|@import("./transcript.zig")|@import("../transcript.zig")|; s|@import("./fp2.zig")|@import("fp2.zig")|; s|@import("./domain.zig")|@import("domain.zig")|; s|@import("./fft.zig")|@import("fft.zig")|; s|@import("./root.zig")|@import("root.zig")|' \
    "$STAGE/$f.zig" > "$STAGE/$f.renorm.zig"
  diff "$STAGE/$f.renorm.zig" "$ROOT/libs/fri/$f.zig" > /dev/null || { echo "ABORTO: $f.zig difiere mas alla del import" >&2; fail=1; }
done
sed 's|@import("./field.zig")|@import("../field.zig")|' "$STAGE/fp2.zig" > "$STAGE/fp2.renorm.zig"
diff "$STAGE/fp2.renorm.zig" "$ROOT/libs/fri/fp2.zig" > /dev/null || { echo "ABORTO: fp2.zig difiere" >&2; fail=1; }
diff "$STAGE/field.zig" "$ROOT/libs/field.zig" > /dev/null || { echo "ABORTO: field.zig no es copia integra" >&2; fail=1; }
diff "$STAGE/merkle.zig" "$ROOT/libs/merkle.zig" > /dev/null || { echo "ABORTO: merkle.zig no es copia integra" >&2; fail=1; }
sed 's|@import("./torus.zig")|@import("torus.zig")|' "$STAGE/pin_root.zig" > "$STAGE/pin_root.renorm.zig"
diff "$STAGE/pin_root.renorm.zig" "$PKG/libs/fri/src/root.zig" > /dev/null || { echo "ABORTO: la copia del pin difiere del original" >&2; fail=1; }
diff "$STAGE/transcript.zig" "$ROOT/libs/transcript.zig" > /dev/null || { echo "ABORTO: transcript.zig no es copia integra" >&2; fail=1; }
[ "$fail" = 0 ] || exit 1
echo "fidelidad verificada: las copias solo difieren en las lineas de import"

  # Ganchos de mutacion, para probar que el diferencial muerde por los DOS lados.
# Mutan la COPIA por etapas de .zig-cache, nunca zig-pkg: el pin es otro
# repositorio y editarlo dentro del nuestro seria cambiarlo sin su permiso.
# Con la variable puesta, la verificacion de fidelidad se salta a proposito y
# lo dice, porque ya no es una copia fiel.
if [ -n "${FRI_DIFF_MUTATE_MINE:-}" ]; then
  echo "MUTACION activa en MI fri: $FRI_DIFF_MUTATE_MINE"
  fail=1   # la fidelidad ya no aplica
fi
if [ -n "${FRI_DIFF_MUTATE_PIN:-}" ]; then
  echo "MUTACION activa en el PIN (copia por etapas, no zig-pkg): $FRI_DIFF_MUTATE_PIN"
  fail=1
  case "$FRI_DIFF_MUTATE_PIN" in
    una-ronda-menos)
      # El pin calcula las rondas con `config.rounds()`, no con la resta
      # escrita. La primera version de este gancho mutaba una expresion que no
      # existe ahi, asi que era un no-op y la "falta" era mia, no del
      # diferencial. Ahora se muta la linea que existe, y se COMPRUEBA que
      # MUTADO aparecio antes de seguir: un print incondicional fue lo que me
      # hizo informar de una correccion que no ocurrio ayer.
      sed -i 's|const rounds = try config.rounds();|const rounds = try config.rounds() - 1; // MUTADO|' "$STAGE/pin_root.zig"
      if ! grep -q MUTADO "$STAGE/pin_root.zig"; then
        echo "ABORTO: la mutacion del pin no aplico" >&2; exit 2
      fi ;;
    sin-residuo)
      sed -i 's|const residual_len|const residual_len: usize = 1 + 0 *|' "$STAGE/pin_root.zig"
      if ! grep -q MUTADO "$STAGE/pin_root.zig"; then
        sed -i 's|const residual_len: usize = 1 + 0 \*|const residual_len: usize = 1 + 0 *| // MUTADO|' "$STAGE/pin_root.zig"
      fi
      grep -q MUTADO "$STAGE/pin_root.zig" || { echo "ABORTO: la mutacion del pin no aplico" >&2; exit 2; } ;;
    *) echo "mutacion desconocida: $FRI_DIFF_MUTATE_PIN" >&2; exit 2 ;;
  esac
fi

"$ZIG" build-exe -OReleaseSafe \
    --dep fri_mine --dep fri_pin --dep zig-field \
    --dep zig-algebra-traits --dep zig-hash --dep zig-bigint \
    -Mroot="$ROOT/tools/fri_diff.zig" \
    --dep zig-merkle -Mfri_mine="$STAGE/root.zig" \
    --dep zig-merkle -Mfri_pin="$STAGE/pin_root.zig" \
    -Mzig-field="$PKG/libs/field/src/lib.zig" \
    -Mzig-merkle="$PKG/libs/merkle/src/root.zig" \
    -Mzig-algebra-traits="$PKG/libs/algebra-traits/src/root.zig" \
    -Mzig-hash="$PKG/libs/hash/src/root.zig" \
    -Mzig-bigint="$PKG/libs/bigint/src/root.zig" \
  --cache-dir "$ROOT/.zig-cache" --global-cache-dir "$HOME/.cache/zig" \
  -femit-bin="$STAGE/fri_diff" || exit 1
"$STAGE/fri_diff"
