#!/bin/sh
# Differential of my FRI composition against the pinned one.
# Pin read from build.zig.zon, never globbed.
# The staged copies are diffed against the originals and the run aborts unless
# the only difference is the rewritten import: a copy that drifts silently
# measures the wrong thing.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)

# El binario de Zig se resuelve UNA vez, en tools/zig_bin.sh. Estaba escrito
# a mano en seis guiones con la ruta de una sola maquina, y en CI —donde
# setup-zig lo pone en el PATH— todos ellos morian al instante.
. "$ROOT/tools/zig_bin.sh"
ZIG=$(zig_bin)
HASH=$(sed -n 's/.*\.hash = "zig_algebra-\(.*\)".*/\1/p' "$ROOT/build.zig.zon")
PKG="$ROOT/zig-pkg/zig_algebra-$HASH"
[ -d "$PKG" ] || { echo "falta el pin $PKG" >&2; exit 1; }
echo "pin leido de build.zig.zon: $(basename "$PKG")"

  STAGE="$ROOT/.zig-cache/fri_diff"
  # El staging vive en tools/stage_fri.sh, no aqui. Este guion tenia su propia
  # copia —veinte lineas de sed con su comprobacion de fidelidad— y cuando los
  # ficheros se movieron la copia se quedo por detras del helper, que es
  # exactamente el fallo que justificaba extraerlo: una solucion que vive
  # dentro de la herramienta que la necesita no es una solucion.
  . "$ROOT/tools/stage_fri.sh"
  stage_fri "$STAGE"


  # Ganchos de mutacion, para probar que el diferencial muerde por los DOS lados.
# Mutan la COPIA por etapas de .zig-cache, nunca zig-pkg: el pin es otro
# repositorio y editarlo dentro del nuestro seria cambiarlo sin su permiso.
# Con la variable puesta, la verificacion de fidelidad se salta a proposito y
# lo dice, porque ya no es una copia fiel.
  # El gancho de MI lado estaba VACIO: ponia fail=1, imprimia "MUTACION activa
  # en MI fri" y no mutaba nada. Un print que anuncia una sustitucion que no
  # ocurrio es el fallo que este repositorio ya ha pagado tres veces, y aqui es
  # el mas caro: el 0 que devuelve se lee como una medicion. Ahora muta de
  # verdad, y ABORTA si el patron no aparece.
  if [ -n "${FRI_DIFF_MUTATE_MINE:-}" ]; then
    echo "MUTACION activa en MI fri: $FRI_DIFF_MUTATE_MINE"
    fail=1   # la fidelidad ya no aplica
    case "$FRI_DIFF_MUTATE_MINE" in
      una-ronda-menos)
        # Las rondas las calcula `Config.validate`, no una resta en el sitio
        # de la llamada, asi que se muta la linea que existe de verdad.
        perl -0pi -e 's/const rounds = self[.]log_domain - self[.]log_final;/const rounds = self.log_domain - self.log_final - 1; \/\/ MUTADO/' "$STAGE/root.zig"
        grep -q MUTADO "$STAGE/root.zig" || { echo "ABORTO: la mutacion de MI fri no aplico" >&2; exit 2; } ;;
      *) echo "mutacion desconocida para MI fri: $FRI_DIFF_MUTATE_MINE" >&2; exit 2 ;;
    esac
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
