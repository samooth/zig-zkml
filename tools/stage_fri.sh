#!/bin/sh
# Stage `libs/fri` as ONE module, in a flat directory, with the pin beside it.
#
# WHY THIS EXISTS. `libs/fri` cannot be built as a module rooted at
# `libs/fri/root.zig`: two files in it import `../field.zig` and
# `../transcript.zig` from inside their own tests, and those escape the module
# root. Zig rejects that with "import of file outside module path".
#
# The fix is to flatten: copy everything into one directory and rewrite the
# relative imports. That was done inside `tools/fri_diff.sh`, which is where it
# is documented — and then it had to be done AGAIN, in the next script that
# needed the same thing, because a working solution that lives inside the tool
# that happened to need it is not a solution. Four copies of the same staging is
# four places to forget the fidelity check.
#
# So the staging lives here, once, with the fidelity check attached to it. Every
# script that needs the FRI sources calls `stage_fri` and gets the same
# guarantee: the copy differs from the original only in the rewritten imports,
# and a mutation hook is the ONLY thing allowed to make it differ more.
#
# Usage:
#     STAGE=$ROOT/.zig-cache/<name>
#     . tools/stage_fri.sh
#     stage_fri "$STAGE"        # populate + verify
#
# Sets: STAGE (already set by the caller), STAGED_ROOT="$STAGE/root.zig".

# Rewrite a staged file back to repository form and diff it against the
# original. Anything other than the rewritten imports is a fidelity failure.
_fri_check_fidelity() {
    _f=$1
    _orig=$2
    # La copia tiene "./field.zig" y "./merkle.zig"; el original tiene "../".
    # Se invierte SOLO en esa direccion: poner las dos reglas hacia adelante y
    # atras en la misma expresion las encadena y produce una sustitucion sin
    # terminar.
    sed -e 's|@import("\./field\.zig")|@import("../field.zig")|' \
        -e 's|@import("\./merkle\.zig")|@import("../merkle.zig")|' \
        -e 's|@import("\./transcript\.zig")|@import("../transcript.zig")|' \
        -e 's|@import("\./\([a-z0-9]*\)\.zig")|@import("../torus/\1.zig")|' \
        "$_f" > "$_f.renorm.zig"
    diff "$_f.renorm.zig" "$_orig" > /dev/null || {
        echo "ABORTO: $(basename "$_orig") difiere del original mas alla de los imports" >&2
        return 1
    }
}

# stage_fri <stage_dir> [mutated]
#   mutated=1 skips the fidelity check, because a mutation hook is deliberately
#   making the copy differ. The caller must SAY so, so "skip the check" is a
#   stated decision and not an accident.
stage_fri() {
    _stage=$1
    _mutated=${2:-0}

    mkdir -p "$_stage"
    cp "$ROOT/libs/field.zig" "$_stage/field.zig"
    cp "$ROOT/libs/merkle.zig" "$_stage/merkle.zig"
    cp "$ROOT/libs/transcript.zig" "$_stage/transcript.zig"
    # root.zig sigue en libs/fri/ —es el FRI, lo unico que queda ahi— y los
    # otros tres se fueron a libs/torus/. El bucle tiene que decir de donde sale
    # cada uno; dar por supuesto que comparten directorio fue lo que rompio esto
    # en la primera pasada, y lo cazaron las tres puertas que leen rutas.
    for f in root domain fft; do
        case "$f" in
            root) src="$ROOT/libs/fri/root.zig" ;;
            *)     src="$ROOT/libs/torus/$f.zig" ;;
        esac
        sed -e 's|@import("../field\.zig")|@import("./field.zig")|' \
            -e 's|@import("../merkle\.zig")|@import("./merkle.zig")|' \
            -e 's|@import("../transcript\.zig")|@import("./transcript.zig")|' \
            -e 's|@import("../torus/\([a-z0-9]*\)\.zig")|@import("./\1.zig")|' \
            "$src" > "$_stage/$f.zig"
    done
    sed -e 's|@import("../field\.zig")|@import("./field.zig")|' \
        -e 's|@import("../merkle\.zig")|@import("./merkle.zig")|' \
        -e 's|@import("../transcript\.zig")|@import("./transcript.zig")|' \
        "$ROOT/libs/torus/fp2.zig" > "$_stage/fp2.zig"

    # El lado del pin, al lado del nuestro, con la unica diferencia de la ruta
    # del import. Sin envoltorio: un fichero no puede pertenecer a dos modulos,
    # asi que mi lado cuelga de UN modulo que ya reexporta Fp2 y Domain.
    sed 's|@import("torus.zig")|@import("./torus.zig")|' \
        "$PKG/libs/fri/src/root.zig" > "$_stage/pin_root.zig"
    cp "$PKG/libs/fri/src/torus.zig" "$_stage/torus.zig"

    if [ "$_mutated" = 1 ]; then
        echo "fidelidad OMITIDA a proposito: hay una mutacion activa" >&2
        return 0
    fi

    for f in root domain fft fp2; do
        case "$f" in
            root) orig="$ROOT/libs/fri/root.zig" ;;
            *)     orig="$ROOT/libs/torus/$f.zig" ;;
        esac
        _fri_check_fidelity "$_stage/$f.zig" "$orig" || return 1
    done
    for f in field merkle transcript; do
        diff "$_stage/$f.zig" "$ROOT/libs/$f.zig" > /dev/null || {
            echo "ABORTO: $f.zig no es copia integra del original" >&2
            return 1
        }
    done
    sed 's|@import("./torus.zig")|@import("torus.zig")|' \
        "$_stage/pin_root.zig" > "$_stage/pin_root.renorm.zig"
    diff "$_stage/pin_root.renorm.zig" "$PKG/libs/fri/src/root.zig" > /dev/null || {
        echo "ABORTO: la copia del pin difiere del original" >&2
        return 1
    }
    echo "fidelidad verificada: las copias solo difieren en las lineas de import"
}