#!/bin/sh
# Resolve the pinned zig-algebra into a local `zig-pkg/` directory, and say where.
#
# Why this exists: the three differentials all want the pin's SOURCE TREE, not
# just a compiled module, because they read and copy files out of it. Until now
# they assumed `zig-pkg/zig_algebra-<hash>/` was already sitting there. Nothing
# created it — not build.zig, not CI, not any document — so the tree existed on
# one machine because somebody had unpacked it by hand, and all three scripts
# failed with "falta el pin" on a clean checkout.
#
# A gate that cannot run in a clean checkout is not a gate. So the pin is
# unpacked here, from the global Zig package cache, which `zig build` populates
# as a matter of course.
#
# The HASH comes from build.zig.zon, never from a glob. An earlier version took
# `ls zig-pkg/zig_algebra-* | head -1`, which resolved to v0.3.0 — measuring a
# different dependency than the repository builds against, and calling it "the
# pin" while doing it.
#
# Usage:
#     . tools/pin_dir.sh          # defines the functions
#     PKG=$(pin_resolve)         # echoes the path, unpacking it if needed
#
# ROOT is taken from the caller when it is already set, because when this file
# is sourced `$0` is the CALLER's name, not this one's — deriving the directory
# from `$0` here reads a path that does not exist.
set -e

pin_hash() {
    [ -n "${ROOT:-}" ] || ROOT=$(cd "$(dirname "$0")/.." && pwd)
    sed -n 's/.*\.hash = "zig_algebra-\(.*\)".*/\1/p' "$ROOT/build.zig.zon"
}

pin_resolve() {
    [ -n "${ROOT:-}" ] || ROOT=$(cd "$(dirname "$0")/.." && pwd)
    # El mismo resolver que los otros cinco guiones. Va dentro de la funcion
    # porque pin_dir.sh se puede cargar antes de que exista ROOT.
    _here=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
    . "$_here/zig_bin.sh"
    ZIG=$(zig_bin)
    GLOBAL_CACHE=${ZIG_GLOBAL_CACHE:-$HOME/.cache/zig}

    H=$(pin_hash)
    if [ -z "$H" ]; then
        echo "ABORTO: build.zig.zon no declara un .hash de zig_algebra" >&2
        return 1
    fi
    DEST="$ROOT/zig-pkg/zig_algebra-$H"

    if [ -d "$DEST" ]; then
        echo "$DEST"
        return 0
    fi

    TAR="$GLOBAL_CACHE/p/zig_algebra-$H.tar.gz"
    if [ ! -f "$TAR" ]; then
        echo "ABORTO: no encuentro el pin en $GLOBAL_CACHE/p." >&2
        echo "  Se esperaba zig_algebra-$H.tar.gz. Corre 'zig build' una vez" >&2
        echo "  para que Zig lo traiga a la cache global, o exporta" >&2
        echo "  ZIG_GLOBAL_CACHE apuntando a la cache correcta." >&2
        return 1
    fi

    mkdir -p "$ROOT/zig-pkg"
    TMP=$(mktemp -d)
    # El tarball lleva un unico directorio de primer nivel con el nombre del
    # paquete. Se comprueba en vez de asumirlo: asumirlo es como se acabaria
    # moviendo el arbol al sitio equivocado, en silencio.
    tar xzf "$TAR" -C "$TMP"
    INNER=$(find "$TMP" -mindepth 1 -maxdepth 1 -type d | head -1)
    if [ -z "$INNER" ]; then
        rm -rf "$TMP"
        echo "ABORTO: el tarball del pin no trae directorio de primer nivel" >&2
        return 1
    fi
    n=$(find "$TMP" -mindepth 1 -maxdepth 1 -type d | wc -l)
    if [ "$n" -ne 1 ]; then
        rm -rf "$TMP"
        echo "ABORTO: el tarball trae $n directorios de primer nivel, se esperaba 1" >&2
        return 1
    fi
    mv "$INNER" "$DEST"
    rm -rf "$TMP"
    echo "$DEST"
}