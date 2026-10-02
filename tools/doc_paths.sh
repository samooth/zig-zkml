#!/bin/bash
# Gate: every path cited in every markdown file must resolve.
#
# Two paths died in the README and stayed there through the commit that killed
# them. Nothing walked the documents. `tools/assert_ledger.sh` walks the CODE
# and checks that every assert has a row in docs/asserts.md; this is the
# complement — it walks the DOCUMENTS and checks that every path they cite still
# exists.
#
# It handles both files and DIRECTORIES, because one of the dead citations was
# `libs/fri/`. A check that only stats files with -f walks straight past a
# directory, so the gate would go green while the README describes a tree that
# is not there.
#
# What it deliberately does NOT check:
#   - line numbers inside cited files. `m61.zig:12` cites a line in a live
#     external package; the line moves and nobody notices. No script can be
#     right about that, so it is a convention and not a gate.
#   - numbers in prose. Those need the instrument that produces them, and a
#     script that greps digits reports nothing useful.
#
# Usage: bash tools/doc_paths.sh    exit 0 = every cited path resolves
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 1

DOCS=$(git ls-files '*.md')

# Is this token a repo path at all?
#
# This is where the gate earns its keep or dies. A first version matched any
# known extension and reported 65 dead paths, and almost all 65 were the gate's
# own noise: a command with its arguments, `libs/field` as shorthand for
# `libs/field.zig`, a brace expansion, a `<engine>` placeholder, and headers
# that belong to llama.cpp and not to us. A gate that cries wolf is worse than
# no gate, because the next real finding arrives already discounted.
#
# So the rule is deliberately narrow: the token must start with a directory we
# own, or be one of our two top-level files. Everything else is not our path.
looks_like_path() {
    case "$1" in
        # not a literal path
        *" "*) return 1 ;;          # a command, with its arguments
        *"{"*|*"<"*|*"["*) return 1 ;;  # brace expansion, placeholder, list
        # must be ours
        libs/*|tools/*|docs/*|include/*|adapters/*|\.github/*) return 0 ;;
        build.zig|build.zig.zon|zkml.zig) return 0 ;;
        *) return 1 ;;
    esac
}

# A README may cite a path relative to its own directory or to the repo root,
# and readers try both. Directory-only or file-only are both accepted, because
# `libs/fri` and `libs/fri/root.zig` are the same subject to a reader.
resolves() {
    _doc=$1
    _p=$2
    _dir=$(dirname "$_doc")
    _p=${_p%%#*}
    # `m61.zig:12` cita una linea de un paquete externo vivo. La linea se mueve
    # y nadie lo nota, asi que un numero de linea no se comprueba: es una
    # convencion y no una puerta. aqui solo se resuelve el fichero.
    _p=$(printf '%s' "$_p" | sed 's/:[0-9][0-9]*$//')
    # Exenciones: rutas citadas que pertenecen a un checkout hermano. Se
    # comprueban aqui, no en el filtro, para que la cuenta de exenciones usadas
    # se pueda imprimir y subirla sea visible.
    if [ -f tools/doc_path_exempt.txt ]; then
        while read -r _ex _why; do
            case "$_ex" in ''|'#'*) continue ;; esac
            [ "$_ex" = "$_p" ] && { EXEMPT_USED=$((EXEMPT_USED + 1)); return 0; }
        done < tools/doc_path_exempt.txt
    fi
    case "$_p" in
        http*|mailto*) return 0 ;;
    esac
    [ -n "$_p" ] || return 0
    for cand in "$_p" "$_dir/$_p" "libs/$_p" "libs/stark/$_p" "libs/fri/$_p"; do
        [ -e "$cand" ] && return 0
    done
    # `libs/field` es la forma corta de `libs/field.zig`, y en un documento se
    # escribe asi a menudo. Se prueba con la extension antes de darlo por muerto,
    # porque un atajo legitimo no es una ruta muerta.
    case "$_p" in
        *.zig|*.py|*.sh) return 1 ;;
    esac
    for ext in .zig .py .sh; do
        for cand in "$_p$ext" "$_dir/$_p$ext" "libs/$_p$ext"; do
            [ -e "$cand" ] && return 0
        done
    done
    return 1
}

# Backtick spans and markdown link targets. Both are paths a reader will try.
candidates() {
    grep -oE '\]\([^)]+\)' "$1" | sed 's/^](//; s/)$//'
    grep -oE '`[^`]+`' "$1" | tr -d '`'
}

n_docs=0
n_cited=0
declare -a dead_lines=()
EXEMPT_USED=0

for doc in $DOCS; do
    # docs/archive/ guarda documentos HISTORICOS a proposito: describen un arbol
    # que ya no existe, y eso es su funcion. Requerir que sus rutas resuelvan
    # seria exigirles que mientan sobre el pasado. Se saltan y se dice.
    case "$doc" in
        docs/archive/*) continue ;;
    esac
    n_docs=$((n_docs + 1))
    while read -r raw; do
        [ -n "$raw" ] || continue
        looks_like_path "$raw" || continue
        clean=$(printf '%s' "$raw" | sed 's/[.,;:]*$//')
        n_cited=$((n_cited + 1))
        if ! resolves "$doc" "$clean"; then
            dead_lines+=("$doc -> $clean")
        fi
    done < <(candidates "$doc" | sort -u)
done

if [ "${#dead_lines[@]}" -gt 0 ]; then
    echo "doc-paths: ${#dead_lines[@]} rutas citadas en .md no resuelven:" >&2
    printf '  %s\n' "${dead_lines[@]}" >&2
    echo "" >&2
    echo "  Dos formas de que esto pase: el documento miente, o el arbol esta" >&2
    echo "  mal. Lo segundo no se arregla editando el documento." >&2
    exit 1
fi

echo "doc-paths: $n_docs documentos (docs/archive/ excluido a proposito), $EXEMPT_USED exenciones usadas, $n_cited rutas citadas, todas resuelven (ficheros y directorios)."
exit 0