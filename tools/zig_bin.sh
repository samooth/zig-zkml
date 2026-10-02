#!/bin/sh
# Resolve the Zig binary ONCE, for every tool that needs it.
#
# WHY THIS EXISTS. Six scripts — the three differentials, the cost bench, the
# wasm sweep and the pin resolver — each began with
#
#     ZIG=${ZIG:-/home/t0m4s/.zvm/0.16.0/zig}
#
# That path exists on exactly one machine: mine. CI installs Zig with
# mlugg/setup-zig, which puts it on PATH and nowhere else, so every one of those
# scripts died there instantly with "No existe el archivo o el directorio" — or,
# worse, with a diff that said the error set had changed, which is what a sweep
# prints when it never ran.
#
# A green gate that only runs on the machine of the person who wrote it is not a
# gate. It is a habit.
#
# The order below is deliberate and the order is the point:
#
#   1. $ZIG if the caller set it. CI and any scripted caller can pin a toolchain
#      explicitly, and that is the only override that counts.
#   2. The reference toolchain, if it exists HERE. AGENTS.md is explicit that
#      0.16.0 stable is the reference and the 0.16.0-dev build is not, and this
#      machine has both. Falling straight to PATH would silently let a dev build
#      stand in for the reference, which is the thing AGENTS warns about twice.
#   3. `zig` from PATH. That is what setup-zig provides, and it is why this
#      file exists.
#
# Usage:
#     . tools/zig_bin.sh
#     ZIG=$(zig_bin)
set -e

ZIG_REFERENCE=/home/t0m4s/.zvm/0.16.0/zig

zig_bin() {
    if [ -n "${ZIG:-}" ] && [ -x "${ZIG}" ]; then
        echo "$ZIG"
        return 0
    fi
    if [ -x "$ZIG_REFERENCE" ]; then
        echo "$ZIG_REFERENCE"
        return 0
    fi
    # Last resort: whatever is on PATH. If that is missing too, say so here
    # rather than three lines later with a bare "command not found" that reads
    # like the gate is broken.
    if command -v zig >/dev/null 2>&1; then
        command -v zig
        return 0
    fi
    echo "ABORTO: no encuentro zig." >&2
    echo "  Ni en ZIG=$ZIG, ni en $ZIG_REFERENCE, ni en el PATH." >&2
    echo "  Exporta ZIG con la ruta, o ponlo en el PATH." >&2
    return 1
}

# Allow this file to be executed directly as a probe: prints the resolved
# binary and its version, so "which zig does CI get" is one command.
if [ "${0##*/}" = "zig_bin.sh" ]; then
    ZIG=$(zig_bin) || exit 1
    echo "$ZIG"
    "$ZIG" version
fi