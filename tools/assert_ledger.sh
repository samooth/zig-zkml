#!/bin/sh
# Count and locate every std.debug.assert in the repository, excluding vendored
# and build-output directories.
#
# The count in docs/asserts.md comes from here, and a test in
# libs/stark/assert_ledger_test.zig fails when that file and this disagree.
# §0 of AGENTS.md: every number enters the repository from a command or a test
# that asserts it, never recomputed by hand.
#
# usage: tools/assert_ledger.sh          # report
#        tools/assert_ledger.sh --count  # just the total, for the test

set -eu

cd "$(dirname "$0")/.."

# Same search the test uses, so the two cannot drift apart.
PATHS="libs tools adapters zkml.zig"

# The ledger's own test is excluded, and the test does the same: it mentions
# `std.debug.assert` in prose, and counting it would make the total depend on
# the file that reports the total. Both sides must exclude identically or they
# disagree by construction.
EXCLUDE="assert_ledger_test\.zig"

count() {
    # shellcheck disable=SC2086
    grep -rn "std\.debug\.assert" --include='*.zig' $PATHS 2>/dev/null \
        | grep -v '/zig-pkg/' \
        | grep -v '/\.zig-cache/' \
        | grep -v "$EXCLUDE" \
        | wc -l | tr -d ' '
}

if [ "${1:-}" = "--count" ]; then
    count
    exit 0
fi

echo "std.debug.assert in the repository"
echo
count | sed 's/^/total: /'
echo
echo "by file:"
# shellcheck disable=SC2086
grep -rn "std\.debug\.assert" --include='*.zig' $PATHS 2>/dev/null \
    | grep -v '/zig-pkg/' \
    | grep -v '/\.zig-cache/' \
    | grep -v "assert_ledger_test\.zig" \
    | cut -d: -f1 \
    | sort \
    | uniq -c \
    | sort -rn
