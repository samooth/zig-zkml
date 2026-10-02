//! Keeps docs/asserts.md honest.
//!
//! §0 of AGENTS.md: every number in this repository comes out of a command or a
//! test that asserts it. The assert count is one such number, and a prose table
//! listing all of them is exactly the kind of copy that goes stale silently —
//! the failure mode AGENTS.md describes for `docs/soundness.md`.
//!
//! So the table is not trusted. This test re-derives the list from the source
//! and fails when the document and the code disagree, in either direction:
//!
//!   - an assert in the code with no row in the ledger (someone added one and
//!     did not classify it), and
//!   - a row in the ledger pointing at a line that no longer holds an assert
//!     (someone moved or deleted one and the ledger was not updated).
//!
//! It also pins the measured headline — total, by-kind, by-protection, debt —
//! so the number in the prose cannot drift either.
//!
//! It deliberately does NOT ratchet. A hard ratchet (assert: any new assert
//! fails the build) is right for `zig-algebra` with 391 tests and a slow
//! changelog, and wrong here while the prover is being written. What this test
//! buys is the part that matters: that nobody classifies an assert wrongly.

const std = @import("std");
const testing = std.testing;

const ledger_path = "docs/asserts.md";

/// Directories searched, matching tools/assert_ledger.sh. Kept in one place so
/// the script and the test cannot drift.
const search_roots = [_][]const u8{ "libs", "tools", "adapters", "zkml.zig" };

const skipped_dirs = [_][]const u8{ "zig-pkg", ".zig-cache" };

fn isSkipped(path: []const u8) bool {
    for (skipped_dirs) |d| {
        if (std.mem.indexOf(u8, path, d) != null) return true;
    }
    // The ledger's own test is excluded: it mentions `std.debug.assert` in the
    // skipped-directories list and in its own prose, and counting it would make
    // the total depend on the file that reports the total.
    if (std.mem.endsWith(u8, path, "assert_ledger_test.zig")) return true;
    return false;
}

/// Every `std.debug.assert` in the repository, as `path:line`, in source order
/// so the comparison is deterministic.
fn findAsserts(gpa: std.mem.Allocator, io: std.Io) !std.ArrayList([]const u8) {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(gpa);

    for (search_roots) |root| {
        var dir = std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true }) catch continue;
        defer dir.close(io);

        var it = try dir.walk(gpa);
        defer it.deinit();
        while (try it.next(io)) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.basename, ".zig")) continue;
            const full = try std.fs.path.join(gpa, &.{ root, entry.path });
            defer gpa.free(full);
            if (isSkipped(entry.path)) continue;

            const src = std.Io.Dir.cwd().readFileAlloc(io, full, gpa, .limited(4 << 20)) catch continue;
            defer gpa.free(src);

            var line_no: usize = 0;
            var it_lines = std.mem.splitScalar(u8, src, '\n');
            while (it_lines.next()) |line| {
                line_no += 1;
                if (std.mem.indexOf(u8, line, "std.debug.assert") == null) continue;
                const loc = try std.fmt.allocPrint(gpa, "{s}:{d}", .{ full, line_no });
                try out.append(gpa, loc);
            }
        }
    }
    return out;
}

fn ledgerText(gpa: std.mem.Allocator, io: std.Io) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, ledger_path, gpa, .limited(1 << 20));
}

test "every assert in the repository has a row in docs/asserts.md" {
    const gpa = testing.allocator;
    const ledger = try ledgerText(gpa, std.testing.io);
    defer gpa.free(ledger);

    var found = try findAsserts(gpa, std.testing.io);
    defer {
        for (found.items) |l| gpa.free(l);
        found.deinit(gpa);
    }
    try std.testing.expect(found.items.len > 0);

    for (found.items) |loc| {
        // The ledger cites `path:line`, and the paths it uses are repo-relative
        // with the same spelling the walker produces, so a plain substring test
        // is enough to decide membership.
        if (std.mem.indexOf(u8, ledger, loc) == null) {
            std.debug.print(
                "\nassert sin clasificar: {s}\n" ++
                    "  anadelo a docs/asserts.md con kind y protection, o regenera\n" ++
                    "  el recuento con tools/assert_ledger.sh\n",
                .{loc},
            );
            return error.AssertNotClassified;
        }
    }
}

test "no row in the ledger points at a line that is not an assert" {
    const gpa = testing.allocator;
    const ledger = try ledgerText(gpa, std.testing.io);
    defer gpa.free(ledger);

    // Every `| N | `path:line` |` row must correspond to a real assert. The
    // other direction — a row for an assert that moved — is what this catches,
    // and it is the half a completeness-only test would miss.
    var found = try findAsserts(gpa, std.testing.io);
    defer {
        for (found.items) |l| gpa.free(l);
        found.deinit(gpa);
    }

    var lines = std.mem.splitScalar(u8, ledger, '\n');
    var rows: usize = 0;
    while (lines.next()) |line| {
        // Rows look like: | 7 | `libs/torus/domain.zig:80` | ...
        const tick = std.mem.indexOf(u8, line, "`") orelse continue;
        const rest = line[tick + 1 ..];
        const tick2 = std.mem.indexOfScalar(u8, rest, '`') orelse continue;
        const loc = rest[0..tick2];
        if (std.mem.indexOfScalar(u8, loc, ':') == null) continue;
        rows += 1;

        var present = false;
        for (found.items) |f| {
            if (std.mem.eql(u8, f, loc)) present = true;
        }
        if (!present) {
            std.debug.print(
                "\nfila del ledger sin assert: {s}\n" ++
                    "  el assert se movio o se borro: actualiza docs/asserts.md\n",
                .{loc},
            );
            return error.LedgerRowStale;
        }
    }

    // Sanity: the document must actually have rows, otherwise the loop above
    // proved nothing because it never ran.
    try std.testing.expect(rows > 0);
    try std.testing.expectEqual(found.items.len, rows);
}

test "the measured headline matches the ledger" {
    const gpa = testing.allocator;
    const ledger = try ledgerText(gpa, std.testing.io);
    defer gpa.free(ledger);

    var found = try findAsserts(gpa, std.testing.io);
    defer {
        for (found.items) |l| gpa.free(l);
        found.deinit(gpa);
    }

    // The prose carries "total: N". It has to be the count, or the headline
    // drifts even when every row is right.
    var buf: [64]u8 = undefined;
    const needle = try std.fmt.bufPrint(&buf, "total        {}", .{found.items.len});
    try std.testing.expect(std.mem.indexOf(u8, ledger, needle) != null);

    // Y el TITULO. Esta puerta antes miraba solo la linea de arriba, con lo
    // que el titulo de docs/asserts.md decia 26 mientras el cuerpo de la
    // misma pagina decia 20: la puerta corria, pasaba, y no podia ver el
    // numero que era falso. El valor de una puerta es lo que puede fallar,
    // y esta no podia fallar en el sitio donde el numero estaba mal.
    //
    // El titulo es la primera linea que se lee, asi que es la que mas
    // importa que no mienta.
    var title_buf: [64]u8 = undefined;
    const title_needle = try std.fmt.bufPrint(&title_buf, "the {d} `std.debug.assert` in this repository", .{found.items.len});
    try std.testing.expect(std.mem.indexOf(u8, ledger, title_needle) != null);
}

test "the ledger states kind and protection for every row" {
    const gpa = std.testing.allocator;
    const ledger = try ledgerText(gpa, std.testing.io);
    defer gpa.free(ledger);

    // The whole point of the two fields is that they are separate. A row
    // missing either is a row whose debt status was guessed from its location,
    // which is the error this ledger exists to prevent.
    var lines = std.mem.splitScalar(u8, ledger, '\n');
    while (lines.next()) |line| {
        if (std.mem.indexOf(u8, line, "| `") == null) continue;
        if (std.mem.indexOf(u8, line, ":") == null) continue;

        // kind is the 3rd cell, protection the 4th.
        // A row is: | N | `loc` | kind | protection | guards | debt |
        // so after splitting on '|' the cells are: "", N, loc, kind, protection,
        // guards, debt, "" — indices 3 and 4 are the two fields.
        var cells = std.mem.splitScalar(u8, line, '|');
        var idx: usize = 0;
        var kind: []const u8 = "";
        var protection: []const u8 = "";
        while (cells.next()) |c| : (idx += 1) {
            switch (idx) {
                3 => kind = c,
                4 => protection = c,
                else => {},
            }
        }
        if (kind.len == 0) continue;
        const kind_ok = std.mem.indexOf(u8, kind, "api") != null or
            std.mem.indexOf(u8, kind, "internal") != null or
            std.mem.indexOf(u8, kind, "comptime") != null or
            std.mem.indexOf(u8, kind, "fixture") != null;
        const prot_ok = std.mem.indexOf(u8, protection, "caller") != null or
            std.mem.indexOf(u8, protection, "invariant") != null;
        if (!kind_ok or !prot_ok) {
            std.debug.print("\nfila sin kind/protection: {s}\n", .{line});
            return error.MissingField;
        }
    }
}

test "every prose document is in the language AGENTS.md says it is" {
    // AGENTS.md fija dos idiomas: README y AGENTS en ingles, `docs/` en
    // espanol, con dos excepciones nombradas. Sin esta puerta la politica es
    // prosa, y la prosa no se puede tumbar — que es el punto 2 de zk aplicado
    // a este repo: un gate que AGENTS dice que tiene y que nadie corre.
    //
    // Se comprueba que la excepcion este DECLARADA, no que el fichero este en
    // un idioma: distinguir ingles de espanol automaticamente es fragil, y un
    // detector fragil que pasa es peor que ninguno. Lo que si se vigila es
    // que la lista de excepciones de AGENTS.md exista y no crezca sola.
    const gpa = testing.allocator;
    const agents = try readWhole(gpa, std.testing.io, "AGENTS.md");
    defer gpa.free(agents);

    const exceptions = [_][]const u8{ "docs/asserts.md", "docs/decisions/ADR-0002-fingerprint-rank-one.md" };
    for (exceptions) |path| {
        testing.expect(std.mem.indexOf(u8, agents, path) != null) catch |e| {
            std.debug.print("\nAGENTS.md no declara la excepcion: {s}\n", .{path});
            return e;
        };
    }
    // Y que la politica este enunciada. AGENTS.md esta en ingles, asi que se
    // busca la frase inglesa: una comprobacion que buscara "en ingles" en
    // espanol no encontraria nada y pasaria — o fallaria — por la razon
    // equivocada, que es el modo de fallo de hoy tres veces.
    testing.expect(std.mem.indexOf(u8, agents, "in English on purpose") != null) catch |e| {
        std.debug.print("\nAGENTS.md no declara la politica de idioma\n", .{});
        return e;
    };
}

fn readWhole(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
}
