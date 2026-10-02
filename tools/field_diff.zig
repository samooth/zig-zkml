//! Differential: `libs/field.zig` against the pinned `zig-algebra` M61.
//!
//! Signature AND result, function by function, on the same inputs. A
//! signature comparison is not a differential: it says the names line up and
//! nothing about whether the two agree on a value. This runs both and prints
//! every disagreement with the inputs that produced it, so a claim about
//! equivalence either holds or names the case where it does not.
//!
//! Run it with `tools/field_diff.sh`, which is kept so the measurement can be
//! repeated. A number measured with a tool that does not exist tomorrow is not
//! a number anybody can check.

const std = @import("std");
const mine = @import("field_mine");
const zf = @import("zig-field");

const Theirs = zf.M61;
var failures: usize = 0;
var checks: usize = 0;

// El desglose lo cuenta la herramienta, no una persona. Ya se escribio
// "1264 - 1024 = 256" en un mensaje cuando el incremento era 240, porque
// contar a mano no detecta el error que uno mismo acaba de cometer.
var by_check: std.StringHashMapUnmanaged(usize) = .{};

fn tally(name: []const u8) void {
    const gop = by_check.getOrPut(std.heap.page_allocator, name) catch @panic("oom");
    if (!gop.found_existing) gop.value_ptr.* = 0;
    gop.value_ptr.* += 1;
}

fn note(comptime what: []const u8, ok: bool) void {
    checks += 1;
    tally(what);
    if (!ok) {
        failures += 1;
        std.debug.print("  DISCREPANCIA {s}\n", .{what});
    }
}

fn mineToU64(a: mine.Goldilocks) u64 {
    return a.toU64();
}

fn theirsToU64(a: Theirs) u64 {
    return @intCast(a.toU64());
}

/// Same input through both constructors, compared on the field value.
fn cmpFrom(av: u64, bv: u64) void {
    const a = mine.Goldilocks.fromU64(av);
    const b = Theirs.fromInt(bv);
    const ok = mineToU64(a) == theirsToU64(b);
    note("fromU64/fromInt", ok);
    if (!ok) {
        std.debug.print("    entrada {d}: mio {d} · predef {d}\n", .{ av, mineToU64(a), theirsToU64(b) });
    }
    // Also compare the encoded bytes: a matching value with a differing
    // encoding is still a difference for anything that hashes them.
    var ab: [8]u8 = undefined;
    a.toBytes(&ab);
    const bb: [8]u8 = b.toBytes(); // firma distinta: el predef devuelve, el mio escribe en un out-param
    note("toBytes", std.mem.eql(u8, &ab, &bb));
    if (!std.mem.eql(u8, &ab, &bb)) {
        std.debug.print("    entrada {d}: mio {x} · predef {x}\n", .{ av, ab, bb });
    }
}

fn cmpBinary(a_in: u64, b_in: u64, comptime op: []const u8) void {
    const A = mine.Goldilocks.fromU64(a_in);
    const B = mine.Goldilocks.fromU64(b_in);
    const TA = Theirs.fromInt(a_in);
    const TB = Theirs.fromInt(b_in);

    const mine_result: u64 = if (comptime std.mem.eql(u8, op, "add"))
        mineToU64(mine.Goldilocks.add(A, B))
    else if (comptime std.mem.eql(u8, op, "sub"))
        mineToU64(mine.Goldilocks.sub(A, B))
    else if (comptime std.mem.eql(u8, op, "mul"))
        mineToU64(mine.Goldilocks.mul(A, B))
    else
        mineToU64(mine.Goldilocks.neg(A));

    const their_result: u64 = if (comptime std.mem.eql(u8, op, "add"))
        theirsToU64(TA.add(TB))
    else if (comptime std.mem.eql(u8, op, "sub"))
        theirsToU64(TA.sub(TB))
    else if (comptime std.mem.eql(u8, op, "mul"))
        theirsToU64(TA.mul(TB))
    else
        theirsToU64(TA.neg());

    const ok = mine_result == their_result;
    note(op, ok);
    if (!ok) {
        std.debug.print("    {d} {s} {d}: mio {d} · predef {d}\n", .{ a_in, op, b_in, mine_result, their_result });
    }
}

fn cmpInv(a_in: u64) void {
    const A = mine.Goldilocks.fromU64(a_in);
    const TA = Theirs.fromInt(a_in);
    // On the REDUCED value, not the raw input: p and 2p are in the corpus and
    // both construct to zero, where inv legitimately fails on either side.
    // Testing the raw input was a bug in this harness, not a difference.
    const reduced = mineToU64(A);
    note("fromU64 reduce igual que fromInt", reduced == theirsToU64(TA));
    if (reduced == 0) {
        // Both must refuse; the names of the failure are reported apart from
        // the values because a wrong error name is a real difference.
        const mine_err = mine.Goldilocks.inv(A);
        const their_err = TA.invChecked();
        note("inv(0) ambos fallan", (mine_err catch null) == null and (their_err catch null) == null);
        std.debug.print("    (inv de 0: ambos fallan — mio ZeroInverse, predef invChecked)\n", .{});
        return;
    }
    const m = mineToU64(mine.Goldilocks.inv(A) catch unreachable);
    const t = theirsToU64(TA.invChecked() catch unreachable);
    note("inv", m == t);
    if (m != t) std.debug.print("    inv({d}): mio {d} · predef {d}\n", .{ a_in, m, t });
}

fn cmpPow(a_in: u64, e: u64) void {
    const A = mine.Goldilocks.fromU64(a_in);
    const TA = Theirs.fromInt(a_in);
    const m = mineToU64(mine.Goldilocks.pow(A, e));
    const t = theirsToU64(TA.pow(e));
    note("pow", m == t);
    if (m != t) std.debug.print("    pow({d},{d}): mio {d} · predef {d}\n", .{ a_in, e, m, t });
}

fn cmpPredicates(a_in: u64) void {
    const A = mine.Goldilocks.fromU64(a_in);
    const TA = Theirs.fromInt(a_in);
    note("isZero", A.isZero() == TA.isZero());
    note("toU64", mineToU64(A) == theirsToU64(TA));
}

/// `eql` is a predicate over a PAIR, so comparing `a.eql(a)` asks both sides a
/// question they answer reflexively and cannot fail. This puts two different
/// inputs in front of both implementations and compares the answers instead.
/// The corpus holds pairs that differ as integers and agree as field elements
/// (0 and p, 1 and p+1, 2p), which is where a differing reduction would show.
fn cmpEql(a_in: u64, b_in: u64) void {
    const m = mine.Goldilocks.fromU64(a_in).eql(mine.Goldilocks.fromU64(b_in));
    const t = Theirs.fromInt(a_in).eql(Theirs.fromInt(b_in));
    note("eql", m == t);
    if (m != t) {
        std.debug.print("    eql({d},{d}): mio {} · predef {}\n", .{ a_in, b_in, m, t });
    }
}

pub fn main() !void {
    const p: u64 = (1 << 61) - 1;
    // Boundaries first: zero, one, p-1, and values that exceed p so the two
    // reduction paths are compared and not just the happy one.
    const vals_seed = [_]u64{
        0,                        1,                     2,                     3,
        p - 1,                    p - 2,                 (p - 1) / 2,           p,
        p + 1,                    2 * p,                 2 * p + 1,             std.math.maxInt(u64),
        std.math.maxInt(u64) - 1, 0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 1234567890123456789,
    };

    // 16 valores para un primo de 61 bits son POCOS: una reduccion mal
    // escrita puede coincidir en esos 16 puntos y divergir en el resto. Se
    // anaden 48 mas de una secuencia determinista —LCG con semilla fija,
    // porque un numero medido con una fuente que no exista manana no es
    // verificable— y algunos limites de cada bloque de la reduccion.
    var vals: [64]u64 = undefined;
    for (vals_seed, 0..) |sv, i| vals[i] = sv;
    var seed: u64 = 0x9E3779B97F4A7C15;
    for (16..vals.len) |i| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        vals[i] = seed;
    }
    vals[16] = p / 7;
    vals[17] = p / 3;
    vals[18] = p - p / 5;
    vals[19] = (p - 1) / 4;
    vals[20] = std.math.maxInt(u64) / 3;
    vals[21] = 1 << 60;
    vals[22] = (1 << 60) + 1;
    vals[23] = 3 * p / 4;
    vals[24] = @intCast((@as(u128, p) * 15) / 16);
    vals[25] = (p + 1) / 2;
    vals[26] = p / 2 - 1;
    vals[27] = 2;
    vals[28] = 3;
    vals[29] = p - 3;
    vals[30] = p / 2 + 1;
    vals[31] = std.math.maxInt(u64) / 2;

    std.debug.print("p = {d}\n", .{p});
    std.debug.print("representacion: mio canonica en `rep` · predef SmallField canonica en `value`\n\n", .{});

    for (vals) |v| {
        cmpFrom(v, v);
    }
    for (vals) |a| {
        for (vals) |b| {
            cmpBinary(a, b, "add");
            cmpBinary(a, b, "sub");
            cmpBinary(a, b, "mul");
            cmpEql(a, b);
        }
        cmpBinary(a, 0, "neg");
        cmpInv(a);
        cmpPredicates(a);
        for ([_]u64{ 0, 1, 2, 61, 62, 255, 1000, 1 << 20 }) |e| {
            cmpPow(a, e);
        }
    }

    std.debug.print("\ndesglose (contado por la herramienta, no a mano):\n", .{});
    const Row = struct { name: []const u8, count: usize };
    var rows: [64]Row = undefined;
    var it = by_check.iterator();
    var n: usize = 0;
    while (it.next()) |e| {
        rows[n] = .{ .name = e.key_ptr.*, .count = e.value_ptr.* };
        n += 1;
    }
    std.mem.sort(Row, rows[0..n], {}, struct {
        fn gt(_: void, a: Row, b: Row) bool {
            return a.count > b.count;
        }
    }.gt);
    for (rows[0..n]) |r| std.debug.print("  {d:>4}  {s}\n", .{ r.count, r.name });
    var sum: usize = 0;
    for (rows[0..n]) |r| sum += r.count;
    std.debug.print("  {s:>26}  {d}\n", .{ "TOTAL (suma del desglose)", sum });
    std.debug.print("\ncomprobaciones: {d}\n", .{checks});
    std.debug.print("discrepancias: {d}\n", .{failures});
    if (failures == 0) {
        std.debug.print("RESULTADO: identicos en todo lo comparado\n", .{});
    } else {
        std.debug.print("RESULTADO: HAY DIFERENCIAS — no se borra nada, se reporta\n", .{});
    }
    // El paso de build se queda con el codigo de salida del proceso, asi que
    // imprimir "HAY DIFERENCIAS" y salir con 0 es una puerta en verde sobre un
    // fallo. Devolver un error es lo que el proceso traduce en exit code.
    if (failures > 0) return error.DifferentialFound;
}
