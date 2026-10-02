//! Differential: the `Fp2` -> `Torus61` conversion the FRI differential needs.
//!
//! It is tested BEFORE the clean FRI differential runs, and mutated before it
//! is trusted. The order matters: a clean run that prints 0 first, with the
//! mutation after, is a formality — the number is already in hand and the
//! mutation looks like a rubber stamp.
//!
//! What is under test is NOT that the conversion is a function, but that it is
//! the IDENTITY: that `Fp2{a,b}` and `Torus61{c0 = a, c1 = b}` are the same
//! element. That matters because if it is, the FRI differential that follows
//! will run the SAME field arithmetic on both sides — so the FRI differential
//! measures the composition, not the field. The field was already measured
//! separately, and saying otherwise would sign for something that measured
//! something else.
//!
//! Every line compares my arithmetic against the PIN's. Nothing here compares
//! my implementation against itself. That rule is the lesson of the `eql`
//! check that compared `A.eql(A)` with `TA.eql(TA)`: reflexive comparisons
//! cannot fail and only inflate the total.
//!
//! Run with `tools/fri_conv_diff.sh`.

const std = @import("std");

const ConversionFailed = error{ConversionFailed};
const fp2_mod = @import("fp2_mine");
const zf = @import("zig-field");

const Fp2 = fp2_mod.Fp2;
const G = fp2_mod.Goldilocks;
const Torus = zf.M61;
const Q = zf.QuadraticExtension(Torus, Torus.fromInt(-1));

var failures: usize = 0;
var checks: usize = 0;
var by_check: std.StringHashMapUnmanaged(usize) = .{};

/// Tambien los fallos por operacion. Reconstruir a mano cuantos pasan con la
/// conversion rota salio mal: dio 91 donde el real eran 68. Este numero lo
/// mide la herramienta o no lo hay.
var fails_by_check: std.StringHashMapUnmanaged(usize) = .{};

fn tally(name: []const u8) void {
    const gop = by_check.getOrPut(std.heap.page_allocator, name) catch @panic("oom");
    if (!gop.found_existing) gop.value_ptr.* = 0;
    gop.value_ptr.* += 1;
}

fn tally_fail(name: []const u8) void {
    const gop = fails_by_check.getOrPut(std.heap.page_allocator, name) catch @panic("oom");
    if (!gop.found_existing) gop.value_ptr.* = 0;
    gop.value_ptr.* += 1;
}

fn note(comptime what: []const u8, ok: bool) void {
    checks += 1;
    tally(what);
    if (!ok) {
        failures += 1;
        tally_fail(what);
        if (@import("builtin").mode == .Debug)
            std.debug.print("  DISCREPANCIA {s}\n", .{what});
    }
}

// --- the conversion under test ---------------------------------------------

/// `fromInt` es generico sobre `anytype`, asi que necesita tipo destino.
fn asTorus(v: u64) Torus {
    return Torus.fromInt(v);
}

/// Fp2{a,b} -> Torus61{c0 = a, c1 = b}. The candidate maps `a` to `c0` and
/// `b` to `c1`; the mutation maps `a` to `c0` and drops `b`.
fn to_torus(x: Fp2) Q {
    return Q.new(asTorus(x.a.toU64()), asTorus(x.b.toU64()));
}

/// Torus61 -> Fp2. The extension has no public coordinate accessor, so the
/// pair is recovered from the serialization. The layout is NOT assumed: the
/// first half is checked to equal `Torus.fromInt(c0).toBytes()` before it is
/// believed. Assuming `c0 || c1` without checking is the same class of move
/// as two names describing one object.
const NB = @divExact(Q.NUM_BYTES, 2);

/// p + 1 = 2^61: the order of the norm-1 torus, and the largest domain our
/// Domain can be asked for. Copied as a number because the pin exposes no
/// constant for it and the check in `validate_two_adicity` needs one.
const torus_log_order: usize = 61;

fn byte_layout_is_c0_then_c1(a: u64, b: u64) bool {
    const full = Q.new(asTorus(a), asTorus(b)).toBytes();
    const lo = Torus.fromInt(a).toBytes();
    const hi = Torus.fromInt(b).toBytes();
    if (full[0..NB].len != lo.len or full[NB..].len != hi.len) return false;
    return std.mem.eql(u8, full[0..lo.len], &lo) and std.mem.eql(u8, full[lo.len..], &hi);
}

fn from_torus(t: Q) error{ConversionFailed}!Fp2 {
    const raw = t.toBytes();
    const lo = Torus.fromBytes(raw[0..NB]) catch return error.ConversionFailed;
    const hi = Torus.fromBytes(raw[NB..]) catch return error.ConversionFailed;
    return .{ .a = G.fromU64(@intCast(lo.toU64())), .b = G.fromU64(@intCast(hi.toU64())) };
}

/// The decomposition must be read in the right order before anything else is
/// worth testing. If `from_torus` swapped `c0` and `c1`, every round-trip
/// would still pass while comparing the wrong thing — the same failure as
/// two names describing one object.
fn validate_decomposition() !void {
    const cases = [_]struct { c0: u64, c1: u64 }{
        .{ .c0 = 0, .c1 = 0 },
        .{ .c0 = 1, .c1 = 0 },
        .{ .c0 = 0, .c1 = 1 }, // the imaginary unit alone
        .{ .c0 = 7, .c1 = 9 }, // BOTH parts non-zero
    };
    for (cases) |cs| {
        note("layout de bytes c0||c1 (pin)", byte_layout_is_c0_then_c1(cs.c0, cs.c1));
        // El check de arriba compara el PIN CONTRA SI MISMO: construye
        // Q.new(c0,c1) y lo contrasta con dos torus calculados aparte. Es una
        // buena comprobacion del pin, y durante mucho tiempo fue la UNICA sobre
        // layout de bytes — con el nombre de una asercion sobre "el layout".
        //
        // Lo que no hacia era llamar a NUESTRO serializador, que es lo que
        // entraria en un transcript. Intercambiar a y b en Fp2.toBytes
        // producia CERO discrepancias: 476 comprobaciones y ninguna lo veia.
        // Una asercion que no se puede fallar no es una asercion, y esta se
        // llevaba un nombre que prometia mas de lo que medía.
        const mine_bytes = Fp2.fromRaw(cs.c0, cs.c1).toBytes();
        const pin_bytes = Q.new(asTorus(cs.c0), asTorus(cs.c1)).toBytes();
        const layout_ok = std.mem.eql(u8, &mine_bytes, &pin_bytes);
        note("layout de bytes (mi toBytes == pin)", layout_ok);
        if (!layout_ok) {
            std.debug.print("    ({d},{d}): mio={x} pin={x}\n", .{ cs.c0, cs.c1, &mine_bytes, &pin_bytes });
        }
        const built = Q.new(asTorus(cs.c0), asTorus(cs.c1));
        const back = try from_torus(built);
        note("descomposicion orden c0,c1", back.a.toU64() == cs.c0 and back.b.toU64() == cs.c1);
        if (back.a.toU64() != cs.c0 or back.b.toU64() != cs.c1) {
            std.debug.print("    construido ({d},{d}) -> leido ({d},{d})\n", .{
                cs.c0, cs.c1, back.a.toU64(), back.b.toU64(),
            });
        }
    }
}

// --- checks -----------------------------------------------------------------

/// The anchor: arithmetic done by the PIN, decomposed, against the same
/// arithmetic done by me. This is a cross-check — the pin's `mul` is what
/// decides whether the conversion is right, not my own.
/// La guarda de canonicidad de `fromBytes`.
///
/// El codigo dice, en su propio comentario, que rechaza codificaciones no
/// canonicas "so rejection sampling works". Eso es una afirmacion sobre el
/// transcript, y este instrumento no la comprobaba: debilitar la guarda de `>=`
/// a `>` en la comparacion de `b` daba CERO discrepancias, porque nunca se
/// alimentaba una codificacion no canonica. Una propiedad que el codigo
/// afirma y el instrumento no mide es una propiedad documentada, no verificada.
/// Our Fp2's `two_adicity` against the pin's.
///
/// `two_adicity` is a member this file's own subject needed: the pin's
/// `proveOn` guards `log_domain` with `F.two_adicity`, and without it our Fp2
/// is not a field the pin can be instantiated over. The value is
/// `v2(p-1) + v2(p+1)` = 1 + 61 = 62 — arithmetic on the prime, but still a
/// number somebody typed, and a wrong one would be invisible: the FRI over ours
/// and the FRI over the pin's would then guard DIFFERENT domains, and every
/// differential between them would be measured over a range neither of them
/// actually allows.
///
/// This is what makes it a measurement instead of a constant. Verified by
/// mutation: 32 instead of 62 gives discrepancies and prints both values.
fn validate_two_adicity() void {
    note("two_adicity: igual que el del pin", Fp2.two_adicity == Q.two_adicity);
    if (Fp2.two_adicity != Q.two_adicity) {
        std.debug.print("    mio={d} pin={d}\n", .{ Fp2.two_adicity, Q.two_adicity });
    }
    // And it must hold the torus we actually use: the order-2^61 subgroup needs
    // a field admitting a 2-power of at least that size, which is the whole
    // reason our Domain exists.
    note("two_adicity >= torus_log_order", Fp2.two_adicity >= torus_log_order);
}

fn validate_canonicality() !void {
    const p: u64 = @intCast(G.p);

    // Canonical de partida: a y b ambos dentro del campo.
    var good = Fp2.fromRaw(7, 9).toBytes();
    note("fromBytes acepta lo canonico", (Fp2.fromBytes(&good) catch null) != null);

    // a == p: no canonico. Debe rechazarse.
    var bad_a = good;
    std.mem.writeInt(u64, bad_a[0..8], p, .little);
    note("fromBytes rechaza a == p", blk: {
        if (Fp2.fromBytes(&bad_a)) |_| break :blk false else |_| break :blk true;
    });

    // b == p: no canonico. Debe rechazarse. Es el caso que la guarda de `b`
    // cubre y el que una comparacion mal escrita dejaria pasar.
    var bad_b = good;
    std.mem.writeInt(u64, bad_b[8..16], p, .little);
    note("fromBytes rechaza b == p", blk: {
        if (Fp2.fromBytes(&bad_b)) |_| break :blk false else |_| break :blk true;
    });

    // p - 1 es el maximo canonico: tiene que ENTRAR. Si esto fallara, la guarda
    // estaria rechazando de mas y la mutacion de `>=` pasaria por buena.
    var edge = good;
    std.mem.writeInt(u64, edge[8..16], p - 1, .little);
    note("fromBytes acepta b == p - 1", blk: {
        if (Fp2.fromBytes(&edge)) |back| {
            break :blk back.b.toU64() == p - 1;
        } else |_| break :blk false;
    });
}

fn cmpArithmetic(a: Fp2, b: Fp2, comptime op: []const u8) !void {
    const ta = to_torus(a);
    const tb = to_torus(b);

    const theirs: Q = if (comptime std.mem.eql(u8, op, "add"))
        ta.add(tb)
    else if (comptime std.mem.eql(u8, op, "sub"))
        ta.sub(tb)
    else if (comptime std.mem.eql(u8, op, "mul"))
        ta.mul(tb)
    else
        ta.neg();

    const mine: Fp2 = if (comptime std.mem.eql(u8, op, "add"))
        a.add(b)
    else if (comptime std.mem.eql(u8, op, "sub"))
        a.sub(b)
    else if (comptime std.mem.eql(u8, op, "mul"))
        a.mul(b)
    else
        a.neg();

    const back = try from_torus(theirs);
    const ok = back.a.toU64() == mine.a.toU64() and back.b.toU64() == mine.b.toU64();
    note(op, ok);
    if (!ok) {
        std.debug.print("    ({d},{d}) {s} ({d},{d}): mio ({d},{d}) · pin ({d},{d})\n", .{
            a.a.toU64(),    a.b.toU64(),    op,             b.a.toU64(),    b.b.toU64(),
            mine.a.toU64(), mine.b.toU64(), back.a.toU64(), back.b.toU64(),
        });
    }
}

/// Round-trip in BOTH directions, which the first draft of this file was
/// going to skip. One direction proves nothing about the other.
fn cmpRoundTrip(v: Fp2) !void {
    const there_and_back = try from_torus(to_torus(v));
    note("rt Fp2->torus->Fp2", there_and_back.a.toU64() == v.a.toU64() and there_and_back.b.toU64() == v.b.toU64());
    if (there_and_back.a.toU64() != v.a.toU64() or there_and_back.b.toU64() != v.b.toU64()) {
        std.debug.print("    ida ({d},{d}) · vuelta ({d},{d})\n", .{
            v.a.toU64(), v.b.toU64(), there_and_back.a.toU64(), there_and_back.b.toU64(),
        });
    }

    // La segunda direccion NO puede ser `to_torus(from_torus(to_torus(v)))`
    // contra `to_torus(v)`: los dos lados pasan por la conversion, asi que
    // una conversion rota —descartar `b`— es invisible. Es el mismo defecto
    // que el `eql` reflexivo, aqui en mi propia prueba. Se ancla en la
    // construccion independiente del pin: `a + b*v` es `fromBase(a)` mas la
    // imaginaria por `fromBase(b)`, hecho con su aritmetica, no con la mia.
    const reference = Q.fromBase(asTorus(v.a.toU64())).add(
        Q.imaginaryUnit().mul(Q.fromBase(asTorus(v.b.toU64()))),
    );
    const mine_t = to_torus(v);
    note("conversion contra referencia del pin", mine_t.eql(reference));
    if (!mine_t.eql(reference)) {
        std.debug.print("    ({d},{d}): mio {x} · pin {x}\n", .{
            v.a.toU64(), v.b.toU64(), mine_t.toBytes(), reference.toBytes(),
        });
    }
}

pub fn main() !void {
    const p: u64 = (1 << 61) - 1;
    try validate_decomposition();
    try validate_canonicality();
    validate_two_adicity();

    // The corpus is chosen against where a broken conversion hides: `y != 0`
    // everywhere except one case (with `y = 0` a conversion could swap `a`
    // and `b` and never show), a case near `p` so a wrong reduction shows,
    // and both parts non-zero in most entries.
    const vals = [_]Fp2{
        .{ .a = G.zero, .b = G.zero },
        .{ .a = G.one, .b = G.zero }, // the ONE case with b == 0
        .{ .a = G.zero, .b = G.one }, // imaginary part alone
        .{ .a = G.one, .b = G.one },
        .{ .a = G.fromU64(7), .b = G.fromU64(9) },
        .{ .a = G.fromU64(p - 1), .b = G.fromU64(1) }, // near p, non-zero imag
        .{ .a = G.fromU64(1), .b = G.fromU64(p - 1) },
        .{ .a = G.fromU64(p - 2), .b = G.fromU64(p - 2) }, // both near p
        .{ .a = G.fromU64(0x0123_4567_89ab_cdef), .b = G.fromU64(0xfedc_ba98_7654_3210) },
        .{ .a = G.fromU64(1234567890123456789), .b = G.fromU64(987654321098765432) },
        .{ .a = G.fromU64(p / 2), .b = G.fromU64(p / 3) },
        .{ .a = G.fromU64(0xdead_beef), .b = G.fromU64(0x0bad_c0de) },
    };

    std.debug.print("p = {d}\n", .{p});
    std.debug.print("corpus: {d} valores, b != 0 en {d}\n\n", .{
        vals.len, vals.len - 1,
    });

    for (vals) |v| {
        try cmpRoundTrip(v);
        for (vals) |w| {
            try cmpArithmetic(v, w, "add");
            try cmpArithmetic(v, w, "sub");
            try cmpArithmetic(v, w, "mul");
        }
        try cmpArithmetic(v, Fp2{ .a = G.one, .b = G.zero }, "neg");
    }

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
    std.debug.print("\ndesglose (contado por la herramienta):\n", .{});
    var sum: usize = 0;
    for (rows[0..n]) |r| {
        std.debug.print("  {d:>4}  {s}\n", .{ r.count, r.name });
        sum += r.count;
    }
    std.debug.print("  {d:>4}  TOTAL\n\n", .{sum});
    std.debug.print("desglose con fallos por operacion (medido):\n", .{});
    for (rows[0..n]) |r| {
        const f = fails_by_check.get(r.name) orelse 0;
        std.debug.print("  {d:>4} de {d:<4} fallan  {s}\n", .{ f, r.count, r.name });
    }
    var pass_mutado: usize = 0;
    for (rows[0..n]) |r| pass_mutado += r.count - (fails_by_check.get(r.name) orelse 0);
    std.debug.print("\n  suman {d} de {d} que NO fallan\n", .{ pass_mutado, checks });
    std.debug.print("\ncomprobaciones: {d}\ndiscrepancias: {d}\n", .{ checks, failures });
    if (failures == 0) {
        std.debug.print("RESULTADO: la conversion es la identidad en todo lo comparado\n", .{});
    } else {
        std.debug.print("RESULTADO: HAY DIFERENCIAS — la conversion no es la identidad\n", .{});
    }
    // El paso de build se queda con el codigo de salida del proceso, asi que
    // imprimir "HAY DIFERENCIAS" y salir con 0 es una puerta en verde sobre un
    // fallo. Devolver un error es lo que el proceso traduce en exit code.
    if (failures > 0) return error.DifferentialFound;
}
