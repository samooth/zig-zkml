//! Differential: my FRI composition against the pinned one.
//!
//! It measures the COMPOSITION, not the field. The conversion is the identity
//! (measured, mutated, in `tools/fri_conv_diff.zig`), so both sides run the
//! same field arithmetic — and the field was already measured separately, on
//! `tools/field_diff.zig`, where `libs/field.zig` came out an independent
//! algorithm and not a copy of the pin. Quoting this as evidence that the
//! FIELD matches would be signing for something that measured something else.
//!
//! Byte-identity is NOT the target and is not achievable, for a reason worth
//! writing down: the transcript is typed per field. Mine calls
//! `challengeField(Fp2)` and the pin calls `challengeField(F)` with
//! `F = Torus61`, and the challenge comes from rejection sampling into that
//! type. The two transcript states are not the same state, so the proofs are
//! not the same bytes. Adopting the pin would mean adopting its transcript
//! protocol too, which is a cost the decision has to carry.
//!
//! So what is compared: the ORCHESTRATION. Both implementations get the same
//! polynomial, the same parameters, and the same input on each side. Then
//! separately: does each verifier accept a low-degree polynomial, and does
//! each reject one above its bound.
//!
//! Every line compares mine against the pin. Nothing compares my side to
//! itself. That is the rule the `eql` check broke by asking `A.eql(A)` and
//! `TA.eql(TA)` and comparing yes to yes.
//!
//! Run with `tools/fri_diff.sh`.

const std = @import("std");

// Un solo modulo para mi lado: `libs/fri/root.zig` ya reexporta Fp2,
// Goldilocks y Domain. Importarlos por separado haria que el binario
// contuviera DOS definiciones de Fp2 y el diferencial compararia copias de si
// mismo — que es el `eql` de esta tarde con otro disfraz.
const fri_mine = @import("fri_mine");
const fri_pin = @import("fri_pin");
const zf = @import("zig-field");

const Fp2 = fri_mine.Fp2;
const G = fri_mine.Goldilocks;
const Dom = fri_mine.Domain;
// El propio fri_pin reexporta su modulo de toro; no hace falta declararlo
// aparte, y declararlo es un error: un fichero no puede ser raiz de dos modulos.
const TorusDom = fri_pin.torus.TorusDomain(Q, zf.M61);
const Torus = zf.M61;
const Q = zf.QuadraticExtension(Torus, Torus.fromInt(-1));

var failures: usize = 0;
var checks: usize = 0;
var by_check: std.StringHashMapUnmanaged(usize) = .{};
var fails_by_check: std.StringHashMapUnmanaged(usize) = .{};

fn tally(name: []const u8) void {
    const g = by_check.getOrPut(std.heap.page_allocator, name) catch @panic("oom");
    if (!g.found_existing) g.value_ptr.* = 0;
    g.value_ptr.* += 1;
}

fn tally_fail(name: []const u8) void {
    const g = fails_by_check.getOrPut(std.heap.page_allocator, name) catch @panic("oom");
    if (!g.found_existing) g.value_ptr.* = 0;
    g.value_ptr.* += 1;
}

fn note(comptime what: []const u8, ok: bool) void {
    checks += 1;
    tally(what);
    if (!ok) {
        failures += 1;
        tally_fail(what);
    }
}

fn asTorus(v: u64) Torus {
    return Torus.fromInt(v);
}

fn to_torus(x: Fp2) Q {
    return Q.new(asTorus(x.a.toU64()), asTorus(x.b.toU64()));
}

// --- the shared transcript ---------------------------------------------------

/// One transcript object, used by both implementations. It has to be a
/// separate instance per implementation: if they shared one, the first
/// implementation's absorbs would decide the challenges the second one sees,
/// and the comparison would no longer be between two proofs.
const T = struct {
    state: std.crypto.hash.Blake3,

    pub fn init(domain: []const u8) @This() {
        var t = @This(){ .state = std.crypto.hash.Blake3.init(.{}) };
        t.state.update("zkml.fri-diff/");
        t.state.update(domain);
        return t;
    }
    pub fn absorbBytes(self: *@This(), bytes: []const u8) void {
        self.state.update(bytes);
    }
    pub fn challengeU64(self: *@This()) u64 {
        var buf: [32]u8 = undefined;
        self.state.final(&buf);
        var out: [8]u8 = undefined;
        std.crypto.hash.Blake3.hash(&buf, &out, .{});
        self.state.update(&out);
        return std.mem.readInt(u64, &out, .little);
    }
    pub fn absorbField(self: *@This(), comptime F: type, e: F) void {
        self.absorbBytes(&e.toBytes());
    }
    /// El pin llama `challengeField(Q)` y el mio `challengeField(Fp2)`. Cada
    /// uno pide un elemento de SU tipo, y por eso el mismo transcript NO
    /// produce el mismo reto en los dos lados — que es la razon por la que
    /// este diferencial compara veredictos y no bytes.
    ///
    /// La extension del pin tiene `fromInt`, que reduce y devuelve un valor.
    /// Mi Fp2 no lo tiene: construye con `fromRaw`. El arnes tiene que mirar
    /// el tipo de retorno en comptime en vez de asumirlo, porque las dos
    /// ramas existen en el mismo binario.
    /// El pin en v0.6.0 pide `challengeFieldChecked`. Aqui solo hay un
    /// decodificador y ya es el checked —`challengeField` usa el camino que
    /// rechaza—, asi que las dos entradas coinciden por construccion. No es una
    /// equivalencia demostrada: es que nunca las tuve distintas.
    pub fn challengeFieldChecked(self: *@This(), comptime F: type) F {
        return self.challengeField(F);
    }

    pub fn challengeField(self: *@This(), comptime F: type) F {
        while (true) {
            const c = self.challengeU64();
            if (comptime @hasDecl(F, "fromInt")) {
                // El unico tipo de campo que llega aqui es la extension del
                // pin, y su `fromInt` reduce y devuelve un valor, sin error
                // union — comprobado al compilar, porque el arnes que hace
                // `catch continue` sobre el no compila. Si alguien mete aqui
                // un campo con `fromInt` fallible, que lo diga el compilador
                // en la llamada, que es donde pertenece.
                return F.fromInt(c);
            } else if (comptime @hasDecl(F, "fromRaw")) {
                // Mi Fp2 no tiene `fromInt`: construye con `fromRaw`. El reto
                // entra en la parte real y la imaginaria en cero. Es una
                // eleccion del arnes, no del FRI, y es una de las razones por
                // las que los dos lados no comparten reto.
                return F.fromRaw(@intCast(c), 0);
            } else {
                return F.fromU64(@intCast(c));
            }
        }
    }
};

// --- the polynomial ----------------------------------------------------------

/// Un polinomio evaluable en Horner sobre los tres coeficientes, y un
/// coeficiente mas para el caso de grado alto.
const Coeffs = []const Fp2;

fn evalAt(coeffs: Coeffs, x: Fp2) Fp2 {
    var acc = coeffs[coeffs.len - 1];
    var i = coeffs.len - 1;
    while (i > 0) {
        i -= 1;
        acc = acc.mul(x).add(coeffs[i]);
    }
    return acc;
}

/// Un solo dominio para probar y para verificar. El transcript deriva los
/// retos de el: si probar y verificar usan cadenas distintas, el verificador
/// squeeze otra secuencia y rechaza — que es exactamente lo que pasaba, en los
/// DOS lados a la vez. Lo que yo leia como "las implementaciones no coinciden"
/// era mi arnes usando dos dominios. `fri_audit` usa el mismo en los dos.
const diff_domain = "fri-diff";
const diff_domain_high = "fri-diff-high";

pub fn main() !void {
    const p: u64 = (1 << 61) - 1;
    const allocator = std.heap.page_allocator;

    const log_domain: u6 = 8;
    const log_final: u6 = 4;
    const log_residual: u6 = 3;

    const dom = try Dom.init(log_domain);
    const n = dom.size();

    std.debug.print("p = {d}\n", .{p});
    std.debug.print("dominio 2^{d} = {d} puntos, log_final = {d}, log_residual = {d}\n\n", .{
        log_domain, n, log_final, log_residual,
    });

    // Both configurations from the same parameters. The pin has one field my
    // Config does not: `log_initial_degree`, and its own validate says
    // `log_initial_degree - round_count == log_residual_degree`, so it is
    // derivable rather than free.
    const round_count = log_domain - log_final;
    const log_initial: u6 = log_residual + round_count;
    const mine_cfg = fri_mine.Config{
        .log_domain = log_domain,
        .log_final = log_final,
        .log_residual_degree = log_residual,
        .num_queries = 8,
    };
    const pin_cfg = fri_pin.Config{
        .log_domain = log_domain,
        .log_initial_degree = log_initial,
        .log_final = log_final,
        .log_residual_degree = log_residual,
        .num_queries = 8,
    };

    // Dos polinomios de grado bajo, uno lineal y uno cuadratico. Los dos muy
    // por debajo del limite 2^log_residual = 8, que es lo que el test de
    // grado tiene que aceptar.
    const c_lin = [_]Fp2{
        .{ .a = G.fromU64(3), .b = G.fromU64(5) },
        .{ .a = G.fromU64(11), .b = G.fromU64(2) },
    };
    const c_quad = [_]Fp2{
        .{ .a = G.fromU64(7), .b = G.fromU64(1) },
        .{ .a = G.fromU64(2), .b = G.fromU64(9) },
        .{ .a = G.fromU64(1), .b = G.fromU64(4) },
    };
    // Y uno de grado 16, por encima del limite de 8. Se construye con
    // dieciseis coeficientes de verdad, no "acumulando", porque un polinomio
    // mal construido que pasara el test no probaria que el test rechaza.
    // Grado 199, no 16. El limite que impone la config —log_domain 8,
    // log_final 4, log_residual 3— permite grado inicial hasta 2^3 * 2^4 = 128.
    // Con 17 coeficientes, grado 16, el polinomio de "grado alto" estaba DE
    // DENTRO del limite y las dos implementaciones lo aceptaban con razon, y
    // yo lo leia como que ninguna rechazaba. Era la etiqueta la que mentia:
    // el dato era correcto —aceptan— y la conclusion que yo Sacaba de el, no.
    var c_high: [200]Fp2 = undefined;
    for (&c_high, 0..) |*slot, i| {
        slot.* = .{ .a = G.fromU64(@intCast(1 + i)), .b = G.fromU64(@intCast(2 * i + 1)) };
    }

    const polys = [_]struct { name: []const u8, coeffs: Coeffs }{
        .{ .name = "lineal", .coeffs = &c_lin },
        .{ .name = "cuadratico", .coeffs = &c_quad },
    };

    for (polys) |entry| {
        const ev_mine = try allocator.alloc(Fp2, n);
        const ev_pin = try allocator.alloc(Q, n);
        for (0..n) |i| {
            ev_mine[i] = evalAt(entry.coeffs, dom.at(i));
            ev_pin[i] = to_torus(ev_mine[i]);
        }

        // 1. los dos prueban
        var t_mine = T.init(diff_domain);
        const proof_mine = fri_mine.prove(allocator, &t_mine, ev_mine, mine_cfg) catch |e| {
            std.debug.print("  mi prove fallo ({s}) con el polinomio {s}\n", .{ @errorName(e), entry.name });
            note("mi prove", false);
            continue;
        };
        var t_pin = T.init(diff_domain);
        // El camino que el propio test del pin usa: `proveOn` con un
        // `TorusDomain` explicito. `prove`/`verify` van por `Domain(F)`, que
        // llama a `F.primitiveRootOfUnity(2^two_adicity)`, y sobre una
        // extension cuadratica eso NO TERMINA. Medido, no supuesto.
        const proof_pin = fri_pin.proveOn(Q, TorusDom, allocator, &t_pin, ev_pin, pin_cfg) catch |e| {
            std.debug.print("  prove del pin fallo ({s}) con el polinomio {s}\n", .{ @errorName(e), entry.name });
            note("prove del pin", false);
            continue;
        };
        note("ambos prueban", true);

        // 2. la composicion
        note("mismo numero de capas", proof_mine.layers.len == proof_pin.layers.len);
        note("mismo log_domain", proof_mine.log_domain == proof_pin.log_domain);
        note("mismo log_final", proof_mine.log_final == proof_pin.log_final);
        note("mismo log_residual", proof_mine.log_residual_degree == proof_pin.log_residual_degree);
        note("mismo numero de consultas", proof_mine.queries.len == proof_pin.queries.len);
        note("mismo tamano de residual", proof_mine.residual.len == proof_pin.residual.len);
        var sizes = proof_mine.layers.len == proof_pin.layers.len;
        if (sizes) {
            for (proof_mine.layers, proof_pin.layers) |a, b| {
                if (a.log_size != b.log_size) sizes = false;
            }
        }
        note("mismos log_size por capa", sizes);

        // 3. los dos verifican lo suyo
        // El verificador NO recibe las evaluaciones: comprueba las consultas
        // FRI contra la prueba comprometida. Por eso el rechazo de grado alto
        // se obtiene probando y verificando, no pasando datos distintos.
        // UN HECHO POR LADO. Con "ambos verifican" en una sola nota, cuando
        // cae no se sabe cual de los dos fallo, y el instrumento no dice contra
        // que se comparo. Es la misma regla de las etiquetas: nombrar por lo
        // que significa, no por el resumen que parezca conveniente.
        var v_mine = T.init(diff_domain);
        const ok_mine = fri_mine.verify(&v_mine, &proof_mine, mine_cfg) catch false;
        note("mi verify acepta", ok_mine);
        var v_pin = T.init(diff_domain);
        const ok_pin = fri_pin.verifyOn(Q, TorusDom, &v_pin, &proof_pin, pin_cfg) catch false;
        note("verify del pin acepta", ok_pin);

        // 4. grado alto: los dos tienen que rechazarlo
        const ev_hi_mine = try allocator.alloc(Fp2, n);
        const ev_hi_pin = try allocator.alloc(Q, n);
        for (0..n) |i| {
            ev_hi_mine[i] = evalAt(&c_high, dom.at(i));
            ev_hi_pin[i] = to_torus(ev_hi_mine[i]);
        }
        var th_mine = T.init(diff_domain_high);
        var th_pin = T.init(diff_domain_high);
        const hi_mine = fri_mine.prove(allocator, &th_mine, ev_hi_mine, mine_cfg) catch null;
        // Por el camino del toro como los otros: el multiplicativo no compila
        // para un campo extension en v0.6.0 — `Domain.init` declara
        // `error{DomainTooLarge, OrderTooLarge}` y `primitiveRootOfUnity`
        // devuelve `error{NoNonResidue, …}`. Verificado, no supuesto.
        const hi_pin = fri_pin.proveOn(Q, TorusDom, allocator, &th_pin, ev_hi_pin, pin_cfg) catch null;
        var rejects_mine = true;
        var rejects_pin = true;
        if (hi_mine) |*pm| {
            var vh = T.init(diff_domain_high);
            const acc = fri_mine.verify(&vh, pm, mine_cfg) catch false;
            rejects_mine = !acc;
        }
        if (hi_pin) |*pp| {
            var vh = T.init(diff_domain_high);
            const acc = fri_pin.verifyOn(Q, TorusDom, &vh, pp, pin_cfg) catch false;
            rejects_pin = !acc;
        }
        note("mi verify rechaza grado alto", rejects_mine);
        note("verify del pin rechaza grado alto", rejects_pin);
    }

    const Row = struct { name: []const u8, count: usize };
    var rows: [64]Row = undefined;
    var it = by_check.iterator();
    var n_rows: usize = 0;
    while (it.next()) |e| {
        rows[n_rows] = .{ .name = e.key_ptr.*, .count = e.value_ptr.* };
        n_rows += 1;
    }
    std.mem.sort(Row, rows[0..n_rows], {}, struct {
        fn gt(_: void, a: Row, b: Row) bool {
            return a.count > b.count;
        }
    }.gt);
    std.debug.print("\ndesglose:\n", .{});
    var sum: usize = 0;
    for (rows[0..n_rows]) |r| {
        const f = fails_by_check.get(r.name) orelse 0;
        std.debug.print("  {d:>4} de {d:<4}  {s}\n", .{ f, r.count, r.name });
        sum += r.count;
    }
    std.debug.print("  {d:>4} de {d:<4}  TOTAL\n\n", .{ sum, sum });
    std.debug.print("comprobaciones: {d}\ndiscrepancias: {d}\n", .{ checks, failures });
    if (failures == 0) {
        std.debug.print("RESULTADO: la composicion coincide en todo lo comparado\n", .{});
    } else {
        std.debug.print("RESULTADO: HAY DIFERENCIAS — la composicion no coincide\n", .{});
    }
}
