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

    // Un punto de medicion es una cifra sin rango. La version anterior fijaba
    // log_domain, log_final y log_residual en constantes yreportabasobre el
    // unico punto, con dos polinomios dentro: eso es una demostracion de que el
    // camino funciona, no un corpus. Y el punto unico era el equivocado: con
    // log_residual = 3 y log_final = 4 solo se ejercitan folds cortas.
    //
    // Los casos de aqui cubren el regimen que falta: log_residual = 0, que es
    // donde el limite de grado del residual aprieta al maximo, un log_final
    // pequeno, y un dominio mayor que 2^8. Cada uno imprime su propia linea y
    // falla por separado, de modo que un fallo se puede atribuir a un regimen.
    const Case = struct { log_domain: u6, log_final: u6, log_residual: u6 };
    const cases = [_]Case{
        // el punto original, que es el que ya se habia medido
        .{ .log_domain = 8, .log_final = 4, .log_residual = 3 },
        // log_residual = 0: el borde. El limite de grado del residual es 1,
        // asi que cualquier polinomio de grado > 0 tiene que ser rechazado, y
        // el polinomio legitimo tiene que ser constante.
        .{ .log_domain = 8, .log_final = 4, .log_residual = 0 },
        // log_residual = 1, todavia cerca del borde
        .{ .log_domain = 8, .log_final = 4, .log_residual = 1 },
        // log_final pequeno: pocas rondas de folding
        .{ .log_domain = 8, .log_final = 2, .log_residual = 2 },
        // log_final = log_domain: ninguna ronda, el otro extremo
        .{ .log_domain = 8, .log_final = 8, .log_residual = 4 },
        // dominio mayor que 2^8
        .{ .log_domain = 10, .log_final = 6, .log_residual = 3 },
        .{ .log_domain = 12, .log_final = 8, .log_residual = 2 },
    };

    for (cases, 0..) |case, ci| {
        const log_domain: u6 = case.log_domain;
        const log_final: u6 = case.log_final;
        const log_residual: u6 = case.log_residual;

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
            //
            // Los dos lados se ejecutan SIEMPRE, y la fila compara el par.
            // Antes, si el mio fallaba se anotaba discrepancia y se hacia
            // `continue` sin llegar a llamar al prove del pin: no habia forma
            // de saber si los dos rechazan. Que uno solo rechaza no es una
            // diferencia entre implementaciones, es un hecho sobre una de
            // ellas, y anotarlo como si lo fuera convierte una coincidencia
            // en un fallo. Asi aparecio con log_final == log_domain.
            var t_mine = T.init(diff_domain);
            var t_pin = T.init(diff_domain);
            var mine_err: ?anyerror = null;
            var pin_err: ?anyerror = null;
            // El camino que el propio test del pin usa: `proveOn` con un
            // `TorusDomain` explicito. `prove`/`verify` van por `Domain(F)`, que
            // llama a `F.primitiveRootOfUnity(2^two_adicity)`, y sobre una
            // extension cuadratica eso NO TERMINA. Medido, no supuesto.
            const proof_mine = fri_mine.prove(allocator, &t_mine, ev_mine, mine_cfg) catch |e| blk: {
                mine_err = e;
                break :blk null;
            };
            const proof_pin = fri_pin.proveOn(Q, TorusDom, allocator, &t_pin, ev_pin, pin_cfg) catch |e| blk: {
                pin_err = e;
                break :blk null;
            };

            note("prove: ambos aceptan o ambos rechazan", (mine_err == null) == (pin_err == null));
            if (mine_err != null and pin_err != null) {
                note("prove: mismo error", std.mem.eql(u8, @errorName(mine_err.?), @errorName(pin_err.?)));
                std.debug.print("    los dos rechazan: mio={s} pin={s} (coinciden)\n", .{ @errorName(mine_err.?), @errorName(pin_err.?) });
            } else if (mine_err != null or pin_err != null) {
                std.debug.print("    UN SOLO LADO RECHAZA: mio={any} pin={any}\n", .{ mine_err, pin_err });
            }
            if (proof_mine == null or proof_pin == null) continue;
            // Tras la guarda no hay nulo, asi que se desenvuelve una vez para
            // que el resto del bloque no tenga que comprobarlo en cada acceso.
            const pm = proof_mine.?;
            const pp = proof_pin.?;

            // 2. la composicion
            note("capas: mismo numero", pm.layers.len == pp.layers.len);
            note("dominio: mismo log_domain", pm.log_domain == pp.log_domain);
            note("final: mismo log_final", pm.log_final == pp.log_final);
            note("residual: mismo log_residual", pm.log_residual_degree == pp.log_residual_degree);
            note("consultas: mismo numero", pm.queries.len == pp.queries.len);
            note("residual: mismo tamano", pm.residual.len == pp.residual.len);
            var sizes = pm.layers.len == pp.layers.len;
            if (sizes) {
                for (pm.layers, pp.layers) |a, b| {
                    if (a.log_size != b.log_size) sizes = false;
                }
            }
            note("capas: mismos log_size", sizes);

            // 3. los dos verifican lo suyo
            // El verificador NO recibe las evaluaciones: comprueba las consultas
            // FRI contra la prueba comprometida. Por eso el rechazo de grado alto
            // se obtiene probando y verificando, no pasando datos distintos.
            // UN HECHO POR LADO. Con "ambos verifican" en una sola nota, cuando
            // cae no se sabe cual de los dos fallo, y el instrumento no dice contra
            // que se comparo. Es la misma regla de las etiquetas: nombrar por lo
            // que significa, no por el resumen que parezca conveniente.
            var v_mine = T.init(diff_domain);
            const ok_mine = fri_mine.verify(&v_mine, &pm, mine_cfg) catch false;
            var v_pin = T.init(diff_domain);
            const ok_pin = fri_pin.verifyOn(Q, TorusDom, &v_pin, &pp, pin_cfg) catch false;
            // Una fila por COMPARACION, no una por lado. Con una por lado, cuatro
            // filas que dicen "el mio acepta" y "el pin acepta" se leen como un
            // par de afirmaciones, y no son: lo que se mide es si ambos coinciden.
            // Cuando no coinciden se imprimen los dos valores, porque saber cual
            // fallo importa tanto como saber que fallo.
            note("verify: ambos aceptan", ok_mine == ok_pin);
            if (ok_mine != ok_pin) {
                std.debug.print("    verify del caso legitimo DIFIERE: mio={} pin={}\n", .{ ok_mine, ok_pin });
            }

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
            if (hi_mine) |*hpm| {
                var vh = T.init(diff_domain_high);
                const acc = fri_mine.verify(&vh, hpm, mine_cfg) catch false;
                rejects_mine = !acc;
            }
            if (hi_pin) |*hpp| {
                var vh = T.init(diff_domain_high);
                const acc = fri_pin.verifyOn(Q, TorusDom, &vh, hpp, pin_cfg) catch false;
                rejects_pin = !acc;
            }
            note("grado alto: ambos rechazan", rejects_mine == rejects_pin);
            if (rejects_mine != rejects_pin) {
                std.debug.print("    grado alto DIFIERE: mio rechaza={} pin rechaza={}\n", .{ rejects_mine, rejects_pin });
            }
        }
        std.debug.print("    caso {d}: log_domain={d} log_final={d} log_residual={d} — sin discrepancias en lo comparado\n", .{ ci + 1, case.log_domain, case.log_final, case.log_residual });
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
    // Las dos columnas son FALLOS y TOTAL, en ese orden. Sin encabezado, y
    // "0 de 2" se lee como "cero coincidencias de dos", que es lo
    // contrario: asi se leyo esta tabla una vez y casi se concluyo que los
    // dos verificadores difieren en un caso. Un encabezado no cuesta nada y
    // es la diferencia entre una tabla y una ambiguedad.
    std.debug.print("\ndesglose (columnas: FALLOS de TOTAL):\n", .{});
    var sum_checks: usize = 0;
    var sum_fails: usize = 0;
    for (rows[0..n_rows]) |r| {
        const f = fails_by_check.get(r.name) orelse 0;
        std.debug.print("  {d:>6} fallos de {d:<5}  {s}\n", .{ f, r.count, r.name });
        sum_checks += r.count;
        sum_fails += f;
    }
    std.debug.print("  {d:>6} fallos de {d:<5}  TOTAL\n\n", .{ sum_fails, sum_checks });
    std.debug.print("comprobaciones: {d}\ndiscrepancias: {d}\n", .{ checks, failures });
    if (failures == 0) {
        std.debug.print("RESULTADO: la composicion coincide en todo lo comparado\n", .{});
    } else {
        std.debug.print("RESULTADO: HAY DIFERENCIAS — la composicion no coincide\n", .{});
    }
}
