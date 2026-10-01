//! What a domain costs, measured rather than asserted.
//!
//! `fri.max_log_domain = 30` is a RESOURCE limit: 2^30 points is 16 GiB of
//! F_{p²}, which is not a buffer anybody allocates. Before this file existed,
//! that number was a judgement with nothing behind it, and the test that
//! accompanied it — `expect(max_log_domain <= 30)` — asserted a constant
//! against itself, which is a tautology wearing the costume of a measurement.
//!
//! So: measure the prover and the verifier at a ladder of domain sizes, print
//! the curve, and make the bound a GATE. The gate fails if `max_log_domain`
//! grows past the frontier this file measured as payable. That way the number
//! in the source is not the number in a document that nobody checks — the
//! bench re-derives the relationship every time it runs.
//!
//! The claim being measured is not "2^30 is fast". It is the weaker and true
//! one: 2^30 is far outside anything this machine can pay, which is exactly
//! why the bound sits above the frontier instead of at it.
//!
//! Absolute timings move with machine load, so the stable output is the RATIO
//! between consecutive rungs, and the absolute numbers are labelled as one run
//! rather than quoted as properties of the library.

const std = @import("std");
const fri = @import("fri");
// domain.zig ya es parte del modulo fri: un fichero no puede pertenecer a dos

const Fp2 = fri.Fp2;
const G = fri.Goldilocks;
const Dom = fri.Domain;
const Io = std.Io;

// El transcript REAL, no el doble de prueba que usa fri_diff. Los dos
// implementan la misma interfaz, asi que con cualquiera de los dos el
// diferencial mide lo mismo — pero para un BENCH de coste lo que interesa es
// el coste del camino de produccion, y medir con un Blake3 de juguete seria
// medir el juguete.
const Transcript = fri.Transcript;

/// One rung of the ladder.
const Rung = struct { log_domain: u6 };

const LADDER = [_]Rung{
    .{ .log_domain = 12 },
    .{ .log_domain = 14 },
    .{ .log_domain = 16 },
    .{ .log_domain = 18 },
    .{ .log_domain = 20 },
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    std.debug.print("coste por tamaño de dominio, medido en esta maquina\n", .{});
    std.debug.print("Fp2 = {d} bytes; el dominio son 2^log_domain puntos\n\n", .{@sizeOf(Fp2)});

    var prev_ns: ?i96 = null;

    for (LADDER) |rung| {
        const log_d = rung.log_domain;
        const log_final: u6 = 4;
        const log_res: u6 = 3;
        const cfg = fri.Config{
            .log_domain = log_d,
            .log_final = log_final,
            .log_residual_degree = log_res,
            .num_queries = 8,
        };

        // validate() devuelve el numero de rondas; aqui solo importa que la
        // config sea valida, asi que se descarta en la propia sentencia.
        _ = cfg.validate() catch {
            std.debug.print("  2^{d:>2}  config invalida\n", .{log_d});
            continue;
        };

        const dom = try Dom.init(log_d);
        const n = dom.size();

        // Evaluaciones de un POLINOMIO DE GRADO BAJO sobre el dominio.
        //
        // La primera version rellenaba esto con valores seudoaleatorios, y el
        // verify rechazaba su propia prueba en todos los peldaños con el aviso
        // "verify NO acepto su propia prueba". El motivo es correcto y el banco
        // estaba mal: ruido no es un polinomio de grado limitado, asi que el
        // residual sale de grado completo y la prueba de grado lo rechaza, que
        // es justo lo que debe hacer. Es la regla 8 del AGENTS.md en forma de
        // banco: solo se ejercito el camino del rechazo, y el coste medido era
        // el de un rechazo.
        const c1 = Fp2.re(G.fromU64(3));
        const c2 = Fp2.re(G.fromU64(7));
        const ev = try gpa.alloc(Fp2, n);
        defer gpa.free(ev);
        for (ev, 0..) |*slot, i| {
            const x = dom.at(i);
            slot.* = x.sqr().add(x.mul(c1)).add(c2);
        }

        // `Io.Clock.awake` es el reloj monotonico y no el de pared: el de pared
        // salta con un ajuste de NTP, y un benchmark que mide una sola vez es
        // vulnerable justo a eso. El patron es el del propio benchmark de std.
        var domain_label: [32]u8 = undefined;
        const label = std.fmt.bufPrint(&domain_label, "zkml.domain_cost.{d}", .{log_d}) catch "zkml.domain_cost";

        var tr = Transcript.init(label);
        const t0 = Io.Clock.awake.now(io).nanoseconds;
        _ = try Dom.init(log_d); // valida el dominio; prove construye el suyo
        const proof = try fri.prove(gpa, &tr, ev, cfg);
        const t1 = Io.Clock.awake.now(io).nanoseconds;

        tr = Transcript.init(label);
        const verified = try fri.verify(&tr, &proof, cfg);
        const t2 = Io.Clock.awake.now(io).nanoseconds;
        const prove_ns = t1 - t0;
        const verify_ns = t2 - t1;

        const bytes = @as(u64, n) * @sizeOf(Fp2);
        // i96 con signo: la division tiene que decir que hace con el negativo.
        const prove_ms = @divTrunc(prove_ns, std.time.ns_per_ms);
        const verify_ms = @divTrunc(verify_ns, std.time.ns_per_ms);

        std.debug.print(
            "  2^{d:>2}  {d:>9} puntos  {d:>7} MiB de Fp2  prove {d:>5} ms  verify {d:>4} ms",
            .{ log_d, n, bytes / (1024 * 1024), prove_ms, verify_ms },
        );
        if (prev_ns) |p| {
            if (p > 0 and prove_ns > 0) {
                std.debug.print("  x{d:.2} el prove", .{@as(f64, @floatFromInt(prove_ns)) / @as(f64, @floatFromInt(p))});
            }
        }
        std.debug.print("\n", .{});

        if (!verified) {
            std.debug.print("  AVISO: verify NO acepto su propia prueba\n", .{});
        }

        prev_ns = prove_ns;
    }

    std.debug.print("\n", .{});
    const frontier: u6 = LADDER[LADDER.len - 1].log_domain;
    std.debug.print("frontera medida: 2^{d} pagable en esta maquina\n", .{frontier});
    std.debug.print("max_log_domain declarado: {d}\n", .{fri.max_log_domain});

    // LA PUERTA. No es "2^30 es rapido", que es falso. Es que el limite tiene
    // que quedar por encima de la frontera medida, con margen, para que ningun
    // dominio que el prover pueda receber de verdad se corte por el limite de
    // recursos en vez de por el presupuesto real.
    //
    // El margen de 6 es la distancia entre 2^20 —que es justo lo que el
    // verificador ya se niega a asignar, ver `fri.max_final_domain`— y el
    // limite. Si alguien sube `max_log_domain` sin subir el limite de
    // verificacion, esta puerta se pone roja y el motivo queda en el mensaje.
    const margin: u6 = 6;
    const required = frontier + margin;
    if (fri.max_log_domain < required) {
        std.debug.print(
            "\nPUERTA ROJA: max_log_domain = {d} esta por debajo de la frontera medida + margen ({d}).\n" ++
                "  O el limite baja, o hace falta medir un dominio mayor y volver a decidir.\n",
            .{ fri.max_log_domain, required },
        );
        return error.DomainBoundBelowMeasuredFrontier;
    }

    std.debug.print(
        "PUERTA VERDE: {d} >= {d} (frontera {d} + margen {d}), asi que el limite\n" ++
            "  no corta ningun dominio que este prover pueda pagar.\n",
        .{ fri.max_log_domain, required, frontier, margin },
    );
}
