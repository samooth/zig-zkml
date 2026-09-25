//! Compiled AIR for the GEMM gadget (F2): the running-sum constraint the
//! gadget's `airFragment` only described, plus the boundary constraints
//! that tie the sum to the claimed output.
//!
//! This is the file that connects `libs/gadgets/gemm` (shape) to
//! `libs/stark` (enforcement). Without the boundaries the AIR is vacuous —
//! the all-zero trace satisfies `s' = s + a*b` — which is why
//! `airFragment`'s degree-2 fragment is not enough on its own.
//!
//! ## Why there is a closing row
//!
//! The trace domain is cyclic, so the composed constraint holds at EVERY
//! row, including the wrap from the last row to row 0. A real GEMM output
//! is not zero, so a prefix sum cannot simply wrap. The last row is
//! therefore a synthetic closing row, and it is the one row where the
//! composed constraints are EXEMPT (`System.transition_exemptions = 1`,
//! Winterfell's default). Two boundary constraints carry the row instead,
//! both checked on authenticated values at fixed rows:
//!
//!   s[0]     = 0
//!   s[last]  = c[last]      the claimed output
//!
//! Together they force `c[last]` to equal the sum of the trace's real
//! products: the sum is telescoping over rows 0..last-1, all of which are
//! enforced, and `s[last]` is what the verifier reads at the last row. A
//! prover claiming a different output must either break the sum (rejected
//! on a non-exempt row) or move `s[last]` (rejected by the boundary).
//!
//! The exemption is what lets the closing row carry NO operand binding.
//! The old pins `a[last] = 1`, `b[last] = -c[last]` made the closing row a
//! dequantized operand of `(1, -C)`, and `-C` is not in the image of
//! `fp16ToFixedQ4_22` for any C, so a per-row scale-provenance constraint
//! could never hold there. See quant_binding.zig and the plan's F2 notes.
//!
//! Trace layout (one output element per trace; multi-tile needs segment
//! selectors, tracked in TODO under F2):
//!
//!   col 0  a   the i-th MAC's A element (dequantized)
//!   col 1  b   the i-th MAC's B element
//!   col 2  s   running sum, s[0] = 0
//!   col 3  c   claimed output element

const std = @import("std");
const tensor = @import("../tensor/root.zig");
const expr = @import("./expr.zig");

pub const Goldilocks = tensor.Goldilocks;
pub const Fp2 = expr.Fp2;
pub const System = expr.System;
pub const Constraint = expr.Constraint;
pub const Term = expr.Term;
pub const Factor = expr.Factor;
pub const ColumnRef = expr.ColumnRef;

/// MACs per trace row in the *chunked* layout the gadget describes. The
/// composed constraint here is one MAC per row, because `s' = s + a*b`
/// carries a single product: aggregating 16 MACs into a row would need the
/// chunked-sum form (per-chunk range checks), which depends on the LogUp
/// argument still open. Tracked in TODO under F2.
pub const macs_per_row: usize = 16;

pub const col_a: u16 = 0;
pub const col_b: u16 = 1;
pub const col_s: u16 = 2;
pub const col_c: u16 = 3;
pub const column_count: usize = 4;

// Container-scope storage: System holds pointers into these.
const kNegOne = Fp2.neg(Fp2.one);

const kFactorsSNext = [_]Factor{.{ .column = .{ .index = col_s, .offset = 1 } }};
const kFactorsS = [_]Factor{.{ .column = .{ .index = col_s } }};
const kFactorsAB = [_]Factor{
    .{ .column = .{ .index = col_a } },
    .{ .column = .{ .index = col_b } },
};
const kRunningSumTerms = [_]Term{
    .{ .factors = &kFactorsSNext },
    .{ .factors = &kFactorsS, .coefficient = kNegOne },
    .{ .factors = &kFactorsAB, .coefficient = kNegOne },
};

const kFirstRowTerms = [_]Term{
    .{ .factors = &kFactorsS },
};
const kCloseCFactors = [_]Factor{.{ .column = .{ .index = col_c } }};
/// s[last] - c[last]: the claim is read straight off the telescoped sum.
const kCloseOutputTerms = [_]Term{
    .{ .factors = kFactorsS[0..1] },
    .{ .factors = kCloseCFactors[0..1], .coefficient = kNegOne },
};

const kConstraints = [_]Constraint{
    .{
        .name = "s' - s - a*b",
        .scope = .composed,
        .terms = &kRunningSumTerms,
    },
    .{ .name = "s[0] = 0", .scope = .boundary_first, .terms = &kFirstRowTerms },
    .{ .name = "s[last] = c[last]", .scope = .boundary_last, .terms = &kCloseOutputTerms },
};

pub fn system(k: usize) BuildError!System {
    const rows = try rowsFor(k);
    return .{
        .constraints = &kConstraints,
        .trace_rows = rows,
        .transition_exemptions = 1,
    };
}

/// Real MAC rows: one per MAC.
pub fn realRowsFor(k: usize) usize {
    return k;
}

/// Total trace rows: one row per MAC and the synthetic closing row. The
/// domain must be a power of two, so only k = 2^m - 1 is representable.
pub fn rowsFor(k: usize) BuildError!usize {
    if (k == 0) return BuildError.EmptyReduction;
    const rows = k + 1;
    if (!std.math.isPowerOfTwo(rows)) return BuildError.UnsupportedTraceSize;
    return rows;
}

pub const Trace = struct {
    rows: usize,
    columns: [][]Fp2,

    pub fn deinit(self: *Trace, allocator: std.mem.Allocator) void {
        for (self.columns) |c| allocator.free(c);
        allocator.free(self.columns);
        self.* = undefined;
    }
};

pub const BuildError = error{
    OutOfMemory,
    /// k == 0: there is nothing to reduce.
    EmptyReduction,
    /// a/b shorter than the reduction they claim to represent.
    ShortOperands,
    /// the operands do not actually multiply to the claimed output.
    OutputMismatch,
    /// k + 1 is not a power of two, so padding would be required.
    UnsupportedTraceSize,
};

/// Build an honest trace for the output element `sum(a[i] * b[i])`.
///
/// `a` and `b` are the dequantized MAC operands (Goldilocks, q4.22
/// scaled, as produced by `tensor.dequantQ4K`); `claimed_output` is what
/// the prover wants to attest as C. Returning OutputMismatch here would
/// be a convenience — the AIR rejects a wrong claim too, via the cycle
/// failing to close (see the tests).
pub fn buildTrace(
    allocator: std.mem.Allocator,
    a: []const Goldilocks,
    b: []const Goldilocks,
    claimed_output: Goldilocks,
) BuildError!Trace {
    if (a.len == 0 or a.len != b.len) return BuildError.EmptyReduction;
    const rows = try rowsFor(a.len);
    const real_rows = realRowsFor(a.len);

    const cols = try allocator.alloc([]Fp2, column_count);
    errdefer allocator.free(cols);
    const av = try allocator.alloc(Fp2, rows);
    errdefer allocator.free(av);
    const bv = try allocator.alloc(Fp2, rows);
    errdefer allocator.free(bv);
    const sv = try allocator.alloc(Fp2, rows);
    errdefer allocator.free(sv);
    const cv = try allocator.alloc(Fp2, rows);
    errdefer allocator.free(cv);

    var total: Goldilocks = Goldilocks.zero;
    for (0..real_rows) |row| {
        av[row] = Fp2.re(a[row]);
        bv[row] = Fp2.re(b[row]);
        cv[row] = Fp2.zero;
        sv[row] = Fp2.re(total);
        total = total.add(a[row].mul(b[row]));
    }

    // Closing row (the last): exempt from the composed constraint, so its a
    // and b are free and the binder needs no scale there. What matters is
    // that s[last] is the telescoped sum and c[last] states it.
    const last = rows - 1;
    av[last] = Fp2.zero;
    bv[last] = Fp2.zero;
    sv[last] = Fp2.re(total);
    cv[last] = Fp2.re(claimed_output);

    cols[0] = av;
    cols[1] = bv;
    cols[2] = sv;
    cols[3] = cv;
    return .{ .rows = rows, .columns = cols };
}

/// The value the AIR actually attests: the sum of the trace's products.
pub fn traceOutput(trace: *const Trace) Goldilocks {
    return trace.columns[col_s][trace.rows - 1].a;
}
