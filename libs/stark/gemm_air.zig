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
//! therefore a synthetic closing row:
//!
//!   a[last] = 1
//!   b[last] = -(sum of the real MACs)
//!   s[last] = +(sum of the real MACs)   == the claimed output
//!
//! which makes the wrap evaluate to `s[0] = s[last] + 1 * (-C) = 0` — the
//! cycle closes. Three boundary constraints pin it down, all checked on
//! authenticated values at fixed rows:
//!
//!   s[0]     = 0
//!   a[last]  = 1
//!   b[last]  = -(c[last])
//!
//! Together they force `c[last]` to equal the sum of the trace's real
//! products. A prover claiming a different output must either break the
//! cycle (rejected by the composed constraint) or change its own a/b
//! products (rejected by the boundary). Binding a/b to the actual
//! quantized weights is the dequant/range argument — LogUp, still open.
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
const kFactorsA = [_]Factor{.{ .column = .{ .index = col_a } }};
const kCloseA = [_]Term{
    // a[last] - 1
    .{ .factors = &kFactorsA },
    .{ .factors = &[_]Factor{.{ .constant = Fp2.one }}, .coefficient = kNegOne },
};
const kFactorsB = [_]Factor{.{ .column = .{ .index = col_b } }};
const kFactorsC = [_]Factor{.{ .column = .{ .index = col_c } }};
const kCloseB = [_]Term{
    // b[last] + c[last]
    .{ .factors = &kFactorsB },
    .{ .factors = &kFactorsC },
};

const kConstraints = [_]Constraint{
    .{
        .name = "s' - s - a*b",
        .scope = .composed,
        .terms = &kRunningSumTerms,
    },
    .{ .name = "s[0] = 0", .scope = .boundary_first, .terms = &kFirstRowTerms },
    .{ .name = "a[last] = 1", .scope = .boundary_last, .terms = &kCloseA },
    .{ .name = "b[last] = -c[last]", .scope = .boundary_last, .terms = &kCloseB },
};

/// The GEMM AIR: composed running sum + the three closing constraints.
pub fn system() System {
    return .{ .constraints = &kConstraints };
}

/// Real MAC rows: one per MAC.
pub fn realRowsFor(k: usize) usize {
    return k;
}

/// Total trace rows: the MAC rows, the synthetic closing row, and enough
/// zero-padding to reach a power of two (the FRI domain size). Padding rows
/// carry a = b = 0, so they leave the running sum untouched.
pub fn rowsFor(k: usize) usize {
    const needed = k + 1;
    return std.math.ceilPowerOfTwo(usize, needed) catch needed;
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
    const rows = rowsFor(a.len);
    const real_rows = realRowsFor(a.len);
    if (real_rows + 1 > rows) return BuildError.EmptyReduction;

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

    // Padding rows between the real ones and the closing row: a = b = 0,
    // so the running sum holds still.
    var pad: usize = real_rows;
    while (pad < rows - 1) : (pad += 1) {
        av[pad] = Fp2.zero;
        bv[pad] = Fp2.zero;
        sv[pad] = Fp2.re(total);
        cv[pad] = Fp2.zero;
    }

    // Closing row (the last): pins the cycle to the claimed output.
    const last = rows - 1;
    av[last] = Fp2.one;
    bv[last] = Fp2.re(Goldilocks.zero.sub(claimed_output));
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
