//! Q4_K binding for the GEMM operands (F2): range checks plus the
//! dequantization equation.
//!
//! The GEMM AIR proves `C = A·B` over the trace's own a/b columns. Without
//! a binding to the quantized weights, a prover could choose any operands
//! it liked and still produce a valid proof. This adds the missing half for
//! a Q4_K-style symmetric block:
//!
//!   every row:  a = (nib_a - 8) * scale_a
//!                b = (nib_b - 8) * scale_b
//!                nib_a in [0, 15],  nib_b in [0, 15]   (GGML raw - 8)
//!
//! The nibble bounds come from `libs/stark/range.zig` (bit decomposition
//! plus `bit*(bit-1) = 0`), so they compose with the existing quadratic IR.
//!
//! ## What the scale column does and does not buy
//!
//! The scale is a witness column constrained only by the equation above.
//! That binds the *representation* — each operand is a raw 4-bit nibble
//! times one scale — but NOT the scale's provenance: proving it came from a
//! real fp16 / GGML sub-scale table needs the lookup half of LogUp, which
//! the IR cannot express yet (BLUE_PRINT §7.4, TODO under F2). A negative
//! test pins this boundary: an operand with a *valid* nibble but a scale it
//! could not have had is currently provable, and that is the known gap.
//!
//! Column layout (on top of gemm_air's four):
//!
//!   4  nib_a       unsigned Q4_K nibble of the A operand
//!   5  nib_b       same for B
//!   6..9   bits_a  binary expansion of nib_a
//!   10..13 bits_b  binary expansion of nib_b
//!   14 scale_a
//!   15 scale_b
//!
//! Rows that are not MAC rows still satisfy both the dequantization
//! equation and the range check, so both stay plain `composed` constraints
//! (no per-row masking, which the IR cannot express):
//!
//!   padding rows   a = b = 0  ->  nibble 8, scale 0
//!   closing row    a = 1, b = -C  ->  nibble 9, scale = the operand
//!                  itself, so (9 - 8) * scale = operand
//!
//! The closing row is synthetic by construction (see gemm_air.zig); giving
//! it nibble 9 rather than exempting it keeps the constraint set
//! homogeneous and the range check applies to it too.

const std = @import("std");
const expr = @import("./expr.zig");
const range = @import("./range.zig");
const gemm_air = @import("./gemm_air.zig");
const tensor = @import("../tensor/root.zig");

pub const Fp2 = expr.Fp2;
pub const System = expr.System;
pub const Constraint = expr.Constraint;
pub const Term = expr.Term;
pub const Factor = expr.Factor;
pub const Goldilocks = tensor.Goldilocks;

pub const col_nib_a: u16 = 4;
pub const col_nib_b: u16 = 5;
pub const col_bits_a: u16 = 6; // 6..9
pub const col_bits_b: u16 = 10; // 10..13
pub const col_scale_a: u16 = 14;
pub const col_scale_b: u16 = 15;
pub const column_count: usize = 16;

pub const nibble_width: u8 = 4;

/// Container-scope storage: System holds pointers into these.
///
///   a - (nib_a - 8)*scale_a = a - nib_a*scale_a + 8*scale_a
///
/// Note the last term is `8*scale_a`, NOT the constant 8: the bias is
/// inside the product with the scale.
const kAFactors = [_]Factor{.{ .column = .{ .index = gemm_air.col_a } }};
const kNibScaleA = [_]Factor{
    .{ .column = .{ .index = col_nib_a } },
    .{ .column = .{ .index = col_scale_a } },
};
const kEightScaleA = [_]Factor{.{ .column = .{ .index = col_scale_a } }};
const kDequantATerms = [_]Term{
    .{ .factors = &kAFactors },
    .{ .factors = &kNibScaleA, .coefficient = Fp2.neg(Fp2.one) },
    .{ .factors = &kEightScaleA, .coefficient = Fp2.re(Goldilocks.fromU64(8)) },
};

const kBFactors = [_]Factor{.{ .column = .{ .index = gemm_air.col_b } }};
const kNibScaleB = [_]Factor{
    .{ .column = .{ .index = col_nib_b } },
    .{ .column = .{ .index = col_scale_b } },
};
const kEightScaleB = [_]Factor{.{ .column = .{ .index = col_scale_b } }};
const kDequantBTerms = [_]Term{
    .{ .factors = &kBFactors },
    .{ .factors = &kNibScaleB, .coefficient = Fp2.neg(Fp2.one) },
    .{ .factors = &kEightScaleB, .coefficient = Fp2.re(Goldilocks.fromU64(8)) },
};

const kDequantConstraints = [_]Constraint{
    .{ .name = "a = (nib_a - 8)*scale_a", .scope = .composed, .terms = &kDequantATerms },
    .{ .name = "b = (nib_b - 8)*scale_b", .scope = .composed, .terms = &kDequantBTerms },
};

/// The range checks the operand columns need.
pub const range_specs = [_]range.Spec{
    .{ .column = col_nib_a, .bit_base = col_bits_a, .width = nibble_width },
    .{ .column = col_nib_b, .bit_base = col_bits_b, .width = nibble_width },
};

pub const BuildError = error{ OutOfMemory, BadWidth };

/// Owns the merged system: GEMM AIR + nibble range checks + dequant
/// equations. `system()` stays valid until `deinit`.
pub const BoundSystem = struct {
    inner: range.BuiltSystem,

    pub fn system(self: BoundSystem) System {
        return self.inner.system;
    }

    pub fn deinit(self: *BoundSystem) void {
        self.inner.deinit();
    }
};

/// GEMM AIR + dequantization equations + nibble range checks.
pub fn buildSystem(allocator: std.mem.Allocator) BuildError!BoundSystem {
    var inner = range.BuiltSystem.init(allocator, gemm_air.system(), &range_specs) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.BadWidth => return error.BadWidth,
    };
    errdefer inner.deinit();

    // Append the dequantization equations. BuiltSystem allocates the merged
    // slice once; growing it here means reallocating that one slice.
    const n = inner.constraints.len;
    const grown = try allocator.alloc(Constraint, n + kDequantConstraints.len);
    @memcpy(grown[0..n], inner.constraints);
    @memcpy(grown[n..], &kDequantConstraints);
    allocator.free(inner.constraints);
    inner.constraints = grown;
    inner.system = .{ .constraints = grown };
    return .{ .inner = inner };
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

pub const BindError = error{ OutOfMemory, NibbleOutOfRange, InconsistentOperands };

/// Extend a GEMM trace with the quantization columns.
///
/// `nib_a` / `nib_b` hold one raw Q4_K nibble (0..15) per MAC; the block
/// scales are given as the engine's q4.22 Goldilocks values. The GEMM
/// operand columns must then equal `(nibble - 8) * scale` exactly, or the
/// trace is not a dequantization of anything and is refused here (the AIR
/// would reject it too, but later and less legibly).
pub fn bindOperands(
    allocator: std.mem.Allocator,
    gemm_trace: *const gemm_air.Trace,
    nib_a: []const u8,
    scale_a: Goldilocks,
    nib_b: []const u8,
    scale_b: Goldilocks,
) BindError!Trace {
    const rows = gemm_trace.rows;
    const real = gemm_air.realRowsFor(nib_a.len);
    if (nib_b.len != nib_a.len or real + 1 > rows) return BindError.InconsistentOperands;

    const cols = try allocator.alloc([]Fp2, column_count);
    errdefer allocator.free(cols);
    var made: usize = 0;
    errdefer for (cols[0..made]) |c| allocator.free(c);
    for (0..column_count) |i| {
        const buf = try allocator.alloc(Fp2, rows);
        @memset(buf, Fp2.zero);
        cols[i] = buf;
        made += 1;
    }
    // Carry the GEMM columns through unchanged.
    for (gemm_trace.columns, 0..) |src, i| @memcpy(cols[i], src);

    for (0..rows) |r| {
        const nib_a_row: u8 = if (r < real) nib_a[r] else @as(u8, if (r == rows - 1) 9 else 8);
        const nib_b_row: u8 = if (r < real) nib_b[r] else @as(u8, if (r == rows - 1) 9 else 8);
        if (nib_a_row > 15 or nib_b_row > 15) return BindError.NibbleOutOfRange;

        cols[col_nib_a][r] = Fp2.re(Goldilocks.fromU64(nib_a_row));
        cols[col_nib_b][r] = Fp2.re(Goldilocks.fromU64(nib_b_row));
        for (0..nibble_width) |bit| {
            const shift: u3 = @intCast(bit);
            cols[col_bits_a + bit][r] = Fp2.re(Goldilocks.fromU64((nib_a_row >> shift) & 1));
            cols[col_bits_b + bit][r] = Fp2.re(Goldilocks.fromU64((nib_b_row >> shift) & 1));
        }

        // Scales: real rows use the block scale, padding rows zero, and the
        // closing row (nibble 9) derives the scale from the synthetic
        // operand so (9 - 8) * scale reproduces it.
        const a_val = gemm_trace.columns[gemm_air.col_a][r].a;
        const b_val = gemm_trace.columns[gemm_air.col_b][r].a;
        const scale_a_row: Goldilocks = if (r < real) scale_a else if (r == rows - 1) a_val else Goldilocks.zero;
        const scale_b_row: Goldilocks = if (r < real) scale_b else if (r == rows - 1) b_val else Goldilocks.zero;
        cols[col_scale_a][r] = Fp2.re(scale_a_row);
        cols[col_scale_b][r] = Fp2.re(scale_b_row);

        if (r < real) {
            const eight = Goldilocks.fromU64(8);
            const want_a = Goldilocks.fromU64(nib_a_row).sub(eight).mul(scale_a_row);
            const want_b = Goldilocks.fromU64(nib_b_row).sub(eight).mul(scale_b_row);
            if (!want_a.eql(a_val) or !want_b.eql(b_val)) return BindError.InconsistentOperands;
        }
    }
    return .{ .rows = rows, .columns = cols };
}
