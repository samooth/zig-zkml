//! The bookkeeping every AIR in this directory needs: record constraints
//! as INDICES while building, then resolve them against the final buffers
//! once.
//!
//! Recording slices during the build is a use-after-free waiting to
//! happen: the backing ArrayLists reallocate as they grow, so every
//! Constraint written before a growth points into a buffer that no longer
//! exists. The first version of float_air.zig did exactly that and
//! segfaulted the verifier mid-walk. Indices survive reallocation; slices
//! are resolved once, at the end, against the final buffers.

const std = @import("std");
const expr = @import("./expr.zig");
const tensor = @import("../tensor/root.zig");

pub const Fp2 = expr.Fp2;
pub const Goldilocks = tensor.Goldilocks;
pub const Factor = expr.Factor;
pub const Scope = expr.Scope;
pub const System = expr.System;
pub const Term = expr.Term;
pub const Constraint = expr.Constraint;

pub const kOne = Fp2.one;
pub const kNegOne = Fp2.neg(Fp2.one);

pub fn g(v: u64) Fp2 {
    return Fp2.re(Goldilocks.fromU64(v));
}

/// -2^i as a field element, for bit-reconstruction coefficients.
pub fn kNegModPow2(i: u16) u64 {
    const p = Goldilocks.p;
    return p - ((@as(u64, 1) << @intCast(i)) % p);
}

/// A run of factors, recorded as INDICES while building.
pub const FRange = struct {
    first: usize,
    len: usize,
};

const TSpec = struct {
    factors: FRange,
    coefficient: Fp2,
};

pub const LinTerm = struct {
    factors: FRange,
    coefficient: Fp2 = kOne,
};

const CSpec = struct {
    name: []const u8,
    scope: Scope,
    first_term: usize,
    n_terms: usize,
};

pub const BuildError = error{OutOfMemory};

pub const Builder = struct {
    allocator: std.mem.Allocator,
    factors: std.ArrayList(Factor) = .empty,
    terms: std.ArrayList(TSpec) = .empty,
    constraints: std.ArrayList(CSpec) = .empty,

    pub fn init(allocator: std.mem.Allocator) Builder {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Builder) void {
        self.factors.deinit(self.allocator);
        self.terms.deinit(self.allocator);
        self.constraints.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn one(self: *Builder, col: u16) BuildError!FRange {
        try self.factors.append(self.allocator, .{ .column = .{ .index = col } });
        return .{ .first = self.factors.items.len - 1, .len = 1 };
    }

    pub fn constant(self: *Builder, v: Fp2) BuildError!FRange {
        try self.factors.append(self.allocator, .{ .constant = v });
        return .{ .first = self.factors.items.len - 1, .len = 1 };
    }

    pub fn pair(self: *Builder, a: u16, b: u16) BuildError!FRange {
        try self.factors.append(self.allocator, .{ .column = .{ .index = a } });
        try self.factors.append(self.allocator, .{ .column = .{ .index = b } });
        return .{ .first = self.factors.items.len - 2, .len = 2 };
    }

    /// A constraint whose terms are the given runs, in order. Every
    /// constraint in an AIR is built here, so the freeze in `Owned` is the
    /// only place slices are made.
    pub fn lin(self: *Builder, name: []const u8, scope: Scope, terms: []const LinTerm) BuildError!void {
        if (terms.len == 0) return;
        const first_term = self.terms.items.len;
        for (terms) |lt| {
            try self.terms.append(self.allocator, .{
                .factors = lt.factors,
                .coefficient = lt.coefficient,
            });
        }
        try self.constraints.append(self.allocator, .{
            .name = name,
            .scope = scope,
            .first_term = first_term,
            .n_terms = terms.len,
        });
    }

    pub fn count(self: *const Builder) usize {
        return self.constraints.items.len;
    }

    /// Copy-constraint: the two columns are equal. One term, degree 1.
    pub fn copy(self: *Builder, name: []const u8, dst: u16, src: u16) BuildError!void {
        try self.lin(name, .composed, &.{
            .{ .factors = try self.one(dst) },
            .{ .factors = try self.one(src), .coefficient = kNegOne },
        });
    }

    /// Booleanity of one witness column, the workhorse of every
    /// decomposition in this directory: x·(x − 1) = 0.
    pub fn boolean(self: *Builder, name: []const u8, col: u16) BuildError!void {
        try self.lin(name, .composed, &.{
            .{ .factors = try self.pair(col, col) },
            .{ .factors = try self.pair(col, col), .coefficient = kNegOne },
        });
    }

    /// "Either this value is zero, or it has an inverse": the field
    /// identity `x·x⁻¹ = 1` has no solution when x = 0, and `flag·x = 0`
    /// forces the zero case. Two constraints, degree 2, no range check and
    /// no comparison — the only way this IR can talk about zero.
    pub fn zeroOrNonZero(self: *Builder, name: []const u8, x: u16, inv: u16, flag: u16) BuildError!void {
        try self.lin(name, .composed, &.{
            .{ .factors = try self.pair(x, inv) },
            .{ .factors = try self.constant(kOne), .coefficient = kNegOne },
            .{ .factors = try self.one(flag) },
        });
        try self.lin(name, .composed, &.{
            .{ .factors = try self.pair(x, flag) },
        });
    }

    /// "This value is not zero", on its own: x·x⁻¹ = 1. Two constraints
    /// with the booleanity of the witness.
    pub fn nonZero(self: *Builder, name: []const u8, x: u16, inv: u16) BuildError!void {
        try self.lin(name, .composed, &.{
            .{ .factors = try self.pair(x, inv) },
            .{ .factors = try self.constant(kOne), .coefficient = kNegOne },
        });
        try self.boolean(name, inv);
    }
};

/// The finished system, owning every buffer its constraints point into.
pub const Owned = struct {
    allocator: std.mem.Allocator,
    factors: []Factor,
    terms: []Term,
    constraints: []Constraint,
    built: System,

    pub fn system(self: *const Owned) System {
        return self.built;
    }

    pub fn deinit(self: *Owned) void {
        self.allocator.free(self.factors);
        self.allocator.free(self.terms);
        self.allocator.free(self.constraints);
        self.* = undefined;
    }
};

/// Freeze the builder into a system with `rows` copies of every
/// constraint, one per row. Takes ownership of the builder's buffers; the
/// builder is left unusable.
pub fn freeze(allocator: std.mem.Allocator, b: *Builder, rows: usize) BuildError!Owned {
    std.debug.assert(rows > 0);
    const factor_buf = try b.factors.toOwnedSlice(allocator);
    errdefer allocator.free(factor_buf);
    const term_specs = try b.terms.toOwnedSlice(allocator);
    defer allocator.free(term_specs);
    const c_specs = try b.constraints.toOwnedSlice(allocator);
    defer allocator.free(c_specs);

    const terms = try allocator.alloc(Term, term_specs.len);
    errdefer allocator.free(terms);
    for (term_specs, 0..) |spec, i| {
        terms[i] = .{
            .factors = factor_buf[spec.factors.first .. spec.factors.first + spec.factors.len],
            .coefficient = spec.coefficient,
        };
    }

    const per_row = c_specs.len;
    const all = try allocator.alloc(Constraint, per_row * rows);
    for (0..rows) |r| {
        for (c_specs, 0..) |spec, i| {
            all[r * per_row + i] = .{
                .name = spec.name,
                .scope = spec.scope,
                .terms = terms[spec.first_term .. spec.first_term + spec.n_terms],
            };
        }
    }
    return .{
        .allocator = allocator,
        .factors = factor_buf,
        .terms = terms,
        .constraints = all,
        .built = .{ .constraints = all },
    };
}

/// A column-major trace: `columns[c][r]`.
pub const Trace = struct {
    rows: usize,
    columns: [][]Fp2,

    pub fn alloc(allocator: std.mem.Allocator, column_count: usize, rows: usize) BuildError!Trace {
        std.debug.assert(rows > 0);
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
        return .{ .rows = rows, .columns = cols };
    }

    pub fn deinit(self: *Trace, allocator: std.mem.Allocator) void {
        for (self.columns) |c| allocator.free(c);
        allocator.free(self.columns);
        self.* = undefined;
    }

    /// Bit i of `v` goes to COLUMN (base + i), row r. The trace is
    /// column-major, so a run of bits is a stride over columns, not a
    /// contiguous range. `v` is u64 because an fp32 significand plus its
    /// product needs 49 bits.
    pub fn writeBits(self: *Trace, base: u16, v: u64, n: u16, r: usize) void {
        for (0..n) |i| {
            self.columns[base + @as(u16, @intCast(i))][r] = Fp2.re(Goldilocks.fromU64((v >> @intCast(i)) & 1));
        }
    }
};
