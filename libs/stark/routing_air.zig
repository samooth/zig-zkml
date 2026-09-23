//! Compiled routing AIR (F2): the selected expert set IS the top-k.
//!
//! `libs/gadgets/routing` is shape-only — it describes the columns and a
//! degree-2 fragment, but nothing evaluates it, so a proof could claim any
//! expert set at all. This is the evaluable form.
//!
//! ## The statement
//!
//! Given gate scores `score_e ∈ [0, 2^m)` for e in [0, E), and k, the
//! selected set is exactly the k largest scores. The selected flags are
//! trace columns, so they are authenticated by the commitment; binding
//! them to the host engine's *public* expert-ID set is the same
//! preprocessed-input work as the LogUp table (TODO under F2).
//!
//! One row per expert, so E rows and E must be a power of two: the trace
//! domain is cyclic, and padding rows would need a per-row "this row is
//! real" flag, which this IR cannot pin without the very segment selectors
//! that are still open. DeepSeek's 256 happens to be a power of two;
//! Qwen's 60 is not, and that is a real limitation, not an oversight.
//!
//! ## How the comparison is expressed without a conditional
//!
//! The obvious constraint — "selected_e = 1 iff score_e >= T" — needs a
//! conditional, and conditionals are what this IR cannot do. The range
//! check hands one over for free: decompose
//!
//!   d_e = score_e - T + 2^m   into   m+1 bits
//!
//! and the TOP bit is 1 exactly when `d_e >= 2^m`, i.e. exactly when
//! `score_e >= T`. The selection flag is therefore the top bit, and the
//! only constraint needed is `selected_e = top_bit(d_e)` — degree 1.
//!
//! ## Why a prover-chosen threshold is still sound
//!
//! T is a witness column, and a careless reading says the prover can slide
//! it until it selects whatever it likes. It cannot: the cyclic
//! accumulator forces `sum_e (selected_e - k/E) = 0`, i.e. EXACTLY k
//! selections, and `selected_e = [score_e >= T]` then pins the set. Every
//! T in `(score_{k+1}, score_k]` yields the same set — the top-k — so the
//! threshold's value is irrelevant and the committed selected set is the
//! only thing the proof says. A test pins exactly that: two different
//! valid thresholds both verify and select identically.
//!
//! The `k/E` term is a per-row-uniform constant, not a per-row one, which
//! is what makes it expressible: E and k are known when the AIR is built.
//!
//! ## Ties at the boundary
//!
//! If `score_k == score_{k+1}` no T gives exactly k selections, since any
//! T takes the whole tie group or none of it. The trace builder refuses
//! such inputs rather than silently imposing a tie-break the AIR does not
//! encode; a convention belongs on the public-input side (F3).
//!
//! ## Scaling, honestly
//!
//! Each row spends `2m + 5 = 29` composed constraints (12+1 for the score
//! range, 13+1 for the difference, one tie, one selection) and uses the
//! same 30 columns for every row. E=16 therefore composes 465 constraints
//! and E=32 composes 929 — both under the verifier's
//! `max_composed_constraints` of 1024, and E=64 is over it.
//!
//! That ceiling is the honest limit of this design: for DeepSeek's 256
//! experts a per-row bit decomposition is simply the wrong tool, and ONE
//! LogUp argument over the score column would replace all 256 of them.
//! Same missing piece as the scale table (TODO under F2), which is why
//! this milestone pins E=16 rather than 256.

const std = @import("std");
const expr = @import("./expr.zig");
const range = @import("./range.zig");
const tensor = @import("../tensor/root.zig");

pub const Fp2 = expr.Fp2;
pub const Goldilocks = tensor.Goldilocks;
pub const System = expr.System;
pub const Constraint = expr.Constraint;
pub const Term = expr.Term;
pub const Factor = expr.Factor;

/// Gate scores are q4.12 fixed point: [0, 2^12) covers any sigmoid output
/// with 12 fractional bits and leaves room for raw pre-activation scores
/// up to 16. The bound belongs to the AIR, not to a witness choice: a
/// score outside it has no decomposition and the proof fails.
pub const score_bits: u8 = 12;
pub const score_modulus: u64 = @as(u64, 1) << score_bits;

// Trace layout. Every expert is a ROW of the same columns.
pub const col_score: u16 = 0;
pub const col_score_bits: u16 = 1; // `score_bits` wide
pub const col_dshift: u16 = col_score_bits + score_bits; // score_e - T + 2^m
pub const col_d_bits: u16 = col_dshift + 1; // `score_bits` + 1 wide
pub const col_selected: u16 = col_d_bits + score_bits + 1;
pub const col_tau: u16 = col_selected + 1;
pub const col_acc: u16 = col_tau + 1;
pub const column_count: usize = col_acc + 1;

/// Index (within the trace's columns) of the difference's TOP bit: 1 iff
/// score_e >= T.
pub const col_top_bit: u16 = col_d_bits + score_bits;

const kBias = Fp2.re(Goldilocks.fromU64(score_modulus));
const kNegBias = Fp2.neg(kBias);
const kNegOne = Fp2.neg(Fp2.one);

// Static storage: the tie and select constraints are IDENTICAL for every
// row (one row = one expert), so all E copies share these terms.
const kDshiftF = [_]Factor{.{ .column = .{ .index = col_dshift } }};
const kScoreF = [_]Factor{.{ .column = .{ .index = col_score } }};
const kTauF = [_]Factor{.{ .column = .{ .index = col_tau } }};
// The bias enters ONCE, as a coefficient on a constant-1 factor. Putting
// 2^m in the factor AND in the coefficient squares it into -2^24, which is
// a real bug this file had for one iteration.
const kOneF = [_]Factor{.{ .constant = Fp2.one }};

/// d_e = score_e - T + 2^m
const kTieTerms = [_]Term{
    .{ .factors = &kDshiftF },
    .{ .factors = &kScoreF, .coefficient = kNegOne },
    .{ .factors = &kTauF },
    .{ .factors = &kOneF, .coefficient = kNegBias },
};

const kSelectedF = [_]Factor{.{ .column = .{ .index = col_selected } }};
const kTopBitF = [_]Factor{.{ .column = .{ .index = col_top_bit } }};

/// selected_e = top_bit(d_e)
const kSelectTerms = [_]Term{
    .{ .factors = &kSelectedF },
    .{ .factors = &kTopBitF, .coefficient = kNegOne },
};

const kAccNextF = [_]Factor{.{ .column = .{ .index = col_acc, .offset = 1 } }};
const kAccF = [_]Factor{.{ .column = .{ .index = col_acc } }};
const kAccZeroTerms = [_]Term{.{ .factors = &kAccF }};

/// k/E as a field element. Exact: E is a power of two, so 1/E exists.
fn frac(k: u64, rows: usize) Goldilocks {
    const e = @as(u64, rows);
    const inv_e = Goldilocks.fromU64(e).inv() catch unreachable;
    return Goldilocks.fromU64(k).mul(inv_e);
}

pub const BuildError = error{
    OutOfMemory,
    BadWidth,
    /// E must be a power of two (see the module docs on padding rows).
    NotPowerOfTwo,
    /// k == 0 or k > E.
    KOutOfRange,
};

/// The routing AIR for `rows` experts and `k` selections, owning every
/// allocation its System points into.
pub const Owned = struct {
    inner: range.BuiltSystem,
    acc_terms: []Term,
    acc_factors: []Factor,

    pub fn system(self: *const Owned) System {
        return self.inner.system;
    }

    pub fn deinit(self: *Owned) void {
        // Read the allocator first: inner.deinit poisons the struct.
        const gpa = self.inner.allocator;
        gpa.free(self.acc_terms);
        gpa.free(self.acc_factors);
        self.inner.deinit();
        self.* = undefined;
    }
};

pub fn buildSystem(allocator: std.mem.Allocator, rows: usize, k: u64) BuildError!Owned {
    if (!std.math.isPowerOfTwo(rows)) return BuildError.NotPowerOfTwo;
    if (k == 0 or k > rows) return BuildError.KOutOfRange;

    // The accumulator constraint is the only one whose coefficients depend
    // on (k, E), so its terms/factors are the only owned ones here.
    const acc_terms = try allocator.alloc(Term, 4);
    errdefer allocator.free(acc_terms);
    const acc_factors = try allocator.alloc(Factor, 6);
    errdefer allocator.free(acc_factors);

    acc_factors[0] = .{ .column = .{ .index = col_acc, .offset = 1 } };
    acc_factors[1] = .{ .column = .{ .index = col_acc } };
    acc_factors[2] = .{ .column = .{ .index = col_selected } };
    acc_factors[3] = .{ .constant = Fp2.re(frac(k, rows)) };

    acc_terms[0] = .{ .factors = acc_factors[0..1] };
    acc_terms[1] = .{ .factors = acc_factors[1..2], .coefficient = kNegOne };
    acc_terms[2] = .{ .factors = acc_factors[2..3], .coefficient = kNegOne };
    acc_terms[3] = .{ .factors = acc_factors[3..4] };

    const base = [_]Constraint{
        .{
            .name = "acc' - acc - selected + k/E",
            .scope = .composed,
            .terms = acc_terms[0..],
        },
        .{ .name = "acc[0] = 0", .scope = .boundary_first, .terms = &kAccZeroTerms },
    };

    // Two range checks per row: the score, and the shifted difference.
    const n_specs = 2 * rows;
    const specs = try allocator.alloc(range.Spec, n_specs);
    defer allocator.free(specs);
    for (0..rows) |r| {
        specs[2 * r] = .{ .column = col_score, .bit_base = col_score_bits, .width = score_bits };
        specs[2 * r + 1] = .{ .column = col_dshift, .bit_base = col_d_bits, .width = score_bits + 1 };
    }

    var inner = range.BuiltSystem.init(allocator, .{ .constraints = &base }, specs) catch |e| switch (e) {
        error.OutOfMemory => return BuildError.OutOfMemory,
        error.BadWidth => return BuildError.BadWidth,
    };
    errdefer inner.deinit();

    // Append E tie constraints and E select constraints. They are
    // identical per row, so the merged array holds copies that share the
    // static term storage above.
    const n = inner.constraints.len;
    const grown = try allocator.alloc(Constraint, n + 2 * rows);
    @memcpy(grown[0..n], inner.constraints);
    for (0..rows) |r| {
        grown[n + r] = .{
            .name = "dshift = score - T + 2^m",
            .scope = .composed,
            .terms = &kTieTerms,
        };
        grown[n + rows + r] = .{
            .name = "selected = top_bit(dshift)",
            .scope = .composed,
            .terms = &kSelectTerms,
        };
    }
    allocator.free(inner.constraints);
    inner.constraints = grown;
    inner.system = .{ .constraints = grown };

    return .{ .inner = inner, .acc_terms = acc_terms, .acc_factors = acc_factors };
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

pub const BuildTraceError = error{
    OutOfMemory,
    NotPowerOfTwo,
    KOutOfRange,
    EmptyScores,
    /// score_k == score_{k+1}: no threshold selects exactly k.
    TieAtBoundary,
    /// The explicit threshold does not select exactly k experts.
    ThresholdSelectsWrongCount,
    /// a score outside [0, 2^m), i.e. outside the AIR's declared range.
    ScoreOutOfRange,
};

/// Goldilocks has no ordering, and the canonical representative is its
/// u64 form, which is what the range checks decompose anyway.
fn lessThan(_: void, a: Goldilocks, b: Goldilocks) bool {
    return a.toU64() < b.toU64();
}

/// Build an honest trace attesting that the k largest of `scores` are the
/// selected ones.
///
/// The threshold is the k-th largest score: the largest T that still
/// yields exactly k selections. Any other valid T proves the same
/// statement — `buildTraceAtTau` exists so a test can check that.
pub fn buildTrace(
    allocator: std.mem.Allocator,
    scores: []const Goldilocks,
    k: u64,
) BuildTraceError!Trace {
    const rows = scores.len;
    if (rows == 0) return BuildTraceError.EmptyScores;
    if (!std.math.isPowerOfTwo(rows)) return BuildTraceError.NotPowerOfTwo;
    if (k == 0 or k > rows) return BuildTraceError.KOutOfRange;
    for (scores) |s| {
        if (s.toU64() >= score_modulus) return BuildTraceError.ScoreOutOfRange;
    }

    // Sort a copy ascending to find the k-th largest and check the tie.
    const sorted = try allocator.dupe(Goldilocks, scores);
    defer allocator.free(sorted);
    std.mem.sort(Goldilocks, sorted, {}, lessThan);
    const kth: usize = @intCast(k);
    if (kth < rows and sorted[rows - kth].eql(sorted[rows - kth - 1])) {
        return BuildTraceError.TieAtBoundary;
    }
    // tau = the k-th largest. `selected_e = [score_e >= tau]`, so the
    // threshold must sit in (score_{k+1}, score_k]: at the k-th largest
    // exactly the top k clear it. Sorted ascending that is sorted[rows-k].
    // When k == rows everything is selected and tau = 0 works, because
    // scores are non-negative.
    const tau: Goldilocks = if (kth == rows) Goldilocks.zero else sorted[rows - kth];
    return buildTraceAtTau(allocator, scores, k, tau);
}

/// Build the trace at an EXPLICIT threshold. Refuses a threshold that does
/// not select exactly k experts: that is the whole soundness argument, so
/// it is checked here rather than left to the AIR to reject later.
pub fn buildTraceAtTau(
    allocator: std.mem.Allocator,
    scores: []const Goldilocks,
    k: u64,
    tau: Goldilocks,
) BuildTraceError!Trace {
    const rows = scores.len;
    if (rows == 0) return BuildTraceError.EmptyScores;
    if (!std.math.isPowerOfTwo(rows)) return BuildTraceError.NotPowerOfTwo;
    if (k == 0 or k > rows) return BuildTraceError.KOutOfRange;
    if (tau.toU64() >= score_modulus) return BuildTraceError.ScoreOutOfRange;
    for (scores) |s| {
        if (s.toU64() >= score_modulus) return BuildTraceError.ScoreOutOfRange;
    }
    {
        var above: usize = 0;
        for (scores) |s| {
            if (s.toU64() >= tau.toU64()) above += 1;
        }
        if (above != k) return BuildTraceError.ThresholdSelectsWrongCount;
    }

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

    // Bits live in one contiguous run of `score_bits` columns, so the
    // column index of bit b of the score is col_score_bits + b and of the
    // difference is col_d_bits + b.
    for (0..rows) |r| {
        const s = scores[r].toU64();
        cols[col_score][r] = Fp2.re(scores[r]);
        for (0..score_bits) |b| {
            cols[col_score_bits + b][r] = Fp2.re(Goldilocks.fromU64((s >> @intCast(b)) & 1));
        }
        // d = score - tau + 2^m, in [1, 2^(m+1)) because both score and
        // tau are in [0, 2^m).
        const d = s + score_modulus - tau.toU64();
        cols[col_dshift][r] = Fp2.re(Goldilocks.fromU64(d));
        for (0..score_bits + 1) |b| {
            cols[col_d_bits + b][r] = Fp2.re(Goldilocks.fromU64((d >> @intCast(b)) & 1));
        }
        const selected: u64 = if (s >= tau.toU64()) 1 else 0;
        cols[col_selected][r] = Fp2.re(Goldilocks.fromU64(selected));
        cols[col_tau][r] = Fp2.re(tau);
    }

    // acc[0] = 0, acc[r+1] = acc[r] + selected_r - k/E. The last row's
    // wrap is what closes the cycle and forces sum(selected) = k.
    const step = frac(k, rows);
    var acc = Goldilocks.zero;
    cols[col_acc][0] = Fp2.zero;
    for (0..rows - 1) |r| {
        const sel = cols[col_selected][r].a;
        acc = acc.add(sel).sub(step);
        cols[col_acc][r + 1] = Fp2.re(acc);
    }

    return .{ .rows = rows, .columns = cols };
}

/// The expert IDs the trace attests as selected. Reads the committed
/// `selected` column; a caller that wants to check them against a public
/// set must open this column (F3 plumbing).
pub fn selectedFlags(trace: *const Trace) []Fp2 {
    return trace.columns[col_selected];
}
