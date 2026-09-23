//! A right shift by a witnessed amount, as an AIR gadget — the barrel
//! shifter, which is the one piece of machinery three queued items all
//! need: a subnormal RESULT in the multiply (shift the significand right
//! by the amount the exponent fell short), the exponent alignment in an
//! fp32 accumulator, and a subnormal source in a widening (shift left by
//! its leading-zero count).
//!
//! The shape is the standard one: decompose the shift amount into bits and
//! apply one conditional shift per bit, from the top down, so after stage
//! k the vector has been shifted by the top k bits of the amount. Each
//! stage is a mux per output bit (degree 2) plus a sticky that remembers
//! everything shifted out — which is what lets the caller round with the
//! right answer instead of losing the tail.
//!
//! Two things this IR cannot say directly, and how they are handled:
//!
//!   - "bit i - 2^k" does not exist for i < 2^k. The constraint reads that
//!     branch as the constant ZERO, so a shift larger than the vector
//!     flushes to zero instead of pointing at a column that isn't there.
//!   - OR is an inequality, so the sticky uses the same sum/inverse trick
//!     as everywhere else, except where two booleans meet: x OR y is
//!     x + y - x·y, one constraint, and it is boolean for free.
//!
//! SCOPE: the amount must be less than or equal to `amount_bits`'s range,
//! i.e. at most 2^k - 1, which the caller proves with its own constraint
//! if it needs a tighter bound. A shift that flushes the whole vector to
//! zero still produces the right answer here, with sticky = 1.

const std = @import("std");
const expr = @import("./expr.zig");
const bld = @import("./air_builder.zig");

pub const Fp2 = expr.Fp2;
pub const Builder = bld.Builder;
pub const LinTerm = bld.LinTerm;

pub const GadgetError = bld.BuildError;

pub const Config = struct {
    /// Columns holding the input vector, LSB first.
    in_base: u16,
    /// How many bits the vector has.
    width: u16,
    /// Columns holding the shift amount, LSB first. `amount_bits` must be
    /// 2 or more: the first stage shifts by 1, so an amount of 0 or 1 bit
    /// cannot express "shift by 0 or 1" without a special case.
    amount_base: u16,
    amount_bits: u16,
    /// Columns for the shifted vector, LSB first.
    out_base: u16,
    /// Column for the OR of everything shifted out, and the one before it
    /// in the layout if the caller wants a running chain.
    sticky_col: u16,
};

/// Columns a Config needs, beyond the input and the amount: the output
/// vector, every stage's intermediate vector, the prefix-OR columns that
/// feed the sticky, the per-stage "this stage shifted" flags, and the
/// sticky chain.
pub fn column_cost(cfg: Config) usize {
    const stages: u16 = cfg.amount_bits;
    // One vector per stage, the prefix ORs, one flag per stage, the
    // internal sticky chain, and the caller's own sticky column.
    return cfg.width * stages + prefixTotal(cfg) + stages + (stages - 1) + 1;
}

/// Where the gadget's own columns sit inside a caller's layout. Exposed
/// because the witness has to fill exactly these.
pub const Bases = struct {
    stage: u16,
    prefix: u16,
    flag: u16,
    sticky_chain: u16,

    pub fn of(comptime cfg: Config) Bases {
        const stages: u16 = cfg.amount_bits;
        const stage: u16 = cfg.out_base;
        const prefix: u16 = stage + cfg.width * stages;
        const flag: u16 = prefix + @as(u16, @intCast(prefixTotal(cfg)));
        return .{ .stage = stage, .prefix = prefix, .flag = flag, .sticky_chain = flag + stages };
    }

    /// The vector after stage k, where stage 0 is the first shift applied
    /// (the top bit of the amount) and the last stage holds the answer.
    pub fn vector(self: Bases, cfg: Config, k: u16) u16 {
        return self.stage + cfg.width * (cfg.amount_bits - 1 - k);
    }

    pub fn prefixOf(self: Bases, cfg: Config, k: u16) u16 {
        return self.prefix + @as(u16, @intCast(prefixOffset(cfg, k)));
    }

    pub fn flagOf(self: Bases, k: u16) u16 {
        return self.flag + k;
    }

    /// Where stage k's accumulated sticky goes. The LAST stage writes the
    /// caller's column, because that is the answer; the rest chain.
    pub fn stickyOf(self: Bases, cfg: Config, k: u16) u16 {
        return if (k == cfg.amount_bits - 1) cfg.sticky_col else self.sticky_chain + k;
    }
};

/// Emit the gadget's constraints into `b`. The caller owns the layout:
/// everything here is an index into columns it has already allocated.
pub fn cname(comptime what: []const u8, comptime k: u16, comptime i: u16) []const u8 {
    return comptime std.fmt.comptimePrint("barrel: stage {d} {s} {d}", .{ k, what, i });
}

pub fn build(b: *Builder, comptime cfg: Config) GadgetError!void {
    const stages: u16 = cfg.amount_bits;
    std.debug.assert(cfg.amount_bits >= 2);

    // stage_k's vector lives at out_base + width*(stages-1-k), so the
    // stages read the previous one top down and the last is the answer.
    // Every base is annotated so the layout arithmetic is comptime: an
    // unannotated const would be a runtime usize and every column index
    // below would need an @intCast.
    const bases: Bases = .of(cfg);
    const stage_base: u16 = bases.stage;
    const prefix_base: u16 = bases.prefix;
    const flag_base: u16 = bases.flag;

    inline for (0..stages) |k| {
        const bit: u16 = @intCast(stages - 1 - k);
        const ctrl: u16 = cfg.amount_base + bit;
        try b.boolean(cname("amount bit is boolean", bit, 0), ctrl);
    }

    inline for (0..stages) |k| {
        // Stage k applies the amount's bit (stages-1-k), so it shifts
        // by 2^(stages-1-k): the top bit of the amount moves the vector
        // furthest, which is what makes the stages composable.
        const amount: u16 = @as(u16, 1) << @intCast(stages - 1 - @as(u16, @intCast(k)));
        // Stage k reads what stage k-1 wrote, which sits one slot
        // higher because the slots run backwards from the answer.
        const src: u16 = if (k == 0) cfg.in_base else stage_base + cfg.width * @as(u16, @intCast(stages - @as(u16, @intCast(k))));
        const dst: u16 = stage_base + cfg.width * @as(u16, @intCast(stages - 1 - k));
        const ctrl: u16 = cfg.amount_base + @as(u16, @intCast(stages - 1 - k));
        const flag: u16 = flag_base + @as(u16, @intCast(k));
        const sticky: u16 = bases.stickyOf(cfg, @as(u16, @intCast(k)));

        // A right shift of the VALUE moves bit i to bit i - amount, so
        // out[i] = in[i + amount]; the bits that fall off the bottom are
        // the ones the sticky remembers, and the bits that come in at the
        // top are the constant ZERO.
        inline for (0..cfg.width) |i| {
            const d: u16 = @intCast(i);
            if (d + amount >= cfg.width) {
                try b.lin(cname("flushes high bit", @as(u16, @intCast(k)), d), .composed, &.{
                    .{ .factors = try b.one(dst + d) },
                    .{ .factors = try b.one(src + d), .coefficient = bld.kNegOne },
                    .{ .factors = try b.pair(ctrl, src + d) },
                });
            } else {
                try b.lin(cname("shifts bit", @as(u16, @intCast(k)), d), .composed, &.{
                    .{ .factors = try b.one(dst + d) },
                    .{ .factors = try b.pair(ctrl, src + d + amount), .coefficient = bld.kNegOne },
                    .{ .factors = try b.one(src + d), .coefficient = bld.kNegOne },
                    .{ .factors = try b.pair(ctrl, src + d), .coefficient = Fp2.one },
                });
            }
        }

        // The bits this stage can shift out are the low `amount` of the
        // CURRENT vector, so each stage's prefix OR chain starts fresh:
        // prefix[j] = src[j] OR prefix[j-1], with prefix[-1] = 0. One
        // constraint per bit, and the result is boolean for free because
        // it is the OR of two booleans.
        const prefix: u16 = prefix_base + @as(u16, @intCast(prefixOffset(cfg, @as(u16, @intCast(k)))));
        inline for (0..amount) |j| {
            const col: u16 = prefix + @as(u16, @intCast(j));
            if (j == 0) {
                try b.copy(cname("first shifted-out bit is the input", @as(u16, @intCast(k)), @as(u16, @intCast(j))), col, src);
                continue;
            }
            try b.lin(cname("shifted-out prefix", @as(u16, @intCast(k)), @as(u16, @intCast(j))), .composed, &.{
                .{ .factors = try b.one(col) },
                .{ .factors = try b.one(src + @as(u16, @intCast(j))), .coefficient = bld.kNegOne },
                .{ .factors = try b.one(col - 1), .coefficient = bld.kNegOne },
                .{ .factors = try b.pair(src + @as(u16, @intCast(j)), col - 1) },
            });
        }

        // flag = ctrl AND the OR of the whole prefix block, which is its
        // LAST column: the bits this stage shifted out.
        const prefix_or = prefix + amount - 1;
        try b.lin(cname("stage shifted something out", @as(u16, @intCast(k)), 0), .composed, &.{
            .{ .factors = try b.one(flag) },
            .{ .factors = try b.pair(ctrl, prefix_or), .coefficient = bld.kNegOne },
        });

        // The first stage's sticky IS its flag; after that it accumulates.
        if (k == 0) {
            try b.copy(cname("first stage sticky is its flag", 0, 0), sticky, flag);
            continue;
        }
        const prev: u16 = bases.stickyOf(cfg, @as(u16, @intCast(k - 1)));
        try b.lin(cname("sticky accumulates", @as(u16, @intCast(k)), 0), .composed, &.{
            .{ .factors = try b.one(sticky) },
            .{ .factors = try b.one(prev), .coefficient = bld.kNegOne },
            .{ .factors = try b.one(flag), .coefficient = bld.kNegOne },
            .{ .factors = try b.pair(prev, flag) },
        });
    }
}

/// The prefix columns are allocated in stage order, and stage k covers
/// 2^(amount_bits-1-k) bits, so the sizes run 8, 4, 2, 1 for four stages
/// and total 2^amount_bits - 1.
fn prefixTotal(cfg: Config) usize {
    return (@as(usize, 1) << @intCast(cfg.amount_bits)) - 1;
}

fn prefixOffset(cfg: Config, k: u16) usize {
    var prefix: usize = 0;
    var i: u16 = 0;
    while (i < k) : (i += 1) {
        prefix += @as(usize, 1) << @intCast(cfg.amount_bits - 1 - i);
    }
    return prefix;
}
