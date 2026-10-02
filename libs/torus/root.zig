//! The 2-adic torus toolkit, minus the FRI.
//!
//! When `libs/fri` was split, this is what stayed: the quadratic extension over
//! M61, the order-2^k torus subgroup, and the FFT that turns coefficients into
//! evaluations on that subgroup. The FRI itself — the prover and the verifier —
//! went to the pin, because three gates measured our composition equal to the
//! pin's and the field argument that justified owning it expired in a pin bump
//! (`zig-algebra` v0.6.0 ships M61 with the same 2^61 − 1).
//!
//! These three stayed for reasons the pin cannot supply:
//!
//!   - **fp2.zig** is the field extension the whole AIR layer is written
//!     against. `fri_conv_diff` measured it to be the identity with the pin's
//!     `QuadraticExtension`, and the pin's FRI is generic over its field, so
//!     nothing needed to move.
//!   - **fft.zig** has no substitute. The pin's `poly` package is dense
//!     univariate arithmetic with a comptime-known degree, not a transform over
//!     a 2-adic domain — and the pin's FRI takes its evaluations from the
//!     caller, so somebody has to produce them.
//!   - **domain.zig** is the Merkle-friendly view of the subgroup.
//!
//! So this file is a barrel, not a layer: it names the three, and holds the one
//! type they no longer have a parent for — the FRI config shape our callers
//! pass. The pin's config has an extra `log_initial_degree` that is derived,
//! not chosen, and `toPin` is where the derivation lives so that it happens in
//! one place.

const std = @import("std");

pub const fp2 = @import("fp2.zig");
pub const domain_lib = @import("domain.zig");
pub const fft = @import("fft.zig");

pub const Fp2 = fp2.Fp2;
pub const Goldilocks = fp2.Goldilocks;
pub const Domain = domain_lib.Domain;
pub const max_log_domain = domain_lib.max_log_domain;

/// The FRI parameters a caller chooses.
///
/// Kept in our shape rather than the pin's, because adding
/// `log_initial_degree` to every caller would invite the one thing that field
/// must never be: hand-set. It is derived, in `toPin`.
pub const Config = struct {
    /// log2 of the initial domain size (order of H_k). The committed
    /// evaluations are the polynomial on the full domain.
    log_domain: u6,
    /// log2 of the last FRI layer's domain size. The residual polynomial is
    /// evaluated on this domain.
    log_final: u6,
    /// log2 of the residual degree bound (d).
    log_residual_degree: u6,
    /// Number of random positions checked.
    num_queries: usize,

    pub const Error = error{InvalidParameters};

    /// Reject the shapes the FRI cannot honour, before any allocation.
    ///
    /// Same three conditions the prover needs, in the order it needs them:
    /// the final domain must be strictly inside the initial one (otherwise
    /// there are no folding rounds), the residual bound must fit in the final
    /// domain, and the bound must be strictly below it — rate 1 leaves no
    /// distance for a query to catch a cheater.
    pub fn validate(self: Config) Error!usize {
        if (self.log_domain > max_log_domain) return Error.InvalidParameters;
        if (self.log_final >= self.log_domain) return Error.InvalidParameters;
        if (self.log_residual_degree > self.log_final) return Error.InvalidParameters;
        if (self.log_residual_degree == self.log_final) return Error.InvalidParameters;
        return self.log_domain - self.log_final;
    }
};

/// Into the pin's config.
///
/// The pin carries `log_initial_degree`, and it is not a free parameter: its own
/// `validate` requires `log_initial_degree - rounds == log_residual_degree`.
/// Deriving it here means the two cannot drift apart, which a hand-set field
/// would allow silently.
pub fn toPin(comptime Pin: type, c: Config) Pin.Config {
    return .{
        .log_domain = c.log_domain,
        .log_initial_degree = c.log_residual_degree + (c.log_domain - c.log_final),
        .log_final = c.log_final,
        .log_residual_degree = c.log_residual_degree,
        .num_queries = c.num_queries,
    };
}
