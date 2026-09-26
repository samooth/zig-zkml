//! L2 zkML gadgets — quantized neural-network operators verifiable via AIR.
//!
//! Each gadget emits an `air.Fragment` (columns + constraints + lookups)
//! that the L3 compiler instantiates into a full AirGraph. The witness
//! comes from the native kernel's recorded trace (§3 dual-path).
//!
//! Layout (docs/BLUE_PRINT.md §7.1):
//!   gemm/      — matrix multiply (AIR chunk-16, v1)
//!   quant/     — dequant/requant, fp8/mxfp8 reencode, GGML scales
//!   nonlin/    — SiLU/GELU/softmax-step lookups (LogUp)
//!   norm/      — rmsnorm/layernorm
//!   routing/   — top-k / group-top2 (DeepSeek-V3)

const std = @import("std");
const tensor = @import("../tensor/root.zig");
const air = @import("../air/root.zig");

pub const gemm = @import("gemm/root.zig");
pub const quant = @import("quant/root.zig");
pub const nonlin = @import("nonlin/root.zig");
pub const norm = @import("norm/root.zig");
pub const routing = @import("routing/root.zig");
