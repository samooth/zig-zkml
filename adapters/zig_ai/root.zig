//! zig-ai adapter for zig-zkml (Stage 2) — module root.
//!
//! The engine imports this single module and receives the compiled pieces:
//! `gguf_attestation` (weights → Merkle root) and `witness_hooks`
//! (per-layer metrics → witness ABI). `integration.zig` is deliberately not
//! re-exported here: it needs the engine's own `gguf` module and is compiled
//! only inside zig-ai's build graph (see README.md).

pub const gguf_attestation = @import("gguf_attestation.zig");
pub const witness_hooks = @import("witness_hooks.zig");

pub const TensorSource = gguf_attestation.TensorSource;
pub const Attestation = gguf_attestation.Attestation;
pub const attestSource = gguf_attestation.attestSource;
pub const ProofSession = gguf_attestation.ProofSession;
pub const parseTensorName = gguf_attestation.parseTensorName;
pub const Role = gguf_attestation.Role;
pub const Category = gguf_attestation.Category;
pub const roleCategory = gguf_attestation.roleCategory;
pub const ParsedName = gguf_attestation.ParsedName;

pub const Recorder = witness_hooks.Recorder;
pub const SlotKey = witness_hooks.SlotKey;
pub const LayerMetricsView = witness_hooks.LayerMetricsView;
pub const install = witness_hooks.install;
pub const uninstall = witness_hooks.uninstall;
pub const onLayer = witness_hooks.onLayer;
pub const layoutMatches = witness_hooks.layoutMatches;
