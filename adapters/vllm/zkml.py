"""ctypes binding to zig-zkml's shared library (vLLM adapter, Stage 3).

Thin, dependency-free wrapper over ``libzkml.so``. Every entry point declares
``argtypes``/``restype`` so argument marshalling is checked by ctypes rather
than by hope, and the opaque ``ZKML_Attestor`` / ``ZKML_Witness`` handles stay
opaque (ctypes ``c_void_p``).

Library discovery order:

1. explicit ``path`` argument;
2. ``$ZKML_LIB``;
3. ``<repo>/zig-out/lib/libzkml.so`` (run ``zig build`` to produce it).

The attestation contract is the engine-wide one (see
``include/zkml_engine.h``): tensor names are canonical, the root is
order-independent, and ``data`` is read once and never retained by the
library — so a tensor may be passed straight from an mmap-backed loader.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable, Optional

__all__ = [
    "ZkmlError",
    "SlotKey",
    "Attestor",
    "Witness",
    "load_library",
    "library_path",
    "attest_items",
    "verify_proof",
    "transcript_seed",
    "ABI_VERSION",
]

# Mirrors ZKML_ABI_VERSION in include/zkml_c.h.
ABI_VERSION = 2

# Status codes (include/zkml_c.h).
OK = 0
OUT_OF_MEMORY = -1
INVALID_ARGUMENT = -2
BAD_PROOF = -3
LEAF_NOT_FOUND = -4
DUPLICATE_NAME = -5

_STATUS_TEXT = {
    OUT_OF_MEMORY: "out of memory",
    INVALID_ARGUMENT: "invalid argument",
    BAD_PROOF: "bad proof",
    LEAF_NOT_FOUND: "leaf not found",
    DUPLICATE_NAME: "duplicate tensor name",
}

# Op ordinals (include/zkml_c.h); mirrors trace.Op in libs/trace/root.zig.
OP_GEMM_A = 0
OP_GEMM_B = 1
OP_GEMM_C = 2
OP_DEQUANT = 3
OP_REQUANT = 4
OP_SWIGLU = 5
OP_LAYERNORM = 6
OP_RMSNORM = 7
OP_ROUTING_TOPK = 8
OP_ROUTING_GATE = 9
OP_OTHER = 10


class ZkmlError(RuntimeError):
    """A non-OK status returned by the library."""

    def __init__(self, code: int, what: str) -> None:
        self.code = code
        self.what = what
        detail = _STATUS_TEXT.get(code, f"status {code}")
        super().__init__(f"{what}: {detail} ({code})")


def _check(code: int, what: str) -> None:
    if code != OK:
        raise ZkmlError(code, what)


class SlotKey(ctypes.Structure):
    """Flat 8-byte witness slot key (layer u32, expert u16, op u8, rank u8)."""

    _fields_ = [
        ("layer", ctypes.c_uint32),
        ("expert", ctypes.c_uint16),
        ("op", ctypes.c_uint8),
        ("rank", ctypes.c_uint8),
    ]


def library_path(path: Optional[os.PathLike | str] = None) -> Path:
    """Resolve the libzkml shared object without loading it."""
    if path is not None:
        return Path(path)
    env = os.environ.get("ZKML_LIB")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    # adapters/vllm/ -> repo root -> zig-out/lib
    return here.parents[2] / "zig-out" / "lib" / "libzkml.so"


_LIB = None


def load_library(path: Optional[os.PathLike | str] = None) -> ctypes.CDLL:
    """Load (once per process) and configure the ctypes signatures."""
    global _LIB
    if _LIB is not None and path is None:
        return _LIB

    so = library_path(path)
    if not so.is_file():
        raise ZkmlError(
            INVALID_ARGUMENT,
            f"libzkml.so not found at {so} — run `zig build` in the zig-zkml repo "
            f"or set $ZKML_LIB",
        )
    lib = ctypes.CDLL(str(so))

    v = ctypes.c_void_p
    u8p = ctypes.POINTER(ctypes.c_ubyte)
    sz = ctypes.c_size_t
    i32 = ctypes.c_int32

    lib.zkml_allocator_process.restype = v
    lib.zkml_allocator_process.argtypes = []

    lib.zkml_attestor_create.restype = v
    lib.zkml_attestor_create.argtypes = [v]

    lib.zkml_attestor_add.restype = i32
    lib.zkml_attestor_add.argtypes = [v, ctypes.c_char_p, sz, v, sz]

    lib.zkml_attestor_finish.restype = i32
    lib.zkml_attestor_finish.argtypes = [v]

    lib.zkml_attestor_root.restype = i32
    lib.zkml_attestor_root.argtypes = [v, u8p]

    lib.zkml_attestor_proof.restype = i32
    lib.zkml_attestor_proof.argtypes = [v, ctypes.c_char_p, sz, ctypes.POINTER(u8p), ctypes.POINTER(sz)]

    lib.zkml_attestor_free_proof.restype = None
    lib.zkml_attestor_free_proof.argtypes = [v, v, sz]

    lib.zkml_attestor_destroy.restype = None
    lib.zkml_attestor_destroy.argtypes = [v]

    lib.zkml_proof_verify.restype = i32
    lib.zkml_proof_verify.argtypes = [v, v, sz, v]

    lib.zkml_transcript_seed.restype = i32
    lib.zkml_transcript_seed.argtypes = [v, sz, u8p]

    lib.zkml_witness_session_create.restype = v
    lib.zkml_witness_session_create.argtypes = [v]

    lib.zkml_witness_begin_layer.restype = i32
    lib.zkml_witness_begin_layer.argtypes = [v, ctypes.c_uint32]

    lib.zkml_witness_record_op.restype = i32
    lib.zkml_witness_record_op.argtypes = [v, ctypes.POINTER(SlotKey), v, sz]

    lib.zkml_witness_end_layer.restype = i32
    lib.zkml_witness_end_layer.argtypes = [v]

    lib.zkml_witness_finalize.restype = i32
    lib.zkml_witness_finalize.argtypes = [v, v, u8p]

    lib.zkml_witness_session_destroy.restype = None
    lib.zkml_witness_session_destroy.argtypes = [v]

    if path is None:
        _LIB = lib
    return lib


class Attestor:
    """Streaming weights attestation.

    Typical use::

        with Attestor() as att:
            for name, tensor in loader:
                att.add(name, tensor_bytes)
            att.finish()
            root = att.root
    """

    def __init__(self, lib: Optional[ctypes.CDLL] = None) -> None:
        self._lib = lib if lib is not None else load_library()
        self._allocator = self._lib.zkml_allocator_process()
        self._handle = self._lib.zkml_attestor_create(self._allocator)
        if not self._handle:
            raise ZkmlError(OUT_OF_MEMORY, "attestor_create")
        self._finished = False
        self._count = 0

    def __enter__(self) -> "Attestor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add(self, name: str, data: bytes) -> None:
        """Hash one tensor. `name` must be canonical and unique."""
        if isinstance(data, memoryview):
            data = data.tobytes()
        elif not isinstance(data, bytes):
            data = bytes(data)
        name_bytes = name.encode("utf-8") if isinstance(name, str) else bytes(name)
        _check(
            self._lib.zkml_attestor_add(
                self._handle, name_bytes, len(name_bytes), data, len(data)
            ),
            f"attestor_add({name!r})",
        )
        self._count += 1

    def finish(self) -> bytes:
        """Freeze the tree and return the 32-byte Merkle root."""
        _check(self._lib.zkml_attestor_finish(self._handle), "attestor_finish")
        self._finished = True
        return self.root

    @property
    def tensor_count(self) -> int:
        return self._count

    @property
    def root(self) -> bytes:
        out = (ctypes.c_ubyte * 32)()
        _check(self._lib.zkml_attestor_root(self._handle, out), "attestor_root")
        return bytes(out)

    def proof(self, name: str) -> bytes:
        """Wire-format ("ZKMP" v1) inclusion proof for one tensor."""
        if not self._finished:
            raise ZkmlError(INVALID_ARGUMENT, "proof before finish")
        name_bytes = name.encode("utf-8")
        ptr = ctypes.POINTER(ctypes.c_ubyte)()
        length = ctypes.c_size_t(0)
        _check(
            self._lib.zkml_attestor_proof(
                self._handle, name_bytes, len(name_bytes), ctypes.byref(ptr), ctypes.byref(length)
            ),
            f"attestor_proof({name!r})",
        )
        return ctypes.string_at(ptr, length.value)

    def close(self) -> None:
        if self._handle:
            self._lib.zkml_attestor_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - safety net
        try:
            self.close()
        except Exception:
            pass


class Witness:
    """Recorded-inference witness session (ABI v2).

    One layer open at a time; `record_op` copies its payload and is safe to
    call from several threads while a layer is open.
    """

    def __init__(self, lib: Optional[ctypes.CDLL] = None) -> None:
        self._lib = lib if lib is not None else load_library()
        self._allocator = self._lib.zkml_allocator_process()
        self._handle = self._lib.zkml_witness_session_create(self._allocator)
        if not self._handle:
            raise ZkmlError(OUT_OF_MEMORY, "witness_session_create")
        self.open_layer: Optional[int] = None

    def __enter__(self) -> "Witness":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def begin_layer(self, layer: int) -> None:
        _check(self._lib.zkml_witness_begin_layer(self._handle, layer), "witness_begin_layer")
        self.open_layer = layer

    def end_layer(self) -> None:
        _check(self._lib.zkml_witness_end_layer(self._handle), "witness_end_layer")
        self.open_layer = None

    def record_op(
        self,
        layer: int,
        op: int,
        payload: bytes,
        expert: int = 0,
        rank: int = 0,
    ) -> None:
        key = SlotKey(layer=layer, expert=expert, op=op, rank=rank)
        data = bytes(payload)
        _check(
            self._lib.zkml_witness_record_op(
                self._handle, ctypes.byref(key), data, len(data)
            ),
            f"witness_record_op(layer={layer}, op={op})",
        )

    def record_in_open_layer(self, op: int, payload: bytes, expert: int = 0, rank: int = 0) -> None:
        if self.open_layer is None:
            raise ZkmlError(INVALID_ARGUMENT, "record with no open layer")
        self.record_op(self.open_layer, op, payload, expert, rank)

    def finalize(self, statement_hash: bytes) -> bytes:
        """Absorb the trace in canonical order, bound to `statement_hash`."""
        if len(statement_hash) != 32:
            raise ZkmlError(INVALID_ARGUMENT, "statement_hash must be 32 bytes")
        out = (ctypes.c_ubyte * 32)()
        _check(
            self._lib.zkml_witness_finalize(self._handle, statement_hash, out),
            "witness_finalize",
        )
        return bytes(out)

    def close(self) -> None:
        if self._handle:
            self._lib.zkml_witness_session_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - safety net
        try:
            self.close()
        except Exception:
            pass


def attest_items(
    items: Iterable[tuple[str, bytes]],
    lib: Optional[ctypes.CDLL] = None,
) -> bytes:
    """Attest an iterable of (name, bytes) and return the root.

    The iteration order does not matter: leaves are name-sorted before the
    tree is folded, so a hash-map or multi-shard loader yields the same root.
    """
    with Attestor(lib) as att:
        for name, data in items:
            att.add(name, data)
        return att.finish()


def verify_proof(
    proof: bytes,
    expected_root: bytes,
    lib: Optional[ctypes.CDLL] = None,
) -> bool:
    """Standalone proof verification: no model, no tree, no runtime."""
    lib = lib if lib is not None else load_library()
    allocator = lib.zkml_allocator_process()
    if len(expected_root) != 32:
        raise ZkmlError(INVALID_ARGUMENT, "expected_root must be 32 bytes")
    return lib.zkml_proof_verify(allocator, proof, len(proof), expected_root) == OK


def transcript_seed(context: bytes, lib: Optional[ctypes.CDLL] = None) -> bytes:
    """Deterministic 32-byte sampling seed for a context string."""
    lib = lib if lib is not None else load_library()
    out = (ctypes.c_ubyte * 32)()
    _check(
        lib.zkml_transcript_seed(context, len(context), out),
        "transcript_seed",
    )
    return bytes(out)
