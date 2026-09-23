"""Witness recording for vLLM (Stage 3).

vLLM has no first-class "observe every per-layer op I/O" plugin hook in this
revision. The mechanisms that exist, and why they are not enough on their own:

* PyTorch module hooks — skipped for `CompilationMode.STOCK_TORCH_COMPILE`
  (`gpu_model_runner.py`) and not invoked for CUDA-graph replays;
* Inductor passes — see the compile-time `fx.Graph`, never runtime values;
* `set_forward_context` — batch/attention metadata, not op witnesses;
* `direct_register_custom_op` — a real runtime hook, but only for code paths
  that call the registered op.

So the adapter provides the recording side (a thin, tested wrapper over the
ABI's witness session) plus the hook pattern that actually holds: wrap the
op explicitly, either by registering a custom op or by wrapping the layer
method in eager mode. `LayerScope` owns the begin/end boundary so a wrapper
cannot desynchronize the session's state machine.

    with LayerScope(witness, layer=7):
        witness.record_in_open_layer(zkml.OP_GEMM_C, out_bytes)

Payloads are the op's raw I/O bytes; the proof layer (F2) will constrain the
arithmetic, so what matters here is that the recorded bytes are exactly the
bytes the kernel produced.
"""

from __future__ import annotations

import threading
from typing import Optional

from . import zkml


def tensor_bytes(tensor) -> bytes:
    """Raw storage bytes of a torch tensor, without changing the tensor."""
    return tensor.contiguous().view(__import__("torch").uint8).numpy().tobytes()


class LayerScope:
    """Context manager owning one `begin_layer` / `end_layer` boundary."""

    def __init__(self, witness: zkml.Witness, layer: int) -> None:
        self.witness = witness
        self.layer = layer

    def __enter__(self) -> "LayerScope":
        self.witness.begin_layer(self.layer)
        return self

    def __exit__(self, *exc) -> None:
        self.witness.end_layer()


class WitnessRecorder:
    """Thread-safe facade over one witness session.

    `record_op` is safe from several threads while a layer is open (the core
    serializes per-slot writes); begin/end/finalize are expected to be driven
    by one thread, and the recorder asserts that.
    """

    def __init__(self) -> None:
        self._witness = zkml.Witness()
        self._boundary_lock = threading.Lock()
        self._boundary_thread: Optional[int] = None

    @property
    def open_layer(self) -> Optional[int]:
        return self._witness.open_layer

    def _claim_boundary(self) -> None:
        me = threading.get_ident()
        if self._boundary_thread is None:
            self._boundary_thread = me
        elif self._boundary_thread != me:
            raise RuntimeError(
                "begin/end_layer must be driven by a single thread "
                f"(owned by {self._boundary_thread}, called from {me})"
            )

    def layer(self, layer: int) -> LayerScope:
        self._claim_boundary()
        return LayerScope(self._witness, layer)

    def record_op(
        self,
        op: int,
        payload: bytes,
        expert: int = 0,
        rank: int = 0,
    ) -> None:
        """Record into the currently open layer (any thread)."""
        if self._witness.open_layer is None:
            raise zkml.ZkmlError(zkml.INVALID_ARGUMENT, "record with no open layer")
        self._witness.record_in_open_layer(op, payload, expert, rank)

    def record_tensor(self, op: int, tensor, expert: int = 0, rank: int = 0) -> None:
        self.record_op(op, tensor_bytes(tensor), expert, rank)

    def finalize(self, statement_hash: bytes) -> bytes:
        self._claim_boundary()
        return self._witness.finalize(statement_hash)

    def close(self) -> None:
        self._witness.close()

    def __enter__(self) -> "WitnessRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def register_custom_op(name: str, op_func, mutates_args=None, fake_impl=None):
    """Register a vLLM custom op that records its output into the session.

    The wrapped function must call `WITNESS.record_tensor(...)` on the value
    it returns; the `fake_impl` is what torch.compile traces, so the
    recording call has to sit outside the compiled region (a side effect in
    the fake impl is never executed).
    """
    from vllm.utils.torch_utils import direct_register_custom_op

    def wrapped(*args, **kwargs):
        out = op_func(*args, **kwargs)
        active = _ACTIVE_RECORDER
        if active is not None:
            active.record_tensor(_OP_OF.get(name, zkml.OP_OTHER), out)
        return out

    direct_register_custom_op(
        name,
        wrapped,
        mutates_args=mutates_args,
        fake_impl=fake_impl,
    )
    return wrapped


# Module-level state for the custom-op wrapper: the process runs one model,
# so a single active recorder is enough (and matches the one-session-per-
# inference model). Guarded because ops may fire from several threads.
_ACTIVE_RECORDER: Optional[WitnessRecorder] = None
_OP_OF: dict[str, int] = {}


def set_active_recorder(recorder: Optional[WitnessRecorder]) -> None:
    global _ACTIVE_RECORDER
    _ACTIVE_RECORDER = recorder


def set_op(name: str, op_ordinal: int) -> None:
    _OP_OF[name] = op_ordinal


def register_oot_op(name: str, op_ordinal: int):
    """Decorator form: mark which ZKML op an out-of-tree CustomOp records.

    Mirrors vLLM's `CustomOp.register_oot` pattern for target layers.
    """
    set_op(name, op_ordinal)

    def decorator(cls):
        cls.zkml_op = op_ordinal  # type: ignore[attr-defined]
        return cls

    return decorator
