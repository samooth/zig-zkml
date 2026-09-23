"""Gate tests for the vLLM adapter (Stage 3).

Runs two ways, because the adapter's whole point is to have no Python
dependencies beyond the standard library:

    pytest adapters/vllm/test_zkml.py     # when pytest is available
    python3 adapters/vllm/test_zkml.py     # zero-dependency fallback runner

The safetensors path is exercised when `safetensors` is importable and
skipped otherwise; the ctypes binding, the attestation algebra and the
witness session are always exercised, against the real `libzkml.so`.

The cross-check is the important one: the root computed through ctypes must
equal the root recomputed by the INDEPENDENT Python auditor
(`tools/verify_weights.py`), which shares no code with the library.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load_package():
    """Import the adapter directory under its installed name, `zkml_vllm`.

    The source directory is called `vllm`; loading it as `zkml_vllm` mirrors
    the installed layout (and the relative imports inside the modules)
    without ever putting a `vllm` package on sys.path.
    """
    if "zkml_vllm" in sys.modules:
        return sys.modules["zkml_vllm"]
    spec = importlib.util.spec_from_file_location(
        "zkml_vllm",
        HERE / "__init__.py",
        submodule_search_locations=[str(HERE)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["zkml_vllm"] = module
    spec.loader.exec_module(module)
    return module


_load_package()

import zkml_vllm  # noqa: E402
from zkml_vllm import witness as _witness  # noqa: E402
from zkml_vllm import zkml  # noqa: E402
from zkml_vllm.zkml import ZkmlError  # noqa: E402

AUDITOR = REPO / "tools" / "verify_weights.py"

FIXTURE = [
    ("model.embed_tokens.weight", b"EMBEDDING-BYTES-0123456789"),
    ("model.layers.0.self_attn.q_proj.weight", b"QW"),
    ("model.layers.0.mlp.down_proj.weight", b"DOWN0-PADDED-XXXX"),
    ("model.layers.7.mlp.up_proj.weight", b"UP7"),
    ("model.norm.weight", b"NORM"),
]


def _lib():
    return zkml.load_library()


# --- attestation --------------------------------------------------------------


def test_attest_root_is_deterministic():
    first = zkml.attest_items(FIXTURE)
    second = zkml.attest_items(FIXTURE)
    assert first == second, "attestation is not deterministic"
    assert len(first) == 32


def test_attest_root_is_order_independent():
    forward = zkml.attest_items(FIXTURE)
    backward = zkml.attest_items(list(reversed(FIXTURE)))
    assert forward == backward, "root must not depend on iteration order"


def test_one_flipped_byte_changes_the_root():
    baseline = zkml.attest_items(FIXTURE)
    corrupted = [(n, (bytes([d[0] ^ 0xFF]) + d[1:]) if i == 1 else d)
                 for i, (n, d) in enumerate(FIXTURE)]
    after = zkml.attest_items(corrupted)
    assert baseline != after, "a corrupted tensor must change the root"


def test_duplicate_names_rejected():
    dup = [("same.weight", b"A"), ("same.weight", b"B")]
    try:
        zkml.attest_items(dup)
    except ZkmlError as exc:
        assert exc.code == zkml.DUPLICATE_NAME, f"unexpected status {exc.code}"
    else:
        raise AssertionError("duplicate tensor names were not rejected")


def test_empty_attestation_rejected():
    try:
        zkml.attest_items([])
    except ZkmlError as exc:
        assert exc.code == zkml.INVALID_ARGUMENT
    else:
        raise AssertionError("an empty attestation must be rejected")


def test_proof_round_trip():
    with zkml.Attestor() as att:
        for name, data in FIXTURE:
            att.add(name, data)
        root = att.finish()
        wire = att.proof("model.norm.weight")
    assert wire[:4] == b"ZKMP", "unexpected proof magic"
    assert zkml.verify_proof(wire, root) is True
    bad = bytes([root[0] ^ 0xFF]) + root[1:]
    assert zkml.verify_proof(wire, bad) is False, "wrong root must be rejected"


def test_transcript_seed_deterministic():
    a = zkml.transcript_seed(b"sampling context v1")
    b = zkml.transcript_seed(b"sampling context v1")
    c = zkml.transcript_seed(b"sampling context v2")
    assert a == b and a != c and len(a) == 32


# --- independent cross-check --------------------------------------------------


def test_root_matches_independent_auditor():
    """The ctypes root must equal the root recomputed by tools/verify_weights.py."""
    root = zkml.attest_items(FIXTURE)
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.json"
        manifest.write_text(
            json.dumps([{"name": n, "data_b64": _b64(d)} for n, d in FIXTURE])
        )
        proc = subprocess.run(
            [sys.executable, str(AUDITOR), "manifest",
             "--root", root.hex(), str(manifest)],
            capture_output=True, text=True,
        )
    assert proc.returncode == 0, f"auditor rejected the root:\n{proc.stdout}{proc.stderr}"
    assert "OK" in proc.stdout


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


# --- witness ------------------------------------------------------------------


def test_witness_lifecycle_and_determinism():
    stmt = bytes([0x5A]) * 32

    def run():
        with zkml.Witness() as w:
            for layer in range(3):
                w.begin_layer(layer)
                w.record_in_open_layer(zkml.OP_GEMM_C, f"OUT-L{layer}".encode())
                w.record_in_open_layer(zkml.OP_DEQUANT, b"Q")
                w.end_layer()
            return w.finalize(stmt)

    first, second = run(), run()
    assert first == second, "witness trace is not deterministic"
    assert len(first) == 32


def test_witness_state_machine_rejects_violations():
    stmt = bytes([1]) * 32
    with zkml.Witness() as w:
        w.begin_layer(0)
        try:
            w.begin_layer(1)
        except ZkmlError as exc:
            assert exc.code == zkml.INVALID_ARGUMENT
        else:
            raise AssertionError("double begin_layer accepted")
        try:
            w.finalize(stmt)
        except ZkmlError as exc:
            assert exc.code == zkml.INVALID_ARGUMENT
        else:
            raise AssertionError("finalize with an open layer accepted")
        w.end_layer()
        try:
            w.record_in_open_layer(zkml.OP_OTHER, b"x")
        except ZkmlError as exc:
            assert exc.code == zkml.INVALID_ARGUMENT
        else:
            raise AssertionError("record with no open layer accepted")
        trace = w.finalize(stmt)
        try:
            w.begin_layer(1)
        except ZkmlError as exc:
            assert exc.code == zkml.INVALID_ARGUMENT
        else:
            raise AssertionError("session not frozen after finalize")
    assert len(trace) == 32


def test_witness_slot_key_layout():
    import ctypes

    assert ctypes.sizeof(zkml.SlotKey) == 8, "SlotKey must be 8 bytes (C layout)"
    offsets = {
        "layer": zkml.SlotKey.layer.offset,
        "expert": zkml.SlotKey.expert.offset,
        "op": zkml.SlotKey.op.offset,
        "rank": zkml.SlotKey.rank.offset,
    }
    assert offsets == {"layer": 0, "expert": 4, "op": 6, "rank": 7}, offsets


def test_witness_recorder_thread_safety():
    """Concurrent record_op calls on one open layer must be accepted."""
    import threading

    stmt = bytes([0x33]) * 32
    with _witness.WitnessRecorder() as rec:
        rec._claim_boundary()
        with rec.layer(0):
            errors: list[BaseException] = []

            def worker(tag: int) -> None:
                try:
                    for i in range(16):
                        rec.record_op(zkml.OP_GEMM_C, f"{tag}:{i}".encode())
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert not errors, f"concurrent record failed: {errors}"
        trace = rec.finalize(stmt)
    assert len(trace) == 32


# --- adapter surface (vLLM-free parts) ---------------------------------------


def test_model_loader_module_imports_without_vllm():
    from zkml_vllm import model_loader

    assert model_loader.LOAD_FORMAT == "zkml_attested"
    # The class is only built on demand: importing must not need vLLM.
    try:
        model_loader.build_loader_cls()
    except ImportError:
        pass  # expected here: vLLM is not installed in this environment
    else:
        raise AssertionError("build_loader_cls should require vLLM to be importable")


def test_find_weight_files():
    from zkml_vllm import model_loader

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "model-00001-of-00002.safetensors").write_bytes(b"")
        (root / "model-00002-of-00002.safetensors").write_bytes(b"")
        (root / "model.safetensors.index.json").write_text("{}")
        (root / "config.json").write_text("{}")
        found = model_loader.find_weight_files(root)
        names = [p.name for p in found]
        assert names == [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ], names
        try:
            model_loader.find_weight_files(root / "nope")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing model dir must raise")


def test_safetensors_attestation_if_available():
    """Exercise the real shard path when safetensors is installed."""
    from zkml_vllm import model_loader

    # Both failure modes mean "the optional dependency is unusable here",
    # not "the code is wrong": ImportError when it is absent, OSError from
    # the dynamic loader when the install is incomplete (a torch missing
    # its libtorch_global_deps.so raises exactly that).
    try:
        import safetensors.torch
        import torch
    except (ImportError, OSError) as exc:
        print(f"    (skipped: safetensors/torch unusable: {exc})")
        return

    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "model.safetensors"
        safetensors.torch.save_file(
            {"a.weight": torch.zeros(4, dtype=torch.float32),
             "b.weight": torch.ones(4, dtype=torch.float32)},
            str(shard),
        )
        loader = model_loader._AttestedLoaderBase()
        root = loader.attest_safetensors(tmp)
    assert len(root) == 32


# --- runner -------------------------------------------------------------------


def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
