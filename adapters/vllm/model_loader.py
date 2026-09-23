"""vLLM model loader that attests weights during load (Stage 3).

The class is built by `build_loader_cls()` rather than declared at module
level so this file stays importable (and testable) without vLLM installed —
vLLM pulls in torch, and the adapter's own logic does not need it.

Usage inside vLLM:

    # one-time registration, via the plugin entry point
    from .model_loader import register
    register()                      # adds load_format "zkml_attested"

    # then select it
    vllm serve <model> --load-format zkml_attested

What it does: the loader keeps vLLM's normal weight path untouched and, as
each `(name, tensor)` pair is produced, streams the tensor's raw bytes into
the zig-zkml attestor. The resulting Merkle root is published on the model
(`model.weights_root`) and logged.

Attestation reads the *checkpoint* names (pre-`WeightsMapper`), which is what
a published root should cover; the engine's internal parameter names differ
(`q_proj` -> `qkv_proj` shard) and are not hashed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, Optional

from . import zkml

logger = logging.getLogger("zkml.vllm")

LOAD_FORMAT = "zkml_attested"

# Checkpoint files a HF repo may contain; `.safetensors` first so the
# preferred format is preferred.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")


def find_weight_files(model_path: str | Path) -> list[Path]:
    """Locate the checkpoint shard files for a local model directory."""
    root = Path(model_path)
    if root.is_file():
        return [root]
    shards = sorted(
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix in _WEIGHT_SUFFIXES and not p.name.endswith(".index.json")
    )
    if not shards:
        raise FileNotFoundError(f"no weight files under {root}")
    return shards


def safetensors_items(paths: Iterable[Path]) -> Iterator[tuple[str, bytes]]:
    """Yield (name, raw_bytes) for every tensor in every safetensors shard.

    Uses `safe_open(..., framework="pt")` — the same one-shard-at-a-time
    mmap view vLLM's own loader uses (`weight_utils.safetensors_weights_iterator`).
    Tensors are yielded unconverted, so the bytes hashed are the bytes on
    disk.
    """
    from safetensors import safe_open  # imported late: optional dependency

    for path in paths:
        with safe_open(str(path), framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                tensor = f.get_tensor(name)
                yield name, tensor_to_bytes(tensor)


def tensor_to_bytes(tensor) -> bytes:
    """Raw, little-endian-by-definition storage bytes of a torch tensor.

    A contiguous copy is taken because the attestor consumes the buffer
    immediately; the tensor itself is left untouched.
    """
    # torch tensors expose the underlying storage; .numpy() would require a
    # CPU tensor and adds a copy, so prefer contiguous() + view.
    contiguous = tensor.contiguous()
    try:
        import torch

        flat = contiguous.view(torch.uint8)
        return bytes(memoryview(flat.numpy()))
    except ModuleNotFoundError:  # numpy-only path (no torch)
        return contiguous.numpy().tobytes()


class _AttestedLoaderBase:
    """Loader mixin holding the attestation logic (vLLM-free)."""

    def attest_weights(self, items: Iterable[tuple[str, bytes]]) -> bytes:
        root = zkml.attest_items(items)
        logger.info("zkml: weights root %s", root.hex())
        return root

    def attest_safetensors(self, model_path: str | Path) -> bytes:
        return self.attest_weights(safetensors_items(find_weight_files(model_path)))


def build_loader_cls():
    """Create the vLLM loader class. Requires vLLM to be importable."""
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.model_loader.weight_utils import (
        safetensors_weights_iterator,
    )
    from vllm.model_executor.model_loader import register_model_loader

    class ZkmlAttestedLoader(_AttestedLoaderBase, BaseModelLoader):
        """DefaultModelLoader's weight path + zig-zkml attestation.

        `load_weights` delegates to the model exactly as the default loader
        does (checkpoint names, model-side `WeightsMapper`), and tees the
        same `(name, tensor)` stream into the attestor. vLLM iterates lazily,
        so the tee must not be a generator that is consumed twice: the model
        load happens first, the attestation second.
        """

        def __init__(self, load_config) -> None:
            super().__init__(load_config)
            self.weights_root: Optional[bytes] = None
            self.attested_tensors = 0

        def download_model(self, model_config) -> None:
            # The default loader downloads through the HF cache; for a local
            # path there is nothing to do. Kept abstract-satisfying.
            return None

        def _weights(self, model_config, model):
            return safetensors_weights_iterator(
                self._shard_files(model_config),
                use_tqdm_on_load=not model_config.disable_tqdm,
            )

        @staticmethod
        def _shard_files(model_config) -> list[str]:
            from vllm.model_executor.model_loader.weight_utils import (
                filter_duplicate_safetensors_files,
                get_all_model_weights_files,
            )

            return get_all_model_weights_files(model_config.model, model_config.revision)

        def load_weights(self, model, model_config) -> None:
            # 1. Normal load first — the model consumes the lazy iterator.
            with zkml.Attestor() as att:
                original = self._weights(model_config, model)

                def tee(items):
                    for name, tensor in items:
                        # Hashed only if the model actually consumes it, so
                        # skipped/filtered tensors are not silently attested.
                        att.add(name, tensor_to_bytes(tensor))
                        yield name, tensor

                model.load_weights(tee(original))
                self.weights_root = att.finish()
                self.attested_tensors = att.tensor_count

            model.weights_root = self.weights_root
            logger.info(
                "zkml: attested %d tensors, root %s",
                self.attested_tensors,
                self.weights_root.hex(),
            )

        def get_all_weights(self, model_config, model):
            """Required for vLLM's disk-based weight reload path."""
            items = self._weights(model_config, model)
            for name, tensor in items:
                yield name, tensor

    return ZkmlAttestedLoader


def register() -> None:
    """vLLM general-plugin entry point: add the `zkml_attested` load format."""
    from vllm.model_executor.model_loader import register_model_loader

    loader_cls = build_loader_cls()
    register_model_loader(LOAD_FORMAT)(loader_cls)
    logger.info("zkml: registered model loader %r", LOAD_FORMAT)
