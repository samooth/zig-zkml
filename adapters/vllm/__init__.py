"""zig-zkml adapter for vLLM (Stage 3).

Installed as the `zkml_vllm` package; vLLM discovers the loader through the
`vllm.general_plugins` entry point (`zkml_vllm.model_loader:register`).

Importing this package does NOT require vLLM or torch — the engine imports
live inside `model_loader.build_loader_cls()` and `model_loader.register()`.
"""

from . import zkml
from .model_loader import register

__all__ = ["zkml", "register", "__version__"]

__version__ = "0.1.0"
