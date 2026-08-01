"""Weight-sharing extension for HFLMBackend.

Load the model weights once into VRAM; hand out multiple HFLMBackend-compatible
objects that share the same nn.Module.  For an 8B bf16 model this halves VRAM
from ~32 GB (two full instances) to ~16 GB.
"""
from __future__ import annotations
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bam.lm_backend import HFLMBackend          # ← bam package


class SharedModelPool:
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = dtype or (
            torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        )
        print(f"[SharedModelPool] Loading {model_name} → {self.device} ({self.dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._model = (
            AutoModelForCausalLM
            .from_pretrained(model_name, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        if self.device.startswith("cuda"):
            used  = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[SharedModelPool] Ready. VRAM: {used:.1f}/{total:.1f} GB")
        else:
            print("[SharedModelPool] Ready (CPU).")

    def make_backend(self) -> HFLMBackend:
        """Return a backend that shares this pool's model weights."""
        backend = object.__new__(HFLMBackend)
        backend.model_name = self.model_name
        backend.device     = self.device
        backend.dtype      = self.dtype
        backend.tokenizer  = self.tokenizer
        backend.model      = self._model
        return backend

    def vram_gb(self) -> float:
        if not self.device.startswith("cuda"):
            return 0.0
        return torch.cuda.memory_allocated() / 1e9
