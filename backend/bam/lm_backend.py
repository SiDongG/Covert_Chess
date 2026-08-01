"""Minimal HuggingFace LM backend shared by surface generation and ArcMark.

Both parties A and B are *instances of the same model*
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFLMBackend:
    model_name: str
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None   # auto: bf16 on CUDA, fp32 on CPU

    tokenizer: AutoTokenizer = field(init=False)
    model: AutoModelForCausalLM = field(init=False)

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dtype is None:
            self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # ---------- properties ----------
    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    # ---------- core LM ops ----------
    @torch.no_grad()
    def next_token_distribution(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Return p(x_t | x_<t) as a length-V tensor on `self.device`."""
        logits = self.model(input_ids.to(self.device)).logits[:, -1, :]
        return torch.softmax(logits, dim=-1).squeeze(0)

    # ---------- tokenizer convenience ----------
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def encode_tensor(self, text: str) -> torch.LongTensor:
        ids = self.encode(text)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)
