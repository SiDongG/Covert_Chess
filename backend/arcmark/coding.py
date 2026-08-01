"""Channel coding for ArcMark multi-bit watermarking.

This module provides the abstract :class:`ChannelCode` interface and a
concrete :class:`RandomLinearCode` implementation.  A channel code maps
*M* messages to codewords of length *n* over an alphabet
{0, ..., alphabet_size - 1}.

**ArcMark pipeline integration:**

- **Encoder side:** ``code = RandomLinearCode.build(...)``, then
  ``codeword = code.encode(msg_idx)`` feeds into
  :meth:`~arcmark.processor.ArcMarkLogitsProcessor.set_trial`.
- **Decoder side:** ``code = RandomLinearCode.build(...)`` (same params),
  then ``code.codebook`` feeds into
  :func:`~arcmark.message_decoder.score_all_messages`.

Both sides must construct the code with **identical parameters and seed**
so the codebooks match.

**Extending to new codes:** subclass :class:`ChannelCode`, implement the
five abstract members (``encode``, ``codebook``, ``num_messages``,
``codeword_length``, ``alphabet_size``), and the base-class utilities
(``int_to_bits``, ``bits_to_int``, ``encode_bits``, ``k_bits``) are
inherited automatically.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor

__all__ = [
    "ChannelCode",
    "RandomLinearCode",
]


# ═══════════════════════════════════════════════════════════════════════════
# Abstract base class
# ═══════════════════════════════════════════════════════════════════════════


class ChannelCode(ABC):
    """Abstract interface for channel codes used in ArcMark watermarking.

    A channel code maps *M* messages (indexed 0 .. M-1) to codewords of
    length *n* over a finite alphabet {0, ..., p-1}.

    Subclasses must implement:

    - :meth:`encode` — map a message index to its codeword.
    - :attr:`codebook` — all *M* codewords as a ``(M, n)`` tensor.
    - :attr:`num_messages` — total number of messages *M*.
    - :attr:`codeword_length` — codeword length *n*.
    - :attr:`alphabet_size` — alphabet size *p*.

    The base class provides concrete utilities for bit-level conversion:
    :attr:`k_bits`, :meth:`int_to_bits`, :meth:`bits_to_int`, and
    :meth:`encode_bits`.
    """

    # ── Abstract members ──────────────────────────────────────────────────

    @abstractmethod
    def encode(self, message_idx: int) -> Tensor:
        """Return the codeword for message *message_idx*.

        Args:
            message_idx: Integer in {0, ..., num_messages - 1}.

        Returns:
            LongTensor of shape ``(codeword_length,)`` with entries in
            {0, ..., alphabet_size - 1}.
        """

    @property
    @abstractmethod
    def codebook(self) -> Tensor:
        """All codewords, shape ``(num_messages, codeword_length)``, dtype long."""

    @property
    @abstractmethod
    def num_messages(self) -> int:
        """Number of distinct messages *M*."""

    @property
    @abstractmethod
    def codeword_length(self) -> int:
        """Codeword length *n* (= number of watermarked tokens)."""

    @property
    @abstractmethod
    def alphabet_size(self) -> int:
        """Code alphabet size *p* (symbols in {0, ..., p-1})."""

    # ── Concrete utilities ────────────────────────────────────────────────

    @property
    def k_bits(self) -> int:
        """Number of bits needed to represent a message index.

        Equals ``ceil(log2(max(2, M)))`` so that at least 1 bit is used.
        """
        return int(math.ceil(math.log2(max(2, self.num_messages))))

    def int_to_bits(self, message_idx: int) -> Tensor:
        """Convert a message index to a binary bit vector (MSB first).

        Args:
            message_idx: Integer in {0, ..., num_messages - 1}.

        Returns:
            ``uint8`` tensor of shape ``(k_bits,)`` with values 0 or 1.

        Raises:
            IndexError: If *message_idx* is out of range.
        """
        M = self.num_messages
        if not (0 <= message_idx < M):
            raise IndexError(
                f"message_idx={message_idx} out of range [0, {M})"
            )
        k = self.k_bits
        # Convert int to binary string, zero-pad to k digits, map to tensor
        bits = []
        val = message_idx
        for _ in range(k):
            bits.append(val & 1)
            val >>= 1
        # bits is LSB-first; reverse for MSB-first
        bits.reverse()
        return torch.tensor(bits, dtype=torch.uint8)

    @staticmethod
    def bits_to_int(bits: Tensor) -> int:
        """Convert a binary bit vector (MSB first) to an integer.

        Args:
            bits: 1-D tensor of 0s and 1s (any integer dtype).

        Returns:
            Non-negative integer.
        """
        val = 0
        for b in bits.tolist():
            val = (val << 1) | int(b)
        return val

    def encode_bits(self, bits: Tensor) -> Tensor:
        """Convenience: convert a bit vector to a message index, then encode.

        Args:
            bits: 1-D tensor of 0s and 1s, length ``k_bits``.

        Returns:
            Codeword tensor, same as :meth:`encode`.
        """
        message_idx = self.bits_to_int(bits)
        return self.encode(message_idx)


# ═══════════════════════════════════════════════════════════════════════════
# Random linear code over Z_p
# ═══════════════════════════════════════════════════════════════════════════


class RandomLinearCode(ChannelCode):
    """Random linear code over Z_p for ArcMark multi-bit watermarking.

    Each message is represented as a *d*-vector in Z_p (where
    *d* = ceil(log_p(M))), and its codeword is the matrix-vector product

        C_m = m_vec @ G  (mod p)

    where G is a random generator matrix of shape ``(d, n)`` with entries
    drawn uniformly from {0, ..., p-1}.

    The code is **fully deterministic** given ``(num_messages,
    codeword_length, alphabet_size, seed)``; encoder and decoder construct
    identical objects from these shared parameters.

    Use the :meth:`build` classmethod to construct instances.

    Example::

        code = RandomLinearCode.build(
            num_messages=256,       # M = 2^8 (8-bit message)
            codeword_length=100,    # n = 100 tokens
            alphabet_size=256,      # p = 256
            seed=42,
        )
        codeword = code.encode(0)           # shape (100,)
        all_cw   = code.codebook            # shape (256, 100)
        bits     = code.int_to_bits(42)     # shape (8,)
    """

    def __init__(
        self,
        *,
        generator_matrix: Tensor,
        message_vectors: Tensor,
        _codebook: Tensor,
        _alphabet_size: int,
        _seed: int,
    ) -> None:
        # Store code parameters (all tensors are CPU, dtype long)
        self._generator_matrix = generator_matrix
        self._message_vectors = message_vectors
        self._codebook = _codebook
        self._alphabet_size = _alphabet_size
        self._seed = _seed

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        num_messages: int,
        codeword_length: int,
        alphabet_size: int,
        seed: int,
    ) -> RandomLinearCode:
        """Construct a random linear code from shared parameters.

        Deterministic: identical arguments always produce the same code.

        Args:
            num_messages:   Number of messages *M* (must be >= 1).
            codeword_length: Codeword length *n* (must be >= 1).
            alphabet_size:  Alphabet size *p* (must be >= 2).
            seed:           Integer seed for the random generator.

        Returns:
            A fully initialised :class:`RandomLinearCode` instance.

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        # --- validation ---
        if num_messages < 1:
            raise ValueError(f"num_messages must be >= 1, got {num_messages}")
        if codeword_length < 1:
            raise ValueError(
                f"codeword_length must be >= 1, got {codeword_length}"
            )
        if alphabet_size < 2:
            raise ValueError(
                f"alphabet_size must be >= 2, got {alphabet_size}"
            )

        M = num_messages
        n = codeword_length
        p = alphabet_size

        # Message vector dimension: d = ceil(log_p(M)), at least 1.
        # For M=1, d=1 and the single message vector is the zero vector.
        d = max(1, int(math.ceil(math.log(M) / math.log(p))))

        # Verify there are enough distinct d-vectors in Z_p^d
        if p ** d < M:
            raise ValueError(
                f"Cannot build code: p^d = {p}^{d} = {p**d} < M = {M}. "
                f"Need alphabet_size^dimension >= num_messages."
            )

        # --- deterministic RNG (matches keygen.py pattern) ---
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        # --- generator matrix G of shape (d, n) ---
        G = torch.randint(0, p, (d, n), generator=gen, dtype=torch.long)

        # --- sample M distinct message vectors in Z_p^d ---
        # Draw vectors one at a time, reject duplicates.
        # This matches the legacy build_random_linear_code logic.
        seen: set[tuple[int, ...]] = set()
        msg_list: list[Tensor] = []

        while len(msg_list) < M:
            # Draw a candidate d-vector
            vec = torch.randint(0, p, (d,), generator=gen, dtype=torch.long)
            key = tuple(vec.tolist())
            if key not in seen:
                seen.add(key)
                msg_list.append(vec)

        message_vectors = torch.stack(msg_list, dim=0)  # shape (M, d)

        # --- compute codebook: C = message_vectors @ G  (mod p) ---
        codebook = (message_vectors @ G) % p  # shape (M, n)

        return cls(
            generator_matrix=G,
            message_vectors=message_vectors,
            _codebook=codebook,
            _alphabet_size=p,
            _seed=seed,
        )

    # ── Abstract implementations ──────────────────────────────────────────

    def encode(self, message_idx: int) -> Tensor:
        """Return the codeword for message *message_idx*.

        Args:
            message_idx: Integer in {0, ..., num_messages - 1}.

        Returns:
            LongTensor of shape ``(codeword_length,)`` with symbols in
            {0, ..., alphabet_size - 1}.

        Raises:
            IndexError: If *message_idx* is out of range.
        """
        M = self._codebook.shape[0]
        if not (0 <= message_idx < M):
            raise IndexError(
                f"message_idx={message_idx} out of range [0, {M})"
            )
        return self._codebook[message_idx]

    @property
    def codebook(self) -> Tensor:
        """All codewords, shape ``(M, n)``, dtype long."""
        return self._codebook

    @property
    def num_messages(self) -> int:
        """Number of messages *M*."""
        return self._codebook.shape[0]

    @property
    def codeword_length(self) -> int:
        """Codeword length *n*."""
        return self._codebook.shape[1]

    @property
    def alphabet_size(self) -> int:
        """Alphabet size *p*."""
        return self._alphabet_size

    # ── Extra properties ──────────────────────────────────────────────────

    @property
    def seed(self) -> int:
        """The seed used to generate this code."""
        return self._seed

    @property
    def dimension(self) -> int:
        """Message vector dimension *d* = rows of the generator matrix."""
        return self._generator_matrix.shape[0]

    @property
    def generator_matrix(self) -> Tensor:
        """Generator matrix G of shape ``(d, n)``, dtype long."""
        return self._generator_matrix

    @property
    def message_vectors(self) -> Tensor:
        """Message vectors of shape ``(M, d)``, dtype long."""
        return self._message_vectors
