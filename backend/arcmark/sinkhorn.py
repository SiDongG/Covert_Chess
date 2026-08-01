"""Entropic optimal-transport solver and ArcMark-specific wrappers.

Two layers:

Layer 1 (core solver):
    :func:`solve_ot` — generic entropic OT via POT's ``ot.sinkhorn``,
    accepting arbitrary marginals and a cost matrix.

Layer 2 (ArcMark wrappers):
    :func:`restrict_vocab` — top-k / top-p vocabulary filtering.
    :func:`solve_arcmark_ot` — builds circular cost matrix, applies
    permutation and vocabulary restriction, calls :func:`solve_ot`.
    :func:`extract_conditional` — extracts P*(x | s) from a coupling.

All public functions accept PyTorch tensors.  POT auto-detects the torch
backend and runs on the same device as the input tensors (CPU/CUDA/MPS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ot as pot
import torch
from torch import Tensor

from arcmark import geometry

if TYPE_CHECKING:
    from arcmark.config import ArcMarkConfig

__all__ = [
    "ArcMarkOTResult",
    "extract_conditional",
    "restrict_vocab",
    "solve_arcmark_ot",
    "solve_ot",
]


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: Core entropic OT solver
# ═══════════════════════════════════════════════════════════════════════════


def solve_ot(
    p_source: Tensor,
    p_target: Tensor,
    cost: Tensor,
    reg: float = 0.05,
    *,
    max_iter: int = 4000,
    stop_thr: float = 1e-9,
    method: str = "sinkhorn",
    warn: bool = True,
) -> Tensor:
    r"""Solve the entropic optimal transport problem.

    Finds the coupling :math:`\gamma^*` that minimises

    .. math::
        \langle C, \gamma \rangle_F
        \;+\; \varepsilon \sum_{ij} \gamma_{ij} \log \gamma_{ij}

    subject to :math:`\gamma \mathbf{1} = a` and
    :math:`\gamma^\top \mathbf{1} = b`.

    This wraps ``ot.sinkhorn`` from the `POT <https://pythonot.github.io/>`_
    library.  POT auto-detects PyTorch tensors and runs on the same
    device as the inputs (CPU, CUDA, or MPS) — no data transfer needed.

    For GPU use, ``method="sinkhorn_log"`` is recommended as it is more
    numerically stable.

    Args:
        p_source: Source marginal, shape ``(m,)``.  Must sum to ≈ 1.
        p_target: Target marginal, shape ``(n,)``.  Must sum to ≈ 1.
        cost:     Cost matrix, shape ``(m, n)``.
        reg:      Entropic regularisation (> 0).
        max_iter: Maximum Sinkhorn iterations.
        stop_thr: Convergence threshold on marginal violation.
        method:   POT method (``"sinkhorn"``, ``"sinkhorn_log"``,
                  ``"sinkhorn_stabilized"``).
        warn:     Whether POT warns on non-convergence.

    Returns:
        Coupling matrix of shape ``(m, n)`` on the same device as *cost*.

    Raises:
        ValueError: On shape mismatch, non-positive *reg*, or marginals
            that do not sum to approximately 1.
    """
    # --- validation ---
    if reg <= 0:
        raise ValueError(f"reg must be > 0, got {reg}")
    if p_source.dim() != 1 or p_target.dim() != 1 or cost.dim() != 2:
        raise ValueError(
            f"Expected 1-D marginals and 2-D cost; got shapes "
            f"{p_source.shape}, {p_target.shape}, {cost.shape}"
        )
    m, n = cost.shape
    if p_source.shape[0] != m:
        raise ValueError(
            f"p_source length {p_source.shape[0]} != cost rows {m}"
        )
    if p_target.shape[0] != n:
        raise ValueError(
            f"p_target length {p_target.shape[0]} != cost cols {n}"
        )
    _check_marginal_sum(p_source, "p_source")
    _check_marginal_sum(p_target, "p_target")

    # --- solve on-device via POT's torch backend ---
    dtype = cost.dtype
    # Use float32 on GPU (MPS lacks float64; CUDA float32 is faster and sufficient).
    compute_dtype = torch.float32 if cost.device.type in ("mps", "cuda") else torch.float64
    _a = p_source.detach().to(compute_dtype)
    _b = p_target.detach().to(compute_dtype)
    _M = cost.detach().to(compute_dtype)

    gamma = pot.sinkhorn(
        _a, _b, _M,
        reg=reg,
        numItermax=max_iter,
        stopThr=stop_thr,
        method=method,
        warn=warn,
    )

    return gamma.to(dtype=dtype)


def _check_marginal_sum(p: Tensor, name: str, tol: float = 1e-3) -> None:
    s = float(p.sum())
    if abs(s - 1.0) > tol:
        raise ValueError(
            f"{name} sums to {s:.6f}, expected ≈ 1.0 (tolerance {tol})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Conditional extraction
# ═══════════════════════════════════════════════════════════════════════════


def extract_conditional(
    coupling: Tensor,
    s_index: int,
    *,
    num_keys: int | None = None,
    full_vocab_size: int | None = None,
    token_indices: Tensor | None = None,
) -> Tensor:
    r"""Extract :math:`P^*(x \mid s = s_{\text{index}})` from a coupling.

    Given coupling :math:`\gamma` of shape ``(r, n)`` whose row marginal
    is uniform(r):

    .. math::
        P^*(x_j \mid s = s_i) = r \cdot \gamma[i, j]

    Args:
        coupling:        OT coupling, shape ``(r, n_sel)``.
        s_index:         Realised side-information key index.
        num_keys:        Number of keys *r*.  Inferred from ``coupling.shape[0]``
                         if not given.
        full_vocab_size: If given with *token_indices*, scatters the result
                         into a tensor of shape ``(full_vocab_size,)``.
        token_indices:   ``LongTensor (n_sel,)`` mapping coupling columns
                         to full-vocabulary positions.

    Returns:
        Conditional distribution, shape ``(n_sel,)`` or ``(full_vocab_size,)``.

    Raises:
        IndexError: If *s_index* is out of range.
        ValueError: If *full_vocab_size* given without *token_indices*.
    """
    r_dim = coupling.shape[0]
    if not (0 <= s_index < r_dim):
        raise IndexError(
            f"s_index={s_index} out of range for coupling with "
            f"{r_dim} source bins"
        )
    if full_vocab_size is not None and token_indices is None:
        raise ValueError(
            "token_indices is required when full_vocab_size is given"
        )

    r = num_keys if num_keys is not None else r_dim
    cond = coupling[s_index, :] * float(r)
    cond = cond.clamp(min=0.0)
    total = cond.sum()
    if total > 0:
        cond = cond / total
    else:
        cond = torch.ones_like(cond) / cond.shape[0]

    if full_vocab_size is None:
        return cond

    out = torch.zeros(
        full_vocab_size, dtype=cond.dtype, device=cond.device
    )
    out.scatter_(0, token_indices.to(cond.device), cond)
    out_sum = out.sum()
    if out_sum > 0:
        out = out / out_sum
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Vocabulary restriction (top-k / top-p)
# ═══════════════════════════════════════════════════════════════════════════


def restrict_vocab(
    probs: Tensor,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
    min_tokens: int = 2,
) -> tuple[Tensor, Tensor]:
    """Restrict vocabulary to the most probable tokens.

    Top-k is applied first, then top-p within the top-k set.  The
    returned probabilities are renormalised to sum to 1.

    Args:
        probs:      Probability distribution, shape ``(V,)``.
        top_k:      Keep at most this many tokens.
        top_p:      Keep the smallest set with cumulative mass ≥ *top_p*.
        min_tokens: Always keep at least this many tokens (default 2).

    Returns:
        ``(indices, restricted_probs)`` — *indices* is a LongTensor of
        original vocabulary positions (sorted by descending probability);
        *restricted_probs* is the renormalised distribution.

    Raises:
        ValueError: If *probs* is not 1-D or all zeros.
    """
    if probs.dim() != 1:
        raise ValueError(f"Expected 1-D probs, got shape {probs.shape}")
    if probs.sum() <= 0:
        raise ValueError("All probabilities are zero")

    V = probs.shape[0]
    min_tokens = max(1, min(min_tokens, V))

    # --- top-k ---
    if top_k is not None and top_k < V:
        k = max(top_k, min_tokens)
    else:
        k = V
    topk_probs, topk_indices = torch.topk(probs, k=k)

    # --- top-p within top-k ---
    if top_p is not None:
        cumsum = topk_probs.cumsum(dim=0)
        n_keep = int((cumsum < top_p).sum().item()) + 1
        n_keep = max(min_tokens, n_keep)
        n_keep = min(n_keep, k)
        topk_probs = topk_probs[:n_keep]
        topk_indices = topk_indices[:n_keep]

    # --- renormalise ---
    topk_probs = topk_probs / topk_probs.sum()

    return topk_indices, topk_probs


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: ArcMark-specific OT wrapper
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ArcMarkOTResult:
    """Result container for :func:`solve_arcmark_ot`.

    Attributes:
        coupling:      OT coupling, shape ``(num_keys, n_sel)``.
        token_indices: Original vocabulary positions, shape ``(n_sel,)``.
        token_probs:   Renormalised probabilities, shape ``(n_sel,)``.
        cost_matrix:   Cost matrix used, shape ``(num_keys, n_sel)``.
    """

    coupling: Tensor
    token_indices: Tensor
    token_probs: Tensor
    cost_matrix: Tensor


def solve_arcmark_ot(
    probs: Tensor,
    *,
    codeword_symbol: int,
    alphabet_size: int,
    num_keys: int,
    vocab_size: int,
    perm: Tensor | None = None,
    phi: float = 0.0,
    reg: float = 0.05,
    top_k: int | None = None,
    top_p: float | None = None,
    min_tokens: int = 2,
    max_iter: int = 4000,
    stop_thr: float = 1e-9,
    method: str = "sinkhorn",
    config: ArcMarkConfig | None = None,
) -> ArcMarkOTResult:
    r"""Solve the ArcMark OT problem for a single token position.

    1. Restricts vocabulary via :func:`restrict_vocab`.
    2. Applies the secret permutation to selected token indices.
    3. Computes (permuted) token angles and side-information angles
       using :mod:`arcmark.geometry`.
    4. Builds the circular-distance cost matrix.
    5. Solves entropic OT with a uniform source marginal over
       *num_keys* side-information bins.

    Args:
        probs:           Full-vocabulary probabilities, shape ``(V,)``.
        codeword_symbol: Encoded symbol :math:`C_m(t)` in
                         ``{0, …, alphabet_size − 1}``.
        alphabet_size:   Code alphabet size *p*.
        num_keys:        Number of secret-key values *r*.
        vocab_size:      Total vocabulary size *N* (for angle computation).
        perm:            Permutation tensor, shape ``(V,)``.  ``perm[i]``
                         is the permuted index of token *i*.
        phi:             Angle offset φ (default 0).
        reg:             Entropic regularisation (default 0.05).
        top_k:           Top-k restriction (optional).
        top_p:           Top-p / nucleus restriction (optional).
        min_tokens:      Minimum tokens to keep (default 2).
        max_iter:        Maximum Sinkhorn iterations.
        stop_thr:        Convergence threshold.
        method:          POT Sinkhorn method.
        config:          Optional :class:`~arcmark.config.ArcMarkConfig`.
                         When provided, its values override *reg*, *top_k*,
                         *top_p*, *min_tokens*, *max_iter*, *stop_thr*,
                         and *method*.

    Returns:
        :class:`ArcMarkOTResult` with coupling, token indices,
        renormalised probabilities, and cost matrix.

    Raises:
        ValueError: If *codeword_symbol* is out of range or *probs*
            is invalid.
    """
    if config is not None:
        reg = config.sinkhorn_reg
        top_k = config.top_k
        top_p = config.top_p
        min_tokens = config.min_tokens
        max_iter = config.max_iter
        stop_thr = config.stop_thr
        method = config.method

    if not (0 <= codeword_symbol < alphabet_size):
        raise ValueError(
            f"codeword_symbol={codeword_symbol} out of range "
            f"[0, {alphabet_size})"
        )

    # 1. restrict vocabulary
    indices, probs_sel = restrict_vocab(
        probs, top_k=top_k, top_p=top_p, min_tokens=min_tokens,
    )

    # 2. apply permutation
    if perm is not None:
        permuted_ids = perm[indices]
    else:
        permuted_ids = indices

    # 3. compute angles (geometry functions)
    tok_angs = geometry.token_angles(permuted_ids, vocab_size)
    w_angs = geometry.side_info_angles(
        codeword_symbol, alphabet_size, num_keys, phi,
    )
    w_angs = w_angs.to(tok_angs.device)

    # 4. build cost matrix — shape (num_keys, n_sel)
    C = geometry.build_cost_matrix(tok_angs, w_angs)
    C = C.to(dtype=probs_sel.dtype)

    # 5. uniform source marginal
    p_s = torch.full(
        (num_keys,),
        1.0 / num_keys,
        dtype=probs_sel.dtype,
        device=probs_sel.device,
    )

    # 6. solve OT
    coupling = solve_ot(
        p_s, probs_sel, C,
        reg=reg,
        max_iter=max_iter,
        stop_thr=stop_thr,
        method=method,
    )

    return ArcMarkOTResult(
        coupling=coupling,
        token_indices=indices,
        token_probs=probs_sel,
        cost_matrix=C,
    )
