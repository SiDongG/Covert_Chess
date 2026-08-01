"""Angle-mapping and circular-distance primitives for ArcMark.

These functions implement the geometric operations on the unit circle that
underpin the ArcMark watermarking scheme (see CLAUDE.md § Paper Summary).
In ArcMark, tokens and codeword symbols are represented as angles on
[0, 2π), and the optimal-transport cost is the geodesic (shortest-arc)
distance between them.

All functions operate on PyTorch tensors, support arbitrary devices
(CPU / CUDA / MPS), and are stateless — no global variables.
"""

import math

import torch
from torch import Tensor

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# 1. Circular distance
# ---------------------------------------------------------------------------

def circular_dist(a: Tensor, b: Tensor) -> Tensor:
    """Shortest arc distance between two angles on the unit circle.

    Paper notation:
        d(θ₁, θ₂) = min{|θ₁ − θ₂|, 2π − |θ₁ − θ₂|}

    This is the cost function used in the OT problem that couples the
    side-information distribution to the token distribution.

    Uses modular arithmetic ``(a - b) % 2π`` rather than ``|a - b|`` so that
    inputs outside [0, 2π) (e.g. negative angles) are handled correctly.

    Args:
        a: Angle tensor (radians), any shape.  Broadcastable with *b*.
        b: Angle tensor (radians), any shape.  Broadcastable with *a*.

    Returns:
        Element-wise shortest arc distance in [0, π].
    """
    diff = (a - b) % TWO_PI
    return torch.minimum(diff, TWO_PI - diff)


# ---------------------------------------------------------------------------
# 2. Token-to-angle mapping
# ---------------------------------------------------------------------------

def token_angles(token_ids: Tensor, vocab_size: int) -> Tensor:
    """Map integer token IDs to uniformly-spaced angles on the unit circle.

    Paper notation:
        Token i ∈ 𝒳 = [0 : N−1] maps to angle  2πi / N,
        where N = |𝒳| is the vocabulary size.

    After applying the secret permutation Πₜ, the *permuted* token angle
    becomes 2π·Πₜ(i)/N.  That permutation step is the caller's
    responsibility (apply ``perm[token_ids]`` before calling this function).

    Args:
        token_ids: Integer tensor of token IDs (dtype typically ``long``).
        vocab_size: Total vocabulary size N.

    Returns:
        Float tensor of angles in [0, 2π), same shape as *token_ids*.
    """
    return TWO_PI * token_ids.float() / float(vocab_size)


# ---------------------------------------------------------------------------
# 3. Side-information (channel-input) angles
# ---------------------------------------------------------------------------

def side_info_angles(
    codeword_symbol: int,
    alphabet_size: int,
    num_keys: int,
    phi: float = 0.0,
) -> Tensor:
    """Compute the ``num_keys`` channel-input angles for a codeword symbol.

    At each token position t, the watermarker picks a codeword symbol
    C_m(t) ∈ {0, …, p−1} from the linear code and a secret key
    V_t ∈ {0, …, r−1}.  The channel-input angle is:

        z_t = (2π · C_m(t) / p  +  2π · V_t / r  +  φ)  mod 2π

    This function returns z_t for *all* possible key values V_t = 0 … r−1,
    yielding a tensor of shape ``(r,)``.  These angles serve as the source
    (row) marginal in the OT cost matrix.

    Paper parameters → function args:
        C_m(t)  →  codeword_symbol
        p       →  alphabet_size
        r       →  num_keys
        φ       →  phi   (capacity-achieving value: π / (2N))

    Args:
        codeword_symbol: The encoded symbol z ∈ {0, …, alphabet_size − 1}.
        alphabet_size:   Code alphabet size p.
        num_keys:        Number of secret-key values r (= side-info cardinality).
        phi:             Fixed angle offset φ (default 0).

    Returns:
        Float32 tensor of shape ``(num_keys,)`` with angles in [0, 2π).
    """
    # Compute in float64 to avoid accumulation errors when num_keys is large,
    # then cast to float32 for downstream GPU compatibility.
    base = TWO_PI * codeword_symbol / float(alphabet_size)
    s_vals = torch.arange(num_keys, dtype=torch.float64)
    angles = (base + TWO_PI * s_vals / float(num_keys) + phi) % TWO_PI
    return angles.float()


# ---------------------------------------------------------------------------
# 4. Deterministic random permutation
# ---------------------------------------------------------------------------

def random_permutation(vocab_size: int, seed: int) -> Tensor:
    """Generate a deterministic permutation of [0, vocab_size).

    In ArcMark the shared secret at position t is S_t = (V_t, Πₜ), where
    Πₜ is a random permutation of the token vocabulary.  This function
    produces Πₜ deterministically from *seed*, so that both encoder and
    decoder reconstruct the identical mapping.

    The caller is responsible for deriving *seed* from whatever combination
    of base key, token position, trial index, etc. is appropriate.

    Args:
        vocab_size: Vocabulary size N (length of the permutation).
        seed:       Integer seed for reproducibility.

    Returns:
        LongTensor of shape ``(vocab_size,)`` on CPU, where
        ``perm[original_id] = permuted_id``.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randperm(vocab_size, generator=gen)


# ---------------------------------------------------------------------------
# 5. OT cost matrix
# ---------------------------------------------------------------------------

def build_cost_matrix(token_angs: Tensor, w_angles: Tensor) -> Tensor:
    """Build the circular-distance cost matrix for the Sinkhorn OT solver.

    Each entry is the geodesic distance between a side-information angle
    and a (permuted) token angle:

        C[i, j] = d(w_angles[i],  token_angs[j])

    where d(·,·) is ``circular_dist``.  The Sinkhorn algorithm minimises
    the expected cost ∑ᵢⱼ γᵢⱼ Cᵢⱼ subject to marginal constraints, yielding
    the optimal coupling γ* from which the watermarked token distribution
    Q*(X_t | Z_t) is extracted.

    Args:
        token_angs: Tensor of (permuted) token angles, shape ``(n,)``.
        w_angles:   Tensor of side-information angles, shape ``(k,)``.

    Returns:
        Cost matrix of shape ``(k, n)`` with entries in [0, π].
    """
    # Broadcasting: (k, 1) vs (1, n) → (k, n)
    return circular_dist(w_angles.unsqueeze(1), token_angs.unsqueeze(0))
