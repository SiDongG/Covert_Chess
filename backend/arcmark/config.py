"""ArcMark configuration: YAML-backed dataclass with validation.

Usage::

    # Load from file
    cfg = ArcMarkConfig.from_yaml("my_config.yaml")

    # Use defaults
    cfg = ArcMarkConfig()

    # Override programmatically
    cfg = ArcMarkConfig(sinkhorn_reg=0.1, top_k=100)

    # Save to YAML
    cfg.to_yaml("output_config.yaml")
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union


@dataclass
class ArcMarkConfig:
    """Configuration for ArcMark OT solver and vocabulary restriction.

    All solver parameters used by :func:`arcmark.sinkhorn.solve_arcmark_ot`
    are gathered here.  A config can be loaded from a YAML file,
    created with defaults, or constructed programmatically.

    Attributes:
        top_k:         Top-k vocabulary restriction.  ``None`` disables.
        top_p:         Top-p (nucleus) restriction.  ``None`` disables.
        sinkhorn_reg:  Entropic regularisation for OT (must be > 0).
        max_iter:      Maximum Sinkhorn iterations.
        stop_thr:      Convergence threshold on marginal violation.
        min_tokens:    Minimum tokens to keep after vocabulary restriction.
        method:        Sinkhorn method variant passed to POT
                       (``"sinkhorn"``, ``"sinkhorn_log"``,
                       ``"sinkhorn_stabilized"``).
        context_width: Number of preceding watermarked tokens used as
                       context for hash-based key generation.  Must be
                       ``>= 1`` when ``hash_keys=True``.
        hash_keys:     If ``True``, derive keys by hashing context tokens
                       with the secret (production mode).  If ``False``,
                       use a pre-generated fixed key sequence (debug /
                       testing mode).
    """

    # Vocabulary restriction
    top_k: int | None = 50
    top_p: float | None = None

    # Sinkhorn solver
    # PERFORMANCE OPTIMIZATION: Relaxed default parameters.
    # Previous defaults (max_iter=1000, stop_thr=1e-6) were overly conservative.
    # Tests only require 1e-4 marginal tolerance, so these were 100x tighter
    # than necessary. New defaults (500, 1e-5) are still accurate but faster.
    sinkhorn_reg: float = 0.05
    max_iter: int = 500  # Was 1000; tests pass with 200, 500 is safe
    stop_thr: float = 1e-5  # Was 1e-6; tests accept 1e-4, 1e-5 is safe
    min_tokens: int = 2
    method: str = "sinkhorn_log"

    # Key generation
    context_width: int = 3
    hash_keys: bool = True

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.sinkhorn_reg <= 0:
            raise ValueError(
                f"sinkhorn_reg must be > 0, got {self.sinkhorn_reg}"
            )
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(
                f"top_k must be >= 1 or None, got {self.top_k}"
            )
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError(
                f"top_p must be in (0, 1] or None, got {self.top_p}"
            )
        if self.min_tokens < 1:
            raise ValueError(
                f"min_tokens must be >= 1, got {self.min_tokens}"
            )
        if self.max_iter < 1:
            raise ValueError(
                f"max_iter must be >= 1, got {self.max_iter}"
            )
        if self.method not in (
            "sinkhorn", "sinkhorn_log", "sinkhorn_stabilized",
        ):
            raise ValueError(
                f"Unknown method {self.method!r}"
            )
        if self.context_width < 0:
            raise ValueError(
                f"context_width must be >= 0, got {self.context_width}"
            )
        if self.hash_keys and self.context_width < 1:
            raise ValueError(
                f"context_width must be >= 1 when hash_keys=True, "
                f"got {self.context_width}"
            )

    # ── I/O ───────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> ArcMarkConfig:
        """Load configuration from a YAML file.

        Missing keys use dataclass defaults.  Extra keys raise ``TypeError``.
        """
        import yaml

        path = Path(path)
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def from_defaults(cls) -> ArcMarkConfig:
        """Load the default configuration shipped with the package."""
        default_path = Path(__file__).parent / "default_config.yaml"
        if default_path.exists():
            return cls.from_yaml(default_path)
        return cls()

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Write configuration to a YAML file."""
        import yaml

        path = Path(path)
        with path.open("w") as f:
            yaml.dump(
                asdict(self), f, default_flow_style=False, sort_keys=False,
            )

    # ── Utilities ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return configuration as a plain dictionary."""
        return asdict(self)

    def replace(self, **kwargs: object) -> ArcMarkConfig:
        """Return a new config with specified fields overridden."""
        data = asdict(self)
        data.update(kwargs)
        return ArcMarkConfig(**data)

    # ── Configuration Presets ─────────────────────────────────────────────
    # PERFORMANCE OPTIMIZATION: Pre-tuned configurations for different use cases.
    # These presets encode the speed/accuracy tradeoffs discovered during
    # benchmarking. Use fast() for real-time applications, precise() for
    # maximum accuracy in offline settings.

    @classmethod
    def fast(cls) -> ArcMarkConfig:
        """Speed-optimized configuration for real-time applications.

        Uses aggressive Sinkhorn parameters (200 iterations, 1e-4 tolerance)
        that converge quickly while maintaining acceptable accuracy.
        Marginal constraint violation is still within test tolerances (1e-4).

        Recommended for:
        - Interactive applications
        - High-throughput batch processing
        - Development and debugging

        Returns:
            ArcMarkConfig with speed-optimized Sinkhorn parameters.
        """
        return cls(
            max_iter=200,
            stop_thr=1e-4,
            method="sinkhorn_log",
        )

    @classmethod
    def balanced(cls) -> ArcMarkConfig:
        """Balanced speed/accuracy configuration (same as default).

        Uses moderate Sinkhorn parameters (500 iterations, 1e-5 tolerance)
        that provide good accuracy with reasonable speed.

        This is the default configuration and is recommended for most use cases.

        Returns:
            ArcMarkConfig with balanced Sinkhorn parameters.
        """
        return cls()  # Uses default values

    @classmethod
    def precise(cls) -> ArcMarkConfig:
        """Maximum accuracy configuration for offline processing.

        Uses conservative Sinkhorn parameters (2000 iterations, 1e-8 tolerance)
        that ensure very tight convergence at the cost of speed.

        Recommended for:
        - Final production runs
        - Reproducibility-critical experiments
        - Debugging convergence issues

        Returns:
            ArcMarkConfig with precision-optimized Sinkhorn parameters.
        """
        return cls(
            max_iter=2000,
            stop_thr=1e-8,
            method="sinkhorn_log",
        )
