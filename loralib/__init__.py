"""
GP-LoRA: Gauge-Projected Low-Rank Adaptation

A PyTorch library for parameter-efficient fine-tuning of large language models
using Low-Rank Adaptation (LoRA) with gauge-fixing projections.

GP-LoRA extends LoRA by exploiting the gauge symmetry of the low-rank factorization
Δ = BA to improve optimization dynamics. After each optimizer step, a gauge
projection enforces the imbalance constraint AA^⊤ = μB^⊤B while preserving
the effective weight update Δ exactly.

Key Features:
- Drop-in replacement for nn.Linear, nn.Embedding, and nn.Conv layers
- Automatic gauge projection via GaugeProjectedOptimizer
- Dimension-calibrated imbalance control (μ = r/m by default)
- Negligible computational overhead (O(r³) per layer)

Example:
    >>> import loralib as lora
    >>> 
    >>> # Replace layers with LoRA versions
    >>> layer = lora.Linear(768, 768, r=8)
    >>> 
    >>> # Use gauge-projected optimizer
    >>> from loralib import GaugeProjectedOptimizer
    >>> optimizer = GaugeProjectedOptimizer(
    ...     torch.optim.AdamW(model.parameters()),
    ...     model,
    ...     mu="auto",
    ...     eps=1e-4
    ... )
    >>> 
    >>> # Training loop
    >>> loss.backward()
    >>> optimizer.step()  # Projection happens automatically
    >>> optimizer.zero_grad()

References:
    - LoRA: https://arxiv.org/abs/2106.09685
    - GP-LoRA builds on invariant foliation theory and gauge symmetry
      in overparameterized neural networks
"""

name = "gplora"

from .layers import *
from .utils import *
from .gauge import gauge_project_factors, GaugeProjectedOptimizer
