#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Gauge projection utilities for GP-LoRA (Gauge-Projected Low-Rank Adaptation).

This module implements the ε-regularized gauge projection that preserves Δ = BA exactly
while enforcing the imbalance constraint AA^T ≈ μ B^T B.
"""

import torch
from torch import Tensor
from typing import Tuple, Union

__all__ = ['gauge_project_factors', 'GaugeProjectedOptimizer']


def _symmetrize(M: Tensor) -> Tensor:
    """Force symmetry: 0.5 * (M + M.T)"""
    return 0.5 * (M + M.transpose(-2, -1))


def _spd_sqrt_and_invsqrt(
    M: Tensor, 
    eig_floor: float = 1e-12
) -> Tuple[Tensor, Tensor]:
    """
    Compute M^{1/2} and M^{-1/2} for a symmetric positive semi-definite matrix M
    via eigendecomposition with eigenvalue clamping.
    
    Args:
        M: Symmetric PSD matrix of shape (..., r, r)
        eig_floor: Minimum eigenvalue to prevent numerical issues
        
    Returns:
        Tuple of (M^{1/2}, M^{-1/2})
    """
    # Ensure symmetry
    M = _symmetrize(M)
    
    # Eigendecomposition: M = V @ diag(evals) @ V^T
    evals, V = torch.linalg.eigh(M)
    
    # Clamp eigenvalues for numerical stability
    evals_clamped = evals.clamp(min=eig_floor)
    
    # Compute sqrt and inverse sqrt of eigenvalues
    evals_sqrt = evals_clamped.sqrt()
    evals_invsqrt = 1.0 / evals_sqrt
    
    # Reconstruct M^{1/2} = V @ diag(sqrt(evals)) @ V^T
    # and M^{-1/2} = V @ diag(1/sqrt(evals)) @ V^T
    M_sqrt = V @ (evals_sqrt.unsqueeze(-1) * V.transpose(-2, -1))
    M_invsqrt = V @ (evals_invsqrt.unsqueeze(-1) * V.transpose(-2, -1))
    
    return M_sqrt, M_invsqrt


def gauge_project_factors(
    A: Tensor,
    B: Tensor,
    mu: Union[float, str] = "auto",
    eps: float = 1e-4,
    eig_floor: float = 1e-12,
    cast_fp32: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    Apply gauge projection Proj_{μ,ε} that preserves Δ = BA exactly while
    enforcing the imbalance constraint AA^T ≈ μ B^T B.
    
    The projection finds R such that (A', B') = (R @ A, B @ R^{-1}) satisfies
    A' A'^T = μ B'^T B' (up to ε regularization), while B' A' = B A.
    
    From the notes (Section 4.2), the closed form is:
        G_A^ε = AA^T + εI
        G_B^ε = B^T B + εI
        S^ε = G_A^{-1/2} @ (G_A^{1/2} @ (μ G_B^ε) @ G_A^{1/2})^{1/2} @ G_A^{-1/2}
        R^ε = (S^ε)^{1/2}
        
    Args:
        A: LoRA A matrix of shape (r, n)
        B: LoRA B matrix of shape (m, r)
        mu: Imbalance ratio. If "auto", uses r/m (dimension-calibrated).
        eps: Regularization parameter for Gram matrices (default 1e-4)
        eig_floor: Minimum eigenvalue for numerical stability (default 1e-12)
        cast_fp32: Whether to compute in fp32 for stability (default True)
        
    Returns:
        Tuple of (A_new, B_new) where A_new = R @ A and B_new = B @ R^{-1}
    """
    # Validate shapes
    r = A.shape[0]
    assert B.shape[1] == r, f"Rank dimension mismatch: A.shape[0]={r}, B.shape[1]={B.shape[1]}"
    
    # Handle mu="auto"
    m = B.shape[0]
    if mu == "auto":
        mu = r / m
    
    # Store original dtype and device
    orig_dtype = A.dtype
    device = A.device
    
    # Optionally cast to fp32 for numerical stability
    if cast_fp32 and orig_dtype != torch.float32:
        A_work = A.float()
        B_work = B.float()
    else:
        A_work = A
        B_work = B
    
    # Build regularized Gram matrices (r x r)
    eye_r = torch.eye(r, dtype=A_work.dtype, device=device)
    G_A = A_work @ A_work.T + eps * eye_r  # (r, r)
    G_B = B_work.T @ B_work + eps * eye_r  # (r, r)
    
    # Compute G_A^{1/2} and G_A^{-1/2}
    G_A_sqrt, G_A_invsqrt = _spd_sqrt_and_invsqrt(G_A, eig_floor)
    
    # Compute the middle term: (G_A^{1/2} @ (μ G_B) @ G_A^{1/2})^{1/2}
    # This is μ * G_A^{1/2} @ G_B @ G_A^{1/2}
    middle = mu * (G_A_sqrt @ G_B @ G_A_sqrt)
    middle_sqrt, _ = _spd_sqrt_and_invsqrt(middle, eig_floor)
    
    # Compute S^ε = G_A^{-1/2} @ middle_sqrt @ G_A^{-1/2}
    S = G_A_invsqrt @ middle_sqrt @ G_A_invsqrt
    
    # Compute R = (S^ε)^{1/2} and R^{-1} = (S^ε)^{-1/2}
    R, R_inv = _spd_sqrt_and_invsqrt(S, eig_floor)
    
    # Apply the gauge transformation
    A_new = R @ A_work
    B_new = B_work @ R_inv
    
    # Cast back to original dtype if needed
    if cast_fp32 and orig_dtype != torch.float32:
        A_new = A_new.to(orig_dtype)
        B_new = B_new.to(orig_dtype)
    
    return A_new, B_new


class GaugeProjectedOptimizer:
    """
    Optimizer wrapper that applies gauge projection after each optimizer step.
    
    This implements Algorithm 1 from the GP-LoRA notes: standard optimizer step
    followed by Proj_{μ,ε} projection on all LoRA layers.
    
    Example:
        >>> opt = GaugeProjectedOptimizer(torch.optim.AdamW(model.parameters()), model)
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     opt.step()
        ...     opt.zero_grad()
    """
    
    def __init__(
        self,
        base_optimizer,
        model,
        mu: Union[float, str] = "auto",
        eps: float = 1e-4,
        eig_floor: float = 1e-12,
        cast_fp32: bool = True
    ):
        """
        Args:
            base_optimizer: The underlying optimizer (e.g., AdamW)
            model: The model containing LoRA layers
            mu: Imbalance ratio. If "auto", uses r/m per adapter.
            eps: Regularization parameter (default 1e-4)
            eig_floor: Minimum eigenvalue for stability (default 1e-12)
            cast_fp32: Whether to compute in fp32 (default True)
        """
        self.base_optimizer = base_optimizer
        self.model = model
        self.mu = mu
        self.eps = eps
        self.eig_floor = eig_floor
        self.cast_fp32 = cast_fp32
    
    def step(self, closure=None):
        """Perform optimization step followed by gauge projection."""
        # Import here to avoid circular imports
        from .utils import gauge_project_model
        
        loss = self.base_optimizer.step(closure)
        gauge_project_model(
            self.model,
            mu=self.mu,
            eps=self.eps,
            eig_floor=self.eig_floor,
            cast_fp32=self.cast_fp32
        )
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        """Clear gradients."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
    
    @property
    def param_groups(self):
        """Access underlying optimizer's param_groups."""
        return self.base_optimizer.param_groups
    
    @param_groups.setter
    def param_groups(self, value):
        """Set underlying optimizer's param_groups."""
        self.base_optimizer.param_groups = value
    
    def state_dict(self):
        """Return state dict of underlying optimizer."""
        return self.base_optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict into underlying optimizer."""
        self.base_optimizer.load_state_dict(state_dict)

