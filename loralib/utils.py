#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict, Union

from .layers import LoRALayer


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def gauge_project_model(
    model: nn.Module,
    mu: Union[float, str] = "auto",
    eps: float = 1e-4,
    eig_floor: float = 1e-12,
    cast_fp32: bool = True
) -> None:
    """
    Apply gauge projection to all LoRA layers in the model.
    
    This is a convenience function that applies the gauge-fixing projection
    to every LoRA layer in the model. The projection preserves the forward
    function (Delta = BA is unchanged) but changes the factorization to
    enforce the imbalance constraint AA^T ≈ μ B^T B.
    
    This should be called after each optimizer step as part of GP-LoRA training:
    
        loss.backward()
        optimizer.step()
        gauge_project_model(model, mu="auto", eps=1e-4)
        optimizer.zero_grad()
    
    Args:
        model: The model containing LoRA layers
        mu: Imbalance ratio. If "auto", uses r/m per adapter (dimension-calibrated).
        eps: Regularization parameter for Gram matrices (default 1e-4)
        eig_floor: Minimum eigenvalue for numerical stability (default 1e-12)
        cast_fp32: Whether to compute in fp32 for stability (default True)
    """
    for m in model.modules():
        if isinstance(m, LoRALayer) and hasattr(m, 'gauge_project_'):
            m.gauge_project_(mu=mu, eps=eps, eig_floor=eig_floor, cast_fp32=cast_fp32)
