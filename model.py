from typing import List

import torch

from MixtureFamily import MixtureFamily


def get_model(
    mixture_family: MixtureFamily,
    K: int,
    D: int
) -> List[torch.nn.Parameter]:
    pi_logits = torch.nn.Parameter(torch.rand(K, dtype=torch.float32), requires_grad=True)
    mus = torch.nn.Parameter(torch.rand(K, D, dtype=torch.float32) * 10, requires_grad=True)

    if mixture_family == MixtureFamily.FULL:
        sigmas_sqrt = torch.nn.Parameter(torch.rand(K, D, D, dtype=torch.float32), requires_grad=True)
        return {
            "pi_logits": pi_logits, 
            "mus": mus,
            "sigmas_sqrt": sigmas_sqrt
        }
    
    else:
        sigmas_diag = torch.nn.Parameter(torch.rand(K, D, dtype=torch.float32), requires_grad=True)
        return {
            "pi_logits": pi_logits, 
            "mus": mus,
            "sigmas_diag": sigmas_diag
        }
