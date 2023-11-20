import torch


"""
class FullNormal(torch.nn.Module):
    def __init__(
        self,
        mixture: torch.Tensor,
        means: torch.Tensor,
        covariance: torch.Tensor
    ):
        super().__init__()

        self.

"""


def get_nll_loss(
    ys: torch.Tensor,  # [X, K, D, 1]
    pis: torch.Tensor,  # [K]
    mus: torch.Tensor,  # [K, D]
    sigmas: torch.Tensor,  # [K, D, D]
):
    num_samples = ys.shape[0]

    log_pis = torch.log_softmax(pis, dim=0)
    log_pis = log_pis.repeat(num_samples, 1)  # [X, K]

    mus = mus.repeat(num_samples, 1, 1)  # [X, K, D]
    mus = mus.unsqueeze(-1)  # [X, K, D, 1]

    sigmas = sigmas.repeat(num_samples, 1, 1, 1)  # [X, K, D, D]

    y_minus_mu = ys - mus  # [X, K, D, 1]
    sigmas_inverse = torch.cholesky_inverse(sigmas)

    first_term = log_pis  # [X, K]
    second_term = -0.5 * y_minus_mu.transpose(-2, -1) @ sigmas_inverse @ y_minus_mu  # [X, K, 1, 1]
    third_term = -0.5 * torch.logdet(sigmas).nan_to_num(nan=0.0, neginf=0.0)  # [X, K]
    
    """
    print(sigmas.shape)
    print(first_term.shape)
    print(second_term.shape)
    print(third_term.shape)
    #exit(0)
    """

    log_likelihoods = torch.logsumexp(
        first_term +
        second_term.squeeze(-1).squeeze(-1) +
        third_term,
        dim=1  # sum across components
    )

    return -1 * torch.mean(log_likelihoods)


"""
def get_nll_loss(
    ys: torch.Tensor,
    pis: torch.Tensor,  # [K]
    mus: torch.Tensor,  # [K, D]
    covariances: torch.Tensor,  # [K, D, D]
):
    z_scores = (ys - mus) / 
    torch.logsumexp(
        torch.log(pis - 0.5 * torch.norm(, p=2))
    )
"""
