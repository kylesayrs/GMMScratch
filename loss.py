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

    #sigmas_sqrt = torch.linalg.cholesky(sigmas)
    sigmas_inverse = torch.cholesky_inverse(sigmas)

    y_minus_mu = ys - mus  # [X, K, D, 1]

    first_term = log_pis  # [X, K]
    second_term = -0.5 * y_minus_mu.transpose(-2, -1) @ sigmas_inverse @ y_minus_mu  # [X, K, 1, 1]
    #third_term = -0.5 * torch.logdet(sigmas)  # [X, K]
    #third_term = -0.5 * torch.log(torch.det(
    third_term = -0.5 * torch.logdet(
        sigmas
    )  # [X, K]
    
    #print(sigmas.shape)
    #print(first_term.shape)
    #print(second_term.shape)
    #print(third_term.shape)
    #exit(0)

    log_likelihoods = torch.logsumexp(
        first_term.flatten() +
        second_term.flatten() +
        third_term.flatten(),
        dim=0
    )

    return -1 * log_likelihoods


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
