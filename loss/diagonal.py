import torch


def get_diagonal_nll_loss(
    _ys: torch.Tensor,  # [X, K, D, 1]
    pis: torch.Tensor,  # [K]
    _mus: torch.Tensor,  # [K, D]
    _sigmas_diag: torch.Tensor,  # [K, D]
):
    num_samples = _ys.shape[0]

    # pis are an element of the K-1 simplex
    _log_pis = torch.log(pis)
    log_pis = _log_pis.repeat(num_samples, 1)  # [X, K]

    # 
    ys = _ys.squeeze(-1)  # [X, K, D]
    mus = _mus.repeat(num_samples, 1, 1)  # [X, K, D]

    sigmas_diag = _sigmas_diag.repeat(num_samples, 1, 1)  # [X, K, D]

    # compute likelihood
    first_term = log_pis  # [X, K]
    second_term = -0.5 * torch.sum(((ys - mus) / sigmas_diag) ** 2, dim=-1)  # [X, K]
    third_term = -1 * torch.sum(torch.abs(torch.log(sigmas_diag)), dim=-1)  # [X, K] numerical stability

    log_likelihoods = torch.logsumexp(
        first_term +
        second_term +
        third_term,
        dim=1  # sum across components
    )

    #return -1 * torch.mean(log_likelihoods)
    return -1 * log_likelihoods.mean()  # mean across samples
