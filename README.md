# GMM Scratch #
A from-scratch implementation of negative log likelihood loss backpropogation for gaussian mixture models. For a full implementation using the [torch.distributions](https://docs.pytorch.org/docs/stable/distributions.html) module, see [GMMPytorch](https://github.com/kylesayrs/GMMPytorch/tree/master).

## Derivation ##
### Normal Distribution ###
```math
\mathcal{N}(\mathbf{x}) = \frac{1}{(2\pi)^{k/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
```

### Mixture of Normal Distributions ###
```math
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
```

### Negative Log Likelihood Objective ###
```math
f(\mathbf{x}) = - \frac{1}{N} \sum_{i=1}^{N} \ln{ \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) }
```

## Implementation ##
```python3
def get_full_nll_loss(
    ys: torch.Tensor,  # [X, K, D]
    pi_logits: torch.Tensor,  # [K]
    mus: torch.Tensor,  # [K, D]
    sigmas: torch.Tensor,  # [K, D, D]
):
    # pis are an element of the K-1 simplex
    log_pis = torch.log_softmax(pi_logits, dim=0)

    # unsqueeze for matrix multiplication
    ys = ys.unsqueeze(-1)  # [X, K, D, 1]
    mus = mus.unsqueeze(-1)  # [X, K, D, 1]

    # repeat for each sample
    num_samples = ys.shape[0]
    log_pis = log_pis.repeat(num_samples, 1)  # [X, K]
    mus = mus.repeat(num_samples, 1, 1, 1)  # [X, K, D, 1]
    sigmas = sigmas.repeat(num_samples, 1, 1, 1)  # [X, K, D, D]

    # precompute terms
    y_minus_mu = ys - mus  # [X, K, D, 1]
    sigmas_inverse = torch.cholesky_inverse(sigmas)  # sigmas is symmetric PSD

    # compute likelihood
    first_term = log_pis  # [X, K]
    second_term = -0.5 * y_minus_mu.transpose(-2, -1) @ sigmas_inverse @ y_minus_mu  # [X, K, 1, 1]
    third_term = -0.5 * torch.logdet(sigmas).nan_to_num(nan=0.0, neginf=0.0)  # [X, K]

    log_likelihoods = torch.logsumexp(
        first_term +
        second_term.squeeze(-1).squeeze(-1) +
        third_term,
        dim=1  # sum across components
    )

    return -1 * torch.mean(log_likelihoods)  # mean across samples
```
