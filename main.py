import torch
import numpy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from loss import get_nll_loss


if __name__ == "__main__":
    K = 20
    D = 2
    samples_per_k = 50

    true_mus = []
    true_sigmas = []
    all_samples = []
    for _ in range(K):
        true_mu = numpy.random.rand(D) * 10
        true_sigma_sqrt = numpy.random.rand(D, D)
        true_sigma = numpy.diag(numpy.random.rand(D))#true_sigma_sqrt.T @ true_sigma_sqrt

        samples = numpy.random.multivariate_normal(true_mu, true_sigma, samples_per_k)

        true_mus.append(true_mu)
        true_sigmas.append(true_sigma)
        all_samples.append(samples)

    all_samples = numpy.concatenate(all_samples)
    all_samples = torch.tensor(all_samples, dtype=torch.float32)

    # all_samples  [X, D]
    ys = all_samples.repeat(K, 1, 1)  # [K, X, D]
    ys = ys.transpose(0, 1)  # [X, K, D]
    ys = ys.unsqueeze(-1)  # [X, K, D, 1]

    pis = torch.nn.Parameter(torch.zeros(K, dtype=torch.float32), requires_grad=True)
    mus = torch.nn.Parameter(torch.rand(K, D, dtype=torch.float32) * 10, requires_grad=True)
    sigmas_sqrt = torch.nn.Parameter(torch.rand(K, D, D, dtype=torch.float32), requires_grad=True)
    #sigmas_diag = torch.nn.Parameter(torch.rand(K, D, dtype=torch.float32), requires_grad=True)
    parameters = [
        pis,
        mus,
        #sigmas_diag,
        sigmas_sqrt
    ]

    optimizer = torch.optim.Adam(parameters, lr=0.5)

    i = 0
    while True:
        #print(pis)
        #print(mus)
        #print(sigmas_sqrt)
        #print(sigmas)

        optimizer.zero_grad()

        sigmas = sigmas_sqrt.transpose(-2, -1) @ sigmas_sqrt
        #sigmas = torch.diag_embed(torch.abs(sigmas_diag))

        loss = get_nll_loss(
            ys,
            pis,
            mus,
            sigmas,
        )

        # visualize
        colors = ["red", "blue", "green", "orange", "purple"] * 10
        if i % 100 == 0:
            print(loss.item())
            print(pis)
            print(mus)
            print(sigmas)
            print("----")
            
            pi_logits = torch.softmax(pis.detach(), dim=0)
            for k in range(K):
                x = numpy.linspace(-10, 10, num=100)
                y = numpy.linspace(-10, 10, num=100)
                X, Y = numpy.meshgrid(x,y)

                distr = multivariate_normal(
                    cov=sigmas.detach()[k],
                    mean=mus.detach()[k]
                )
                pdf = numpy.zeros(X.shape)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])

                plt.contour(X, Y, pdf, colors=colors[k], alpha=float(pi_logits[k]))

            plt.scatter(*all_samples.T)

            plt.show()

        # backpropagate
        loss.backward()
        optimizer.step()

        i += 1
        
