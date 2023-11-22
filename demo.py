import torch
import numpy
import argparse
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from data import sample_data
from loss.full import get_full_nll_loss
from loss.diagonal import get_diagonal_nll_loss


parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", default=500)
parser.add_argument("--num_clusters", default=2)
parser.add_argument("--num_mixtures", default=5)
parser.add_argument("--seed", type=int, default=43)


if __name__ == "__main__":
    args = parser.parse_args()
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)

    K = 2
    D = 2

    all_samples, true_mus, true_sigmas = sample_data(args.num_samples, args.num_clusters, D)

    # all_samples  [X, D]
    ys = all_samples.repeat(K, 1, 1)  # [K, X, D]
    ys = ys.transpose(0, 1)  # [X, K, D]
    ys = ys.unsqueeze(-1)  # [X, K, D, 1]

    pi_initial = torch.zeros(K, dtype=torch.float32)
    #pi_initial = torch.rand(K, dtype=torch.float32)
    #pi_initial[0] += 0.1
    pi_logits = torch.nn.Parameter(pi_initial, requires_grad=True)
    mus = torch.nn.Parameter(torch.rand(K, D, dtype=torch.float32) * 10, requires_grad=True)
    #sigmas_sqrt = torch.nn.Parameter(torch.rand(K, D, D, dtype=torch.float32), requires_grad=True)
    sigmas_diag = torch.nn.Parameter(torch.rand(K, D, dtype=torch.float32), requires_grad=True)
    parameters = [
        pi_logits,
        mus,
        sigmas_diag,
        #sigmas_sqrt
    ]

    optimizer = torch.optim.Adam(parameters, lr=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2000)

    i = 0
    while True:
        #print(pis)
        #print(mus)
        #print(sigmas_sqrt)
        #print(sigmas)

        optimizer.zero_grad()

        # activation functions
        pis = torch.softmax(pi_logits, dim=0)

        #_sigmas_sqrt = torch.exp(sigmas_sqrt + 1e-6)
        #sigmas = _sigmas_sqrt.transpose(-2, -1) @ _sigmas_sqrt
        #sigmas = torch.diag_embed(torch.abs(sigmas_diag + 1e-6))  singularity issue
        sigmas = torch.diag_embed(torch.exp(sigmas_diag + 1e-6))

        loss = get_full_nll_loss(ys, pis, mus, sigmas)
        #loss = get_diagonal_nll_loss(ys, pis, mus, torch.exp(sigmas_diag + 1e-6))

        # visualize
        colors = ["red", "blue", "green", "orange", "purple"] * 10
        if i % 1000 == 0:
            print(loss.item())
            print(pis)
            print(mus)
            print(sigmas)
            print("----")
            
            pis_detached = pis.detach()
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

                plt.contour(X, Y, pdf, colors=colors[k], alpha=float(pis_detached[k]))

            plt.scatter(*all_samples.T)

            plt.show()

        # backpropagate
        loss.backward()
        optimizer.step()
        scheduler.step()

        i += 1
        
