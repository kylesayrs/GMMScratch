import torch
import numpy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from loss import get_nll_loss


if __name__ == "__main__":
    K = 1
    D = 2
    samples_per_k = 500

    true_mus = []
    true_sigmas = []
    all_samples = []
    for _ in range(K):
        true_mu = numpy.random.rand(D) * 10
        true_sigma_sqrt = numpy.random.rand(D, D) * 10
        true_sigma = numpy.eye(2)#true_sigma_sqrt @ true_sigma_sqrt.T

        samples = numpy.random.multivariate_normal(true_mu, true_sigma, samples_per_k)

        true_mus.append(true_mu)
        true_sigmas.append(true_sigma)
        all_samples.append(samples)

    all_samples = numpy.concatenate(all_samples)
    all_samples = torch.tensor(all_samples, dtype=torch.float32)

    plt.scatter(*all_samples.T)
    plt.show()

    # all_samples  [X, D]
    ys = all_samples.repeat(K, 1, 1)  # [K, X, D]
    ys = ys.transpose(0, 1)  # [X, K, D]
    ys = ys.unsqueeze(-1)  # [X, K, D, 1]

    pis = torch.nn.Parameter(torch.rand(K, dtype=torch.float32), requires_grad=True)
    mus = torch.nn.Parameter(torch.rand(K, D, dtype=torch.float32), requires_grad=True)
    #sigmas_sqrt = torch.nn.Parameter(torch.rand(K, D, D, dtype=torch.float32), requires_grad=True)
    #sigmas = torch.nn.Parameter(torch.rand(K, D, dtype=torch.float32), requires_grad=True)
    parameters = [
        pis, mus#, sigmas, #sigmas_sqrt
    ]

    optimizer = torch.optim.SGD(parameters, lr=1, momentum=0.1)

    for i in range(1000):
        #print(pis)
        print(mus)
        #print(sigmas_sqrt)
        #print(sigmas)

        optimizer.zero_grad()

        loss = get_nll_loss(
            ys,
            torch.tensor([1]),#pis,
            mus,
            torch.tensor(true_sigmas, dtype=torch.float32)
            #torch.tensor([torch.diag(s) for s in sigmas])
            #sigmas_sqrt @ sigmas_sqrt.transpose(-2, -1)
        )
        #print(loss)
        print(loss.item())

        loss.backward()
        optimizer.step()




        x = numpy.linspace(-10, 10, num=100)
        y = numpy.linspace(-10, 10, num=100)
        X, Y = numpy.meshgrid(x,y)

        distr = multivariate_normal(
            cov=true_sigmas[0],
            mean=mus[0].detach()
        )
        pdf = numpy.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])

        plt.contourf(X, Y, pdf, cmap='viridis')
        plt.scatter(*all_samples.T)
        plt.show()

        print(true_mus)
        
