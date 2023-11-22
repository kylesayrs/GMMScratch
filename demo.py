import torch
import numpy
import argparse
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from data import sample_data
from loss.full import get_full_nll_loss
from loss.diagonal import get_diagonal_nll_loss
from MixtureFamily import MixtureFamily, FAMILY_NAMES, get_mixture_family_from_str
from model import get_model


parser = argparse.ArgumentParser()
parser.add_argument("--samples", default=500)
parser.add_argument("--clusters", default=2)
parser.add_argument("--mixtures", default=5)
parser.add_argument("--dims", default=2)
parser.add_argument("--family", type=str, default="diagonal", choices=FAMILY_NAMES.keys())
parser.add_argument("--seed", type=int, default=43)


if __name__ == "__main__":
    args = parser.parse_args()
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)

    mixture_family = get_mixture_family_from_str(args.family)

    all_samples, true_mus, true_sigmas = sample_data(
        args.samples, args.clusters, args.dims, mixture_family
    )

    # all_samples  [X, D]
    ys = all_samples.repeat(args.mixtures, 1, 1)  # [K, X, D]
    ys = ys.transpose(0, 1)  # [X, K, D]

    # set up model
    model = get_model(mixture_family, args.mixtures, args.dims)

    optimizer = torch.optim.Adam(model.values(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2000)

    i = 0
    while True:
        optimizer.zero_grad()

        if mixture_family == MixtureFamily.FULL:
            sigmas = model["sigmas_sqrt"].transpose(-2, -1) @ model["sigmas_sqrt"]
            loss = get_full_nll_loss(ys, model["pi_logits"], model["mus"], sigmas)
        else:
            sigmas = torch.diag_embed(torch.abs(model["sigmas_diag"]))
            loss = get_diagonal_nll_loss(ys, model["pi_logits"], model["mus"], model["sigmas_diag"])

        # visualize
        colors = ["red", "blue", "green", "orange", "purple"] * 10
        if i % 1000 == 0:
            print(loss.item())
            print(model["pi_logits"])
            print(model["mus"])
            print(sigmas)
            print("----")
            
            pis_detached = torch.softmax(model["pi_logits"], dim=0).detach()
            for k in range(args.mixtures):
                x = numpy.linspace(-10, 10, num=100)
                y = numpy.linspace(-10, 10, num=100)
                X, Y = numpy.meshgrid(x,y)

                distr = multivariate_normal(
                    cov=sigmas.detach()[k],
                    mean=model["mus"].detach()[k]
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
        
