import string

import autograd.numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from . import format, tools

CMAP = "Greys"
EPS = 1e-10
EMPIRICAL_COLOR = "C0"
EXPECTED_COLOR = "C1"
PRECISION = 1e-4


def random_matrix_figure(rm, distribution_name, precision=1e-4):
    fig = plt.figure(figsize=(12, 4))

    spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
    mat_ax = fig.add_subplot(spec[0, 0])
    spec_ax = fig.add_subplot(spec[0, 1])
    cdf_ax = fig.add_subplot(spec[0, 2])
    axs = [mat_ax, spec_ax, cdf_ax]

    plot_matrix_entries(
        rm, cmap=CMAP, ax=mat_ax)

    plot_observed_spectrum(
        rm, ax=spec_ax)
    plot_expected_spectrum(
        rm, label=distribution_name, ax=spec_ax)

    plot_observed_cumulative_spectrum(
        rm, ax=cdf_ax)
    plot_expected_cumulative_spectrum(
        rm, ax=cdf_ax)

    do_labels_legend_layout(fig, axs)

    return fig, axs


def plot_matrix_entries(rm, cmap, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rm.M, cmap="Greys")
    ax.axis("off")

    return ax


def plot_expected_spectrum(rm, precision=PRECISION, label="", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    lams = rm.generate_lams(rm.max_lam + precision, precision)
    expected_spectrum = [rm.expected_spectral_density(lam) for lam in lams]

    ax.plot(lams, expected_spectrum,
            lw=format.LINEWIDTH, linestyle="--",
            label=label, color=EXPECTED_COLOR)

    ax.set_ylabel(r"$\rho$", fontdict={"size": "large"})
    ax.set_xlabel(r"$\lambda$", fontdict={"size": "large"})

    return ax


def plot_observed_spectrum(rm, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    spectrum = rm.eigvals()
    nonsingular_spectrum = spectrum[np.abs(spectrum) > EPS]
    observed_nonsingular_mass = len(nonsingular_spectrum) / len(spectrum)
    hist, edges = np.histogram(
        nonsingular_spectrum, density=True)
    hist = hist * observed_nonsingular_mass

    extended_hist = [0] + list(hist) + [0]
    extended_edges = list(edges) + [edges[-1] + EPS]

    ax.step(
        extended_edges, extended_hist,
        lw=format.LINEWIDTH,
        label="Empirical",
        color=EMPIRICAL_COLOR)

    ax.set_ylabel(r"$\rho$", fontdict={"size": "large"})
    ax.set_xlabel(r"$\lambda$", fontdict={"size": "large"})

    return ax


def do_labels_legend_layout(fig, axs, legend_ax_idx=1):
    fig.tight_layout()
    axs[legend_ax_idx].legend(loc=[0.45, -0.4], ncol=2)

    [tools.add_panel_label(letter, ax)
     for letter, ax in zip(string.ascii_uppercase, axs)]


def plot_expected_cumulative_spectrum(rm, ax=None, **plot_kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    rm.display_expected_cumulative_spectral_distribution(
            ax, lw=format.LINEWIDTH, linestyle="--", color=EXPECTED_COLOR)

    ax.set_ylabel(r"$\rho(\lambda \leq \Lambda)$", fontdict={"size": "large"})
    ax.set_xlabel(r"$\lambda$", fontdict={"size": "large"})

    return ax


def plot_observed_cumulative_spectrum(rm, ax=None, **plot_kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    spectrum = rm.eigvals()
    N = len(spectrum)
    extended_lams = np.array([rm.min_lam] + list(spectrum) + [rm.max_lam])
    extended_cds = np.divide([0] + list(range(N)) + [N], N)
    ax.step(extended_lams,
            extended_cds,
            lw=format.LINEWIDTH, color=EMPIRICAL_COLOR)

    ax.set_ylabel(r"$\rho(\lambda \leq \Lambda)$", fontdict={"size": "large"})
    ax.set_xlabel(r"$\lambda$", fontdict={"size": "large"})

    return ax
