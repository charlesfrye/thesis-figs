import autograd.numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import format as fmt

EIGVAL_CUTOFF = 1e-5
COLORS = {"success": "C0",
          "failure": "C1",
          "other": "xkcd:brick"}
ANALYTICAL_CPS_COLOR = "gray"


def compute_index(spectrum, eigval_cutoff=EIGVAL_CUTOFF):
    return np.mean(spectrum < -eigval_cutoff)


def make_trajectory_panel(cp_df, failed_cp_df, ax=None,
                          include_x_label=True, include_y_label=True,
                          include_legend=False, title=None):

    if ax is None:
        f, ax = plt.subplots(figsize=(6, 6))

    plot_trajectories(cp_df, "squared_grad_norms", ax=ax,
                      plot_func_kwargs={"color": COLORS["success"], "alpha": 0.45, "lw": 3})

    if failed_cp_df is not None:
        plot_trajectories(
            failed_cp_df, "squared_grad_norms", ax=ax,
            plot_func_kwargs={"color": COLORS["failure"], "alpha": 0.45, "lw": 3})

    ax.set_yscale("log")
    ax.set_ylim([1e-30, 1e1])

    if include_x_label:
        ax.set_xlabel(r"Iterations",
                      fontsize=fmt.LABELSIZE)
    if include_y_label:
        ax.set_ylabel(r"$\vert\vert\nabla L(\theta)\vert\vert^2 $",
                      fontsize=fmt.LABELSIZE)
    if include_legend:
        legend_lines = [matplotlib.lines.Line2D([0], [0], color=COLORS["success"], lw=2),
                        matplotlib.lines.Line2D([0], [0], color=COLORS["failure"], lw=2)]

        ax.legend(legend_lines, ["successful runs", "failed runs"], loc="lower left")

    if title is not None:
        ax.set_title(title, fontsize=fmt.TITLESIZE)

    return ax


def plot_trajectories(cp_df, key, plot_func="plot",
                      func=lambda x: x, ax=None,
                      subplots_kwargs=None, plot_func_kwargs=None):

    if ax is None:
        if subplots_kwargs is None:
            subplots_kwargs = {}

        f, ax = plt.subplots(**subplots_kwargs)

    if plot_func_kwargs is None:
        plot_func_kwargs = {}

    hs = []

    for ii, row in cp_df.iterrows():
        ys = func(row[key])
        if plot_func == "plot":
            hs = ax.plot(ys, **plot_func_kwargs)

    return ax, hs


def make_loss_index_panel(cp_df, analytical_cp_df=None, ax=None,
                          color=None,
                          include_x_label=True,
                          include_y_label=True,
                          include_legend=False):

    if ax is None:
        f, ax = plt.subplots(figsize=(6, 6))

    if analytical_cp_df is not None:
        ax.scatter(
            analytical_cp_df.morse_index, analytical_cp_df.cost,
            color=ANALYTICAL_CPS_COLOR, alpha=0.65, label="analytical")

    if color is not None:
        ax.scatter(cp_df.morse_index, cp_df.candidate_loss,
                   color=color, label="numerical")
    else:
        ax.scatter(cp_df.morse_index, cp_df.candidate_loss,
                   label="numerical",
                   color=COLORS["success"])

    if include_x_label:
        ax.set_xlabel("Index", fontsize=fmt.LABELSIZE)
    if include_y_label:
        ax.set_ylabel(r"$L(\theta)$", fontsize=fmt.LABELSIZE)

    if include_legend:
        ax.legend(loc="lower right")

    return ax
