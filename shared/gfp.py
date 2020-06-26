import os

import autograd.numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from . import format as fmt

maxiter = 500

SGN_CUTOFF = 1e-8
GFP_CUTOFF = 9e-1

COLORS = {"gfp": fmt.cal_colors[1],
          "cp": fmt.cal_colors[0],
          "other": "black"}

dummy_line = lambda kwargs: matplotlib.lines.Line2D([0], [0], lw=4, **kwargs) # noqa

other_line = dummy_line({"color": COLORS["other"]})
gfp_line = dummy_line({"color": COLORS["gfp"]})
cp_line = dummy_line({"color": COLORS["cp"]})

color_legend_elements = ([gfp_line, cp_line, other_line],
                         ["Gradient-Flat", "Critical", "Other"])

EIGVAL_CUTOFF = 1e-3


def make_traj_df(traj, traj_idx=None, maxiter=None):
    traj_df = pd.DataFrame()
    for key in traj.keys():
        if key in ["theta", "mrqlp_outputs"]:
            continue
        traj_df[key] = pd.Series(traj[key])
    if traj_idx is not None:
        traj_df["traj_idx"] = traj_idx
    traj_df["step_idx"] = np.arange(len(traj_df))
    if maxiter is not None:
        traj_df = traj_df[traj_df["step_idx"] < maxiter]
    return traj_df


def make_overall_traj_df(trajs, maxiter=None):
    traj_dfs = [make_traj_df(traj, ii, maxiter=maxiter) for ii, traj in enumerate(trajs)]
    return pd.concat(traj_dfs)


def collect_trajs(cf_output_paths):
    trajs = []
    for output_path in cf_output_paths:
        traj_files = [elem for elem in os.listdir(output_path)
                      if elem.endswith(".npz")]

        for ii, traj_file in enumerate(traj_files):

            traj = np.load(output_path / traj_file, allow_pickle=True)
            trajs.append(traj)
    return trajs


def make_mrqlp_df(trajs, maxiter=None):

    mrqlp_outputs = [traj["mrqlp_outputs"][1:] for traj in trajs]

    dicts = []
    for ii, output in enumerate(mrqlp_outputs):
        for step_idx, step in enumerate(output):
            if maxiter is not None and step_idx < maxiter:
                dct = {"traj_idx": ii}
                dct.update({"flag": step[0],
                            "iters": step[1],
                            "miter": step[2],
                            "qlpiter": step[3],
                            "relres": step[4].item(),
                            "relares": step[5].item(),
                            "anorm": step[6].item(),
                            "acond": step[7].item(),
                            "xnorm": step[8],
                            "axnorm": step[9].item(),
                            "step_idx": step_idx})
                dicts.append(dct)

    mrqlp_df = pd.DataFrame(dicts)

    return mrqlp_df


def make_max_flat_point_df(joined_df):
    max_flags, max_sgns, max_relres, max_relares, max_steps = get_values_at_max(
        joined_df)
    max_flat_point_df = pd.DataFrame(
        {"relres": max_relres, "relares": max_relares, "flag": max_flags,
         "sgn": max_sgns, "traj_idx": np.arange(len(max_flags)),
         "step_idx": max_steps})

    return max_flat_point_df


def get_values_at_max(joined_df):
    max_flags, max_sgns, max_relres, max_relares, max_steps = [], [], [], [], []

    for idx, traj in joined_df.groupby(["traj_idx"]):
        unsatisfiable_rows = traj[traj.flag == 2]
        if len(unsatisfiable_rows) > 0:
            maximum_relres = unsatisfiable_rows["relres"].max()
            max_row = unsatisfiable_rows[unsatisfiable_rows["relres"] == maximum_relres].iloc[0]

            max_flags.append(max_row["flag"])
            max_sgns.append(2 * max_row["g_theta"])
            max_relres.append(max_row["relres"])
            max_relares.append(max_row["relares"])
            max_steps.append(max_row["step_idx"])
        else:
            max_flags.append(np.nan)
            max_sgns.append(np.nan)
            max_relres.append(np.nan)
            max_relares.append(np.nan)
            max_steps.append(np.nan)

    max_flags = pd.Series(max_flags)
    max_sgns = pd.Series(max_sgns)
    max_relres = pd.Series(max_relres)
    max_relares = pd.Series(max_relares)
    max_steps = pd.Series(max_steps)

    return max_flags, max_sgns, max_relres, max_relares, max_steps


def make_final_point_df(joined_df, final_iter=-1):
    final_flags, final_sgns, final_relres, final_relares = get_final_values(
        joined_df, final_iter=final_iter)
    final_point_df = pd.DataFrame(
        {"relres": final_relres, "relares": final_relares, "flag": final_flags,
         "sgn": final_sgns, "traj_idx": np.arange(len(final_flags)),
         "step_idx": -1})

    return final_point_df


def get_final_values(joined_df, final_iter=-1):

    final_flags, final_sgns, final_relres, final_relares = [], [], [], []

    for idx, traj in joined_df.groupby(["traj_idx"]):
        final_row = traj.iloc[final_iter]

        final_flags.append(final_row["flag"])
        final_sgns.append(2 * final_row["g_theta"])
        final_relres.append(final_row["relres"])
        final_relares.append(final_row["relares"])

    final_flags = pd.Series(final_flags)
    final_sgns = pd.Series(final_sgns)
    final_relres = pd.Series(final_relres)
    final_relares = pd.Series(final_relares)

    return final_flags, final_sgns, final_relres, final_relares


def get_loss_index(eigval_dict, trajs, eigval_cutoff=EIGVAL_CUTOFF):
    f_thetas, indices = [], []
    for key in eigval_dict.keys():
        eigvals = eigval_dict[key]
        traj_idx, step_idx = [int(strng) for strng in key.split("_")]

        f_theta = trajs[traj_idx]["f_theta"][step_idx]
        index = np.mean(np.less(eigvals, -eigval_cutoff))

        f_thetas.append(f_theta), indices.append(index)

    return f_thetas, indices


def get_eigval_dict(point_df, trajs, npz_path, network):
    if not os.path.exists(npz_path):
        eigval_dict = {}

        for ii, point_row in point_df.iterrows():
            traj_idx = int(point_row["traj_idx"])
            step_idx = int(point_row["step_idx"])
            theta = trajs[traj_idx]["theta"][step_idx]
            hessian = np.squeeze(network.hess(theta))
            eigvals = np.linalg.eigvalsh(hessian)
            eigval_dict[str(ii)+"_"+str(step_idx)] = eigvals

        np.savez(npz_path, **eigval_dict)
    else:
        eigval_dict = dict(np.load(npz_path))

    return eigval_dict


def make_fig_and_axes():
    f = plt.figure(figsize=(12, 12), constrained_layout=False)

    gs = f.add_gridspec(6, 4)

    sgn_ax = f.add_subplot(gs[:3, :2])
    relres_ax = f.add_subplot(gs[2, 2:])
    relres_axs = [f.add_subplot(gs[i, 2:], sharey=relres_ax) for i in range(2)]
    [ax.tick_params(direction='in', bottom=False) for ax in relres_axs]
    relres_axs = relres_axs + [relres_ax]

    ecdf_axs = [f.add_subplot(gs[3:4, :2]), f.add_subplot(gs[4:5, :2])]
    loss_index_ax = f.add_subplot(gs[3:, 2:])

    legend_ax = f.add_subplot(gs[5, :2])
    legend_ax.axis("off")

    return f, sgn_ax, relres_axs, ecdf_axs, loss_index_ax, legend_ax


def plot_gradient_norms(joined_df, final_point_df, ax=None):
    if ax is None:
        f, ax = plt.subplots(figsize=(12, 6))

    for traj_idx in range(max(final_point_df.traj_idx) + 1):

        final_point = final_point_df[final_point_df["traj_idx"] == traj_idx].iloc[0]
        traj_sub_df = joined_df[joined_df["traj_idx"] == traj_idx]

        iterations = np.arange(len(traj_sub_df))

        color = COLORS["other"]
        if final_point.flat:
            color = COLORS["gfp"]
        if final_point.critical:
            color = COLORS["cp"]

        ax.semilogy(
            iterations, np.multiply(2, traj_sub_df["g_theta"]), lw=2,
            color=color, alpha=0.7)

    ax.set_ylabel(r"$\|\|\nabla L(\theta)\|\|^2$", size=fmt.LABELSIZE)
    ax.set_xlim(1, 100)
    ax.set_xlabel("Iterations", size=fmt.LABELSIZE)

    return ax


def sgn_annotate(ax, df, idx, color, s=r"$\ast$"):
    row = df.iloc[idx]
    sgn = row["sgn"]
    ylims = np.log10(ax.get_ylim())
    y_pos = (np.log10(sgn) - ylims[0]) / (ylims[1] - ylims[0])

    ax.annotate(s, (1.01, y_pos), color=color, size="x-large",
                xycoords=ax.transAxes,
                verticalalignment="center")
    return


def plot_compare_relres(df, relres_axs, idxs, colors, rolling=10):

    for ax, idx, color in zip(relres_axs, idxs, colors):
        sub_df = df[df["traj_idx"] == idx]
        plot_relres_relares(sub_df, rolling=rolling, color=color, ax=ax, legend=False)

    relres_axs[1].set_ylabel("Gradient-Flatness", size=fmt.LABELSIZE)
    [ax.set_xticks([]) for ax in relres_axs[:-1]]
    [ax.set_xticklabels([]) for ax in relres_axs[:-1]]
    relres_axs[-1].set_xlabel("Iterations", size=fmt.LABELSIZE)


def plot_relres_relares(traj_df, color="k", legend=True, rolling=0,
                        ax=None, plot_relares=False):

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    relres, relares = traj_df["relres"], traj_df["relares"]

    if rolling > 0:

        ax.plot(
            np.arange(len(relres)), relres, lw=2, ls="-",
            color=color, label=r"$\|r\|$", alpha=0.2)
        if plot_relares:
            ax.plot(
                np.arange(len(relares)), relares, lw=2, ls="--",
                color=color, label=r"$\|Hr\|$", alpha=0.2)

        relres = relres.rolling(rolling).mean()
        relares = relares.rolling(rolling).mean()

    ax.plot(np.arange(len(relres)), relres, lw=2, ls="-", color=color, label=r"$\|r\|$")

    if plot_relares:
        ax.plot(np.arange(len(relares)), relares, lw=2, ls="--", color=color, label=r"$\|Hr\|$")

    if legend:
        ax.legend()

    return ax


def plot_relres_ecdf(point_df, gfp_cutoff=GFP_CUTOFF, sgn_cutoff=SGN_CUTOFF,
                     log_transform=False, xlabel="", ylabel="", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    relres, sgns = point_df["relres"], point_df["sgn"]

    if log_transform:
        relres = np.log10(relres)

    relres = relres.sort_values()
    other_relres = relres[relres < gfp_cutoff]
    gfp_relres = relres[relres >= gfp_cutoff]
    gfp_fraction = len(gfp_relres) / len(relres)

    ax.step(other_relres, np.linspace(0, 1 - gfp_fraction, len(other_relres)),
            lw=2, color=COLORS["other"])
    ax.hlines(1 - gfp_fraction, other_relres.iloc[-1], gfp_relres.iloc[0],
              lw=2, color=COLORS["other"])
    ax.step(gfp_relres, np.linspace(1 - gfp_fraction, 1, len(gfp_relres)),
            lw=2, color=COLORS["gfp"])
    ax.vlines(relres[sgns < sgn_cutoff], 0, 0.15, lw=3, color=COLORS["cp"])

    ax.set_ylabel(ylabel, size=fmt.LABELSIZE)
    ax.set_xlabel(xlabel)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])

    return ax


def loss_index_scatter(loss_index_pairs, loss_index_point_df, ax=None, sgn_filter=1e-4):
    if ax is None:
        _, ax = plt.subplots()

    flat_selector = loss_index_point_df["relres"].dropna() > GFP_CUTOFF
    sgn_filter = loss_index_point_df["sgn"].dropna() < sgn_filter
    critical_selector = loss_index_point_df["sgn"].dropna() < SGN_CUTOFF

    strict_flat_selector = (flat_selector & sgn_filter & ~critical_selector).dropna()
    other_selector = (~flat_selector & sgn_filter & ~critical_selector).dropna()

    selectors = [other_selector, strict_flat_selector, critical_selector]

    color_keys = ["other", "gfp", "cp"]

    for selector, color_key in zip(selectors, color_keys):
        ax.scatter(
            loss_index_pairs[1, selector],
            loss_index_pairs[0, selector],
            color=COLORS[color_key])

    ax.set_ylabel(r"$L(\theta)$", size=fmt.LABELSIZE)
    ax.set_xlabel("Index", size=fmt.LABELSIZE)
    ax.set_xlim(0, 0.5)
