import os
import typing as t

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from arviz.data.inference_data import InferenceData
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pymc.model.core import Model


def _get_meds(I, M, num_points=100):
    meds = np.zeros(num_points)

    lengths = np.diff(I + [1])
    proportions = lengths / np.sum(lengths)

    num_points_parts = np.round(proportions * num_points).astype(int)

    start_index = 0
    for value, num_points_part in zip(M, num_points_parts):
        meds[start_index : start_index + num_points_part] = value
        start_index += num_points_part

    return meds


def plot_hrv(
    args,
    observed: np.ndarray,
    samples: np.ndarray,
    model_name: str,
    samples_type: str = "prior",
):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    bin_width = 0.5
    axs[0].hist(
        x=observed,
        bins=np.arange(
            min(observed),
            max(observed) + bin_width,
            bin_width,
        ),
        density=True,
    )
    axs[1].hist(
        x=samples,
        density=True,
        bins=np.arange(
            min(samples),
            max(samples) + bin_width,
            bin_width,
        ),
    )
    axs[0].set_title("Observed HRV")
    axs[1].set_title(f"{samples_type.capitalize()} predictive over HRV")
    fig.savefig(
        fname=os.path.join(
            args.working_dir,
            f"{model_name}_{samples_type}_predictive_hrv.{args.format}",
        )
    )


def plot_hrv_on_improvement(
    args,
    data: pd.DataFrame,
    trace: InferenceData,
    model_name: str,
    coords: t.Dict,
    n_samples: int = 100,
):
    impr = xr.DataArray(np.linspace(0, 1, n_samples), dims=["plot_dim"])
    fig, axs = plt.subplots(
        nrows=5, ncols=len(coords["sub_ids"]) // 4 + 1, figsize=(35, 25)
    )
    axs = axs.flatten()
    for idx, sub_id in enumerate(coords["sub_ids"]):
        position_name = (
            "YMRS_improvement"
            if list(set(data[data["Sub_ID"] == sub_id]["status"]))[0] == "ME"
            else "HDRS_improvement"
        )
        meds = xr.DataArray(
            _get_meds(
                I=list(data[data["Sub_ID"] == sub_id][position_name]),
                M=list(data[data["Sub_ID"] == sub_id]["meds"]),
                num_points=n_samples,
            ),
            dims=["plot_dim"],
        )
        x_true = data[data["Sub_ID"] == sub_id][position_name]
        y_true = np.log(data[data["Sub_ID"] == sub_id]["hrv_rmssd_avg"])
        if hasattr(trace, "prior"):
            y = (
                trace.prior[f"{model_name}::beta_0"][..., idx]
                + trace.prior[f"{model_name}::beta_impr"][..., idx] * impr
                + trace.prior[f"{model_name}::beta_meds"] * meds
            )
            y = y.stack(sample=("chain", "draw"))
        else:
            y = (
                trace.posterior[f"{model_name}::beta_0"][..., idx]
                + trace.posterior[f"{model_name}::beta_impr"][..., idx] * impr
                + trace.posterior[f"{model_name}::beta_meds"] * meds
            )
            sampled_indices = np.random.choice(
                y.stack(sample=("chain", "draw")).shape[1],
                size=n_samples,
                replace=False,
            )
            y = y.stack(sample=("chain", "draw"))[:, sampled_indices]

        axs[idx].plot(impr, y, c="k", alpha=0.4)
        axs[idx].scatter(
            x_true,
            y_true,
            color="red" if position_name == "YMRS_improvement" else "blue",
        )
        axs[idx].set_xticks([])
        axs[idx].set_xlabel("")
        axs[idx].set_ylabel("")
        axs[idx].set_title(f"Sub_ID {sub_id}")
    if len(axs) > len(coords["sub_ids"]):
        for i in np.arange(len(coords["sub_ids"]), len(axs)):
            axs[i].remove()
    name = "prior" if hasattr(trace, "prior") else "posterior"
    fig.savefig(
        fname=os.path.join(
            args.working_dir, f"{model_name}_{name}_hrv_vs_improvement.{args.format}"
        )
    )


def plot_hrv_vs_improvement_by_subjects(
    args,
    data: pd.DataFrame,
    samples: np.ndarray,
    model_name: str,
    samples_type: str = "prior",
):
    def _compute_hdi(x, hdi_prob: float = 0.95):
        return az.hdi(x, hdi_prob=hdi_prob)

    fig, axs = plt.subplots(
        nrows=5, ncols=len(np.unique(data["Sub_ID"])) // 4 + 1, figsize=(35, 25)
    )
    axs = axs.flatten()
    for idx, sub_id in enumerate(np.unique(data["Sub_ID"])):
        position_name = (
            "YMRS_improvement"
            if list(set(data[data["Sub_ID"] == sub_id]["status"]))[0] == "ME"
            else "HDRS_improvement"
        )
        x_true = np.array(data[data["Sub_ID"] == sub_id][position_name])
        y_true = np.array(np.log(data[data["Sub_ID"] == sub_id]["hrv_rmssd_avg"]))

        sub_id_idx = np.where(data["Sub_ID"] == sub_id)[0]

        y_hat_mean = np.mean(samples[:, sub_id_idx], axis=0)
        y_hat_hdi = np.apply_along_axis(
            _compute_hdi, axis=0, arr=samples[:, sub_id_idx]
        )
        axs[idx].scatter(
            x_true,
            y_true,
            color="red" if position_name == "YMRS_improvement" else "blue",
            marker="x",
        )
        axs[idx].scatter(
            x_true,
            y_hat_mean,
            color="k",
            marker="D",
        )
        for i, x_loc in enumerate(x_true):
            y_limits = y_hat_hdi[:, i]
            axs[idx].vlines(
                x_loc,
                ymin=y_limits[0],
                ymax=y_limits[1],
                color="green",
                linestyle="-",
                alpha=0.6,
                linewidth=2,
            )

        axs[idx].set_xticks([])
        axs[idx].set_xlabel("")
        axs[idx].set_ylabel("")
        axs[idx].set_title(f"Sub_ID {sub_id}")
    if len(axs) > len(np.unique(data["Sub_ID"])):
        for i in np.arange(len(np.unique(data["Sub_ID"])), len(axs)):
            axs[i].remove()
    fig.savefig(
        fname=os.path.join(
            args.working_dir,
            f"{model_name}_{samples_type}_predictive_hrv_vs_improvement.{args.format}",
        )
    )


def plot_model_graph(args, model: Model):
    gv = pm.model_to_graphviz(model)
    gv.render(filename=os.path.join(args.working_dir, model.name), format=args.format)


def slope_figure(
    args,
    data: pd.DataFrame,
    model: Model,
    posterior_trace: InferenceData,
    n_samples: int = 100,
):
    with model:
        prior_trace = pm.sample_prior_predictive(
            samples=len(
                posterior_trace.posterior[f"{model.name}::beta_impr"].values.flatten()
            ),
            random_seed=args.seed,
        )

    markersize, ticksize, labelsize, titlesize = 100, 18, 24, 27

    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(21, 5),
        facecolor=None,
        gridspec_kw={
            "wspace": 0.05,
            "hspace": 0,
            "width_ratios": [1, 1.2, 1],
        },
        dpi=args.dpi,
    )

    axs[0].hist(
        x=prior_trace.prior[f"{model.name}::beta_impr"].values.flatten(),
        alpha=0.6,
        density=True,
        color="mediumspringgreen",
    )
    axs[0].set_xlabel(r"$\beta_{1}$", fontsize=labelsize)
    axs[0].set_ylabel("density", fontsize=labelsize)
    axs[0].tick_params(axis="both", which="major", labelsize=ticksize)
    axs[0].tick_params(axis="both", which="minor", labelsize=ticksize)
    axs[0].set_xticks([-0.3, 0, 0.3])
    axs[0].set_xticklabels(["-0.3", "0", "0.3"])
    axs[0].set_yticks([])
    axs[0].set_title("(a)", fontsize=titlesize)

    ###########################################################################

    median_value = np.median(
        posterior_trace.posterior[f"{model.name}::beta_impr"].values.flatten()
    )
    hdi = az.hdi(
        posterior_trace.posterior[f"{model.name}::beta_impr"].values.flatten(),
        hdi_prob=0.95,
    )
    rope = [-0.05, 0.05]

    axs[1].hist(
        x=posterior_trace.posterior[f"{model.name}::beta_impr"].values.flatten(),
        density=True,
        alpha=0.6,
        color="mediumspringgreen",
    )

    # Plot median line
    axs[1].axvline(median_value, color="red", linestyle="--", linewidth=4)
    axs[1].text(
        median_value + 0.1,
        plt.ylim()[1] - 0.01,
        f"Median={median_value:.03f}",
        fontsize=22,
        color="black",
        verticalalignment="bottom",
        horizontalalignment="center",
    )

    # Plot HDI segment
    hdi_bottom, hdi_top = hdi
    axs[1].hlines(
        y=0.1,
        xmin=hdi_bottom,
        xmax=hdi_top,
        color="black",
        linestyle="-",
        linewidth=2,
    )
    axs[1].text(
        (hdi_bottom + hdi_top) / 2,
        0.12,
        "HDI-95",
        fontsize=22,
        color="black",
        verticalalignment="bottom",
        horizontalalignment="center",
    )

    # Plot ROPE segment
    rope_bottom, rope_top = rope
    axs[1].hlines(
        y=0.43,
        xmin=rope_bottom,
        xmax=rope_top,
        color="black",
        linestyle="-",
        linewidth=2,
    )
    axs[1].text(
        (rope_bottom + rope_top) / 2,
        0.45,
        "ROPE",
        fontsize=22,
        color="black",
        verticalalignment="bottom",
        horizontalalignment="center",
    )

    axs[1].set_xlabel(r"$\beta_{1}$", fontsize=labelsize)
    axs[1].set_ylabel("density", fontsize=labelsize)
    axs[1].tick_params(axis="both", which="major", labelsize=ticksize)
    axs[1].tick_params(axis="both", which="minor", labelsize=ticksize)
    axs[1].set_xlim(left=-0.3, right=0.7)
    axs[1].set_xticks([-0.3, -0.2, 0, 0.2, 0.5])
    axs[1].set_xticklabels(["-0.3", "-0.2", "0", "0.2", "0.5"])
    axs[1].set_yticks([])
    axs[1].set_title("(b)", fontsize=titlesize)

    ###########################################################################

    sub_id = 40
    sub_id_idx = np.where(np.unique(data["Sub_ID"]) == sub_id)[0][0]
    position_name = (
        "YMRS_improvement"
        if list(set(data[data["Sub_ID"] == sub_id]["status"]))[0] == "ME"
        else "HDRS_improvement"
    )
    impr = xr.DataArray(np.linspace(0, 1, n_samples), dims=["plot_dim"])
    meds = xr.DataArray(
        _get_meds(
            I=list(data[data["Sub_ID"] == sub_id][position_name]),
            M=list(data[data["Sub_ID"] == sub_id]["meds"]),
            num_points=n_samples,
        ),
        dims=["plot_dim"],
    )

    x_true = data[data["Sub_ID"] == sub_id][position_name]
    y_true = np.log(data[data["Sub_ID"] == sub_id]["hrv_rmssd_avg"])
    y = (
        posterior_trace.posterior[f"{model.name}::beta_0"][..., sub_id_idx]
        + posterior_trace.posterior[f"{model.name}::beta_impr"][..., sub_id_idx] * impr
        + posterior_trace.posterior[f"{model.name}::beta_meds"] * meds
    )
    sampled_indices = np.random.choice(
        y.stack(sample=("chain", "draw")).shape[1], size=n_samples, replace=False
    )
    axs[2].plot(
        impr,
        y.stack(sample=("chain", "draw"))[:, sampled_indices],
        c="k",
        alpha=0.1,
        zorder=1,
    )
    axs[2].plot(
        impr,
        np.mean(y.stack(sample=("chain", "draw")), axis=1).values,
        c="mediumspringgreen",
        zorder=2,
        linestyle="--",
    )
    axs[2].scatter(
        x_true,
        y_true,
        color="red" if position_name == "YMRS_improvement" else "blue",
        marker="x",
        s=markersize,
        zorder=2,
    )
    axs[2].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axs[2].set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    axs[2].set_yticks([3, 3.5, 4])
    axs[2].tick_params(axis="both", which="major", labelsize=ticksize)
    axs[2].tick_params(axis="both", which="minor", labelsize=ticksize)
    axs[2].set_xlabel("Symptoms' Improvement - $I_{i=a,t}$", fontsize=labelsize)
    axs[2].set_ylabel("lnRMSSD", fontsize=labelsize)
    axs[2].set_title("(c)", fontsize=titlesize)
    axs[2].legend(
        [
            Line2D([0], [0], color="mediumspringgreen", linestyle="--"),
            Line2D([0], [0], marker="x", color="red", markersize=10, linestyle="None"),
        ],
        ["Predictive Mean", "Observed"],
        loc="upper left",
        fontsize=20,
    )
    fig.savefig(
        os.path.join(
            args.working_dir, f"{model.name}_prior_vs_posterior.{args.format}"
        ),
        dpi=args.dpi,
    )


def suppl_subjects(
    args,
    data: pd.DataFrame,
    model: Model,
    posterior_trace: InferenceData,
    n_samples: int = 100,
):
    markersize, ticksize, labelsize, titlesize = 150, 27, 40, 45
    impr = xr.DataArray(np.linspace(0, 1, n_samples), dims=["plot_dim"])
    fig, axs = plt.subplots(
        nrows=5,
        ncols=len(np.unique(data["Sub_ID"])) // 4 + 1,
        figsize=(35, 25),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )
    axs = axs.flatten()
    for idx, sub_id in enumerate(np.unique(data["Sub_ID"])):
        position_name = (
            "YMRS_improvement"
            if list(set(data[data["Sub_ID"] == sub_id]["status"]))[0] == "ME"
            else "HDRS_improvement"
        )
        meds = xr.DataArray(
            _get_meds(
                I=list(data[data["Sub_ID"] == sub_id][position_name]),
                M=list(data[data["Sub_ID"] == sub_id]["meds"]),
                num_points=n_samples,
            ),
            dims=["plot_dim"],
        )
        x_true = data[data["Sub_ID"] == sub_id][position_name]
        y_true = np.log(data[data["Sub_ID"] == sub_id]["hrv_rmssd_avg"])
        sub_id_idx = np.where(np.unique(data["Sub_ID"]) == sub_id)[0][0]
        y = (
            posterior_trace.posterior[f"{model.name}::beta_0"][..., sub_id_idx]
            + posterior_trace.posterior[f"{model.name}::beta_impr"][..., sub_id_idx]
            * impr
            # + posterior_trace.posterior[f"{model.name}::beta_meds"] * meds
        )
        sampled_indices = np.random.choice(
            y.stack(sample=("chain", "draw")).shape[1], size=n_samples, replace=False
        )
        axs[idx].plot(
            impr,
            y.stack(sample=("chain", "draw"))[:, sampled_indices],
            c="k",
            alpha=0.1,
            zorder=1,
        )
        axs[idx].plot(
            impr,
            np.mean(y.stack(sample=("chain", "draw")).values, axis=1),
            c="mediumspringgreen",
            zorder=2,
            linestyle="--",
        )
        axs[idx].scatter(
            x_true,
            y_true,
            color="red" if position_name == "YMRS_improvement" else "blue",
            marker="x",
            s=markersize,
            zorder=2,
        )
        axs[idx].set_xticks([0, 0.25, 0.5, 0.75, 1])
        axs[idx].set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        axs[idx].tick_params(axis="both", which="major", labelsize=ticksize)
        axs[idx].tick_params(axis="both", which="minor", labelsize=ticksize)
        if idx in np.arange(0, len(np.unique(data["Sub_ID"])), 6):
            axs[idx].set_ylabel("lnRMSSD", fontsize=labelsize)
        # if idx in np.arange(6 * 3, len(np.unique(data["Sub_ID"])) + 1):
        #     axs[idx].set_xlabel("Symptoms' Improvement", fontsize=labelsize)
        if sub_id == 40:
            axs[idx].set_title("subject-id = a", fontsize=titlesize)
    axs[idx + 1].scatter(
        [], [], color="red", marker="x", s=markersize, label="(Hypo)manic Episode"
    )
    axs[idx + 1].scatter(
        [], [], color="blue", marker="x", s=markersize, label="Major Depressive Episode"
    )
    axs[idx + 1].plot(
        [], [], color="mediumspringgreen", linestyle="--", label="Predictive Mean"
    )
    axs[idx + 1].legend(prop={"size": labelsize}, loc="upper left")
    axs[idx + 1].set_xticks([])
    axs[idx + 1].set_yticks([])
    axs[idx + 1].grid(False)
    axs[idx + 1].set_facecolor("white")
    if len(axs) + 1 > len(np.unique(data["Sub_ID"])):
        for i in np.arange(len(np.unique(data["Sub_ID"])) + 1, len(axs)):
            axs[i].remove()
    fig.text(
        0.5,
        0.15,
        "Symptoms' Improvement",
        horizontalalignment="center",
        fontsize=labelsize,
    )
    fig.savefig(
        fname=os.path.join(
            args.working_dir,
            f"posteriors.{args.format}",
        ),
        dpi=args.dpi,
    )

    markersize, ticksize, labelsize, titlesize = 5, 10, 15, 20
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4),
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
    )

    colors, titles = ["red", "blue"], [
        "(Hypo)manic Episode",
        "Major Depressive Episode",
    ]
    for idx, s in enumerate(["ME", "MDE_BD"]):
        indeces = [
            idx
            for idx, val in enumerate(np.unique(data["Sub_ID"]))
            if val in np.unique(data[data["status"] == s]["Sub_ID"])
        ]

        y = (
            posterior_trace.posterior[f"{model.name}::beta_0"][..., indeces]
            + posterior_trace.posterior[f"{model.name}::beta_impr"][..., indeces] * impr
            # + posterior_trace.posterior[f"{model.name}::beta_meds"] * meds
        ).values
        y = y.reshape(-1, y.shape[-1])
        hdi_limits = az.hdi(y, hdi_prob=0.95)
        expect_val = np.mean(y, axis=0)
        axs[idx].plot(
            impr,
            expect_val,
            c=colors[idx],
            zorder=2,
            linestyle="--",
        )
        axs[idx].fill_between(
            impr, hdi_limits[:, 0], hdi_limits[:, 1], alpha=0.2, color=colors[idx]
        )
        axs[idx].set_xlabel("Symptoms' Improvement", fontsize=labelsize)
        axs[idx].set_ylabel("lnRMSSD", fontsize=labelsize)
        axs[idx].set_title(titles[idx], fontsize=titlesize)
        legend_elements = [
            Patch(
                facecolor=colors[idx],
                edgecolor=colors[idx],
                label="HDI-95",
            ),
            Line2D([0], [0], color=colors[idx], linestyle="--", label="Expected Value"),
        ]
        axs[idx].legend(handles=legend_elements, loc="upper left")
    fig.savefig(
        fname=os.path.join(
            args.working_dir,
            f"posteriors_by_status.{args.format}",
        ),
        dpi=args.dpi,
    )


def suppl_covariates(args, posterior_trace: InferenceData, model: Model):
    markersize, ticksize, labelsize, titlesize = 150, 25, 28, 35
    fig, axs = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(18, 5),
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
    )

    coeffs = ["alpha_age", "alpha_sex", "alpha_onset_severity", "beta_meds"]
    covars = ["Age", "Sex", "Onset Severity", "Medications #"]
    mean = -0.1
    std_dev = -0.1
    x = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 1000)
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mean) / std_dev) ** 2
    )

    for idx, (coeff, covar) in enumerate(zip(coeffs, covars)):
        samples = posterior_trace.posterior[f"{model.name}::{coeff}"].values.flatten()
        hdi_bottom, hdi_top = az.hdi(samples, hdi_prob=0.95)
        axs[idx].hist(samples, density=True, alpha=0.5, color="gray")
        axs[idx].hlines(
            y=0.1,
            xmin=hdi_bottom,
            xmax=hdi_top,
            color="black",
            linestyle="-",
            linewidth=2,
        )
        axs[idx].text(
            (hdi_bottom + hdi_top) / 2,
            0.12,
            "HDI-95",
            fontsize=labelsize,
            color="black",
            verticalalignment="bottom",
            horizontalalignment="center",
        )
        axs[idx].plot(x, -y, color="blue", label="prior")
        axs[idx].set_title(covar, fontsize=titlesize)
        axs[idx].set_ylabel("density", fontsize=labelsize)
        axs[idx].set_yticklabels([])
        axs[idx].set_xlabel("coefficient value", fontsize=labelsize)
        axs[idx].set_xticks([1, 2, 3, 4, 5, 6])
        axs[idx].set_xlim(left=-0.5, right=0.5)
        axs[idx].set_ylim(0, 10)
        axs[idx].set_xticks([-0.45, -0.25, 0, 0.25])
        axs[idx].set_xticklabels(["-0.45", "-0.25", "0", "0.25"])
        legend_elements = [
            Patch(
                facecolor="grey",
                edgecolor="grey",
                label="posterior",
            ),
            Line2D([0], [0], color="blue", linestyle="-", label="prior"),
        ]
        axs[idx].legend(handles=legend_elements, loc="upper right")

    fig.savefig(
        os.path.join(args.working_dir, f"covars_posterior.{args.format}"),
        dpi=args.dpi,
    )
