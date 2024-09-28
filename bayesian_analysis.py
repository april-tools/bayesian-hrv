import argparse
import json
import os
import random
import typing as t

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import pymc as pm
import seaborn as sns
import xarray as xr
from arviz.data.inference_data import InferenceData
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pymc.model.core import Model
from scipy import optimize
from scipy.integrate import trapz
from scipy.stats import beta, gaussian_kde, halfnorm, invgamma, iqr, norm

from timebase.data.static import *
from timebase.plot import plot
from timebase.utils.utils import generatePrime

az.style.use("arviz-darkgrid")


def kl_divergence(a: np.ndarray, b: np.ndarray):
    mu1 = np.mean(a)
    sigma1 = np.std(a, ddof=1)
    mu2 = np.mean(b)
    sigma2 = np.std(b, ddof=1)

    # Calculate the three terms of the KL divergence formula
    term1 = np.log(sigma2**2 / sigma1**2)
    term2 = (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2)
    term3 = -0.5

    # Return the KL divergence
    return term1 + term2 + term3


def compute_hdi(distribution, level=0.95):
    # For a given lower limit, we can compute the corresponding 95% interval
    def interval_width(lower):
        upper = distribution.ppf(distribution.cdf(lower) + level)
        return upper - lower

    # Find such interval which has the smallest width
    # Use equal-tailed interval as initial guess
    initial_guess = distribution.ppf((1 - level) / 2)
    optimize_result = optimize.minimize(interval_width, initial_guess)

    lower_limit = optimize_result.x[0]
    width = optimize_result.fun
    upper_limit = lower_limit + width

    return (lower_limit, upper_limit)


def prior_predictive_checks(
    args, data: pd.DataFrame, model: Model, n_samples: int = 100
):
    with model:
        prior_trace = pm.sample_prior_predictive(
            samples=n_samples, random_seed=args.seed
        )

    plot.plot_hrv(
        args,
        observed=prior_trace.observed_data[
            f"{model.name}::hrv_observed"
        ].values.ravel(),
        samples=prior_trace.prior_predictive[
            f"{model.name}::hrv_observed"
        ].values.ravel(),
        model_name=model.name,
    )

    plot.plot_hrv_vs_improvement_by_subjects(
        args,
        data=data,
        samples=np.squeeze(
            prior_trace.prior_predictive[f"{model.name}::hrv_observed"].values
        ),
        model_name=model.name,
    )

    plot.plot_hrv_on_improvement(
        args,
        data=data,
        trace=prior_trace,
        coords=model.coords,
        model_name=model.name,
    )
    return prior_trace


def posterior_predictive_checks(
    args, data: pd.DataFrame, model: Model, trace: InferenceData
):
    with model:
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    plot.plot_hrv(
        args,
        observed=trace.observed_data[f"{model.name}::hrv_observed"].values,
        samples=trace.posterior_predictive[
            f"{model.name}::hrv_observed"
        ].values.reshape(-1),
        model_name=model.name,
        samples_type="posterior",
    )
    plot.plot_hrv_vs_improvement_by_subjects(
        args,
        data=data,
        samples=trace.posterior_predictive[
            f"{model.name}::hrv_observed"
        ].values.reshape(-1, len(data)),
        model_name=model.name,
        samples_type="posterior",
    )
    plot.plot_hrv_on_improvement(
        args,
        data=data,
        trace=trace,
        coords=model.coords,
        model_name=model.name,
    )


def probability_of_direction(
    args, posterior_samples: np.ndarray, method: t.Literal["direct", "kde"] = "direct"
):
    # https://easystats.github.io/bayestestR/articles/probability_of_direction.html
    match method:
        case "direct":
            median_sign = np.sign(np.median(posterior_samples))
            same_sign_samples = posterior_samples[posterior_samples * median_sign > 0]
            prob_direction = len(same_sign_samples) / len(posterior_samples) * 100
        case "kde":
            kde = gaussian_kde(posterior_samples)
            x_range = np.linspace(
                min(posterior_samples) - iqr(posterior_samples),
                max(posterior_samples) + iqr(posterior_samples),
                10000,
            )
            pdf_values = kde(x_range)
            positive_indices = x_range > 0
            positive_pdf_values = pdf_values[positive_indices]
            positive_x_range = x_range[positive_indices]
            auc_positive_side = trapz(positive_pdf_values, positive_x_range)
            if auc_positive_side > 0.5:
                prob_direction = auc_positive_side * 100
                median_sign = 1
            else:
                prob_direction = (1 - auc_positive_side) * 100
                median_sign = 0
        case _:
            raise NotImplementedError(
                f"{method} for probability of direction estimation not implemented"
            )
    if args.verbose:
        effect_sign = "positive" if median_sign else "negative"
        print(f"Probability of {effect_sign}: {prob_direction}")
    return median_sign, prob_direction


def get_models(
    args,
    data: pd.DataFrame,
    prior: t.Union[
        t.Literal["sceptical"], t.Literal["biased2positive"], t.Literal["biased2zero"]
    ] = "sceptical",
):
    log_hrv = np.log(data["hrv_rmssd_avg"])
    diagnoses, diagnoses_levels = (data.groupby("Sub_ID").head(1)["status"]).factorize()
    subjects, sub_levels = data.Sub_ID.factorize()
    coords = {"diagnoses": diagnoses_levels, "sub_ids": sub_levels}
    age = np.array(data.groupby("Sub_ID").head(1)["age_scaled"])
    sex = np.array(data.groupby("Sub_ID").head(1)["sex"])
    onset_severity = np.array(data.groupby("Sub_ID").head(1)["onset_severity"])

    with pm.Model(coords=coords, name="two_polarities_model") as two_polarities_model:
        sub_idx = pm.MutableData("sub_idx", subjects, dims="obs_id")
        diagnosis_idx = pm.MutableData("diagnosis_idx", diagnoses, dims="sub_ids")
        meds_num = pm.MutableData("meds", data["meds"], dims="obs_id")
        symptoms_time = pm.MutableData(
            "symptoms",
            [
                data.loc[i, "YMRS_improvement"]
                if data.loc[i, "status"] == "ME"
                else data.loc[i, "HDRS_improvement"]
                for i in range(len(data))
            ],
            dims="obs_id",
        )
        onset_severity = pm.MutableData(
            "onset_severity", onset_severity, dims="sub_ids"
        )
        age = pm.MutableData("age", age, dims="sub_ids")
        sex = pm.MutableData("sex", sex, dims="sub_ids")

        # Priors
        gamma_p = pm.Uniform("gamma_p", lower=-1, upper=1, dims="diagnoses")
        alpha_0 = pm.Normal("alpha_0", mu=log_hrv.mean(), sigma=0.1)
        alpha_1 = pm.Normal("alpha_age", mu=-0.1, sigma=0.1)
        alpha_2 = pm.Normal("alpha_sex", mu=-0.1, sigma=0.1)
        alpha_3 = pm.Normal("alpha_onset_severity", mu=-0.1, sigma=0.1)

        mu_beta_0 = pm.Deterministic(
            "beta_0_mu",
            alpha_0 + alpha_1 * age + alpha_2 * sex + alpha_3 * onset_severity,
        )
        beta_0 = pm.Normal("beta_0", mu=mu_beta_0, sigma=0.5, dims="sub_ids")
        beta_1 = pm.Normal(
            "beta_impr", mu=gamma_p[diagnosis_idx], sigma=0.1, dims="sub_ids"
        )
        beta_2 = pm.Normal("beta_meds", mu=-0.1, sigma=0.1)

        # Model error
        y_sd = pm.InverseGamma("hrv_sd", alpha=3, beta=0.5, dims="sub_ids")
        # Expected value
        y_mu = pm.Deterministic(
            "hrv_mu",
            beta_0[sub_idx] + beta_1[sub_idx] * symptoms_time + beta_2 * meds_num,
        )
        y = pm.Normal(
            "hrv_observed",
            mu=y_mu,
            sigma=y_sd[sub_idx],
            observed=log_hrv,
            dims="obs_id",
        )

    del coords["diagnoses"]
    match prior:
        case "sceptical":
            name = "one_disease_model"
        case "biased2positive":
            name = "biased2positive_model"
        case "biased2zero":
            name = "biased2zero_model"
        case _:
            raise ValueError(f"{prior} Invalid prior specification for beta_1")
    with pm.Model(coords=coords, name=name) as one_disease_model:
        sub_idx = pm.MutableData("sub_idx", subjects, dims="obs_id")
        meds_num = pm.MutableData("meds", data["meds"], dims="obs_id")
        symptoms_time = pm.MutableData(
            "symptoms",
            [
                data.loc[i, "YMRS_improvement"]
                if data.loc[i, "status"] == "ME"
                else data.loc[i, "HDRS_improvement"]
                for i in range(len(data))
            ],
            dims="obs_id",
        )
        # Priors
        alpha_0 = pm.Normal("alpha_0", mu=log_hrv.mean(), sigma=0.1)
        alpha_1 = pm.Normal("alpha_age", mu=-0.1, sigma=0.1)
        alpha_2 = pm.Normal("alpha_sex", mu=-0.1, sigma=0.1)
        alpha_3 = pm.Normal("alpha_onset_severity", mu=-0.1, sigma=0.1)

        mu_beta_0 = pm.Deterministic(
            "beta_0_mu",
            alpha_0 + alpha_1 * age + alpha_2 * sex + alpha_3 * onset_severity,
        )
        beta_0 = pm.Normal("beta_0", mu=mu_beta_0, sigma=0.5, dims="sub_ids")
        match prior:
            case "sceptical":
                beta_1 = pm.Uniform("beta_impr", lower=-1, upper=1, dims="sub_ids")
            case "biased2positive":
                pre_beta_1 = pm.Beta("pre_beta_impr", alpha=5, beta=2, dims="sub_ids")
                beta_1 = pm.Deterministic("beta_impr", 1.5 * pre_beta_1 - 0.85)
            case "biased2zero":
                beta_1 = pm.Normal("beta_impr", mu=0, sigma=1, dims="sub_ids")
            case _:
                raise ValueError(f"{prior} Invalid prior specification for beta_1")

        beta_2 = pm.Normal("beta_meds", mu=-0.1, sigma=0.1)

        # Model error
        y_sd = pm.InverseGamma("hrv_sd", alpha=3, beta=0.5, dims="sub_ids")
        # Expected value
        y_mu = pm.Deterministic(
            "hrv_mu",
            beta_0[sub_idx] + beta_1[sub_idx] * symptoms_time + beta_2 * meds_num,
        )
        y = pm.Normal(
            "hrv_observed",
            mu=y_mu,
            sigma=y_sd[sub_idx],
            observed=log_hrv,
            dims="obs_id",
        )

    return two_polarities_model, one_disease_model


def load_dataset(args):
    data = pd.read_csv(os.path.join(args.working_dir, "hrv_bipolar.csv"))
    data = data[data["Sub_ID"] != 93].reset_index(drop=True)
    data = data.drop_duplicates(subset=["Sub_ID", "time"], keep="first").reset_index(
        drop=True
    )

    scores = []
    for sub_id in np.unique(data["Sub_ID"]):
        s_df = data[data["Sub_ID"] == sub_id]
        if s_df["status"].values[0] == "ME":
            score = s_df["YMRS_position"].values[0]
        else:
            score = s_df["HDRS_position"].values[0]
        score = [1 - score] * len(s_df)
        scores.extend(score)
    data["onset_severity"] = scores

    age_mean, age_std = data["age"].mean(), data["age"].std()
    data["age_scaled"] = (data["age"] - age_mean) / age_std
    data["meds"] = np.array(
        np.sum(
            data.loc[
                :,
                [
                    "Lithium",
                    "SSRI",
                    "SNRI",
                    "Tryciclics",
                    "MAOI",
                    "Other_AD",
                    "AP_1st",
                    "AP_2nd",
                    "Anticonvulsants",
                    "Beta-blockers",
                    "Opioids",
                    "Amphetamines",
                    "Antihistamines",
                    "Antiarrhythmic",
                    "Other_medication_with_anticholinergic_effects",
                    "BZD",
                ],
            ],
            axis=1,
        )
    )
    return data


def prior_checks_figure(
    args,
    data: pd.DataFrame,
    two_polarities_model: Model,
    one_disease_model: Model,
    two_polarities_trace: InferenceData,
    one_disease_trace: InferenceData,
    n_samples: int = 100,
):
    markersize, ticksize, labelsize, titlesize = 90, 14, 18, 22
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 13),
        facecolor=None,
        gridspec_kw={"wspace": 0.01, "hspace": 0.08},
        dpi=args.dpi,
    )
    titles = ["two-polarities-model", "one-disease-model"]
    sub_id = 40
    sub_id_idx = np.where(np.unique(data["Sub_ID"]) == sub_id)[0][0]
    a = one_disease_trace.prior_predictive[
        f"{one_disease_model.name}::hrv_observed"
    ].values.flatten()
    b = two_polarities_trace.prior_predictive[
        f"{two_polarities_model.name}::hrv_observed"
    ].values.flatten()
    kld = kl_divergence(a=a, b=b)
    print(f"KL(one_disease|two_polarities)={kld}")
    for idx, (trace, model_name) in enumerate(
        zip(
            [two_polarities_trace, one_disease_trace],
            [two_polarities_model.name, one_disease_model.name],
        )
    ):
        qq = [
            np.quantile(
                a=trace.prior_predictive[
                    f"{model_name}::hrv_observed"
                ].values.flatten(),
                q=q,
            )
            for q in [0.05, 0.5, 0.95]
        ]
        print(f"{model_name}: q_05={qq[0]}, q_5={qq[1]}, q_95={qq[2]}")
        axs[0, idx].hist(
            x=trace.observed_data[f"{model_name}::hrv_observed"].values,
            density=True,
            alpha=0.5,
            color="deeppink",
        )
        axs[0, idx].hist(
            x=trace.prior_predictive[f"{model_name}::hrv_observed"].values.flatten(),
            density=True,
            alpha=0.8,
            color="mediumspringgreen",
        )
        axs[0, idx].set_xlabel("lnRMSSD", fontsize=labelsize)
        axs[0, idx].set_ylabel("density", fontsize=labelsize)
        axs[0, idx].set_title(titles[idx], fontsize=titlesize, weight="bold")
        axs[0, idx].tick_params(axis="both", which="major", labelsize=ticksize)
        axs[0, idx].tick_params(axis="both", which="minor", labelsize=ticksize)
        axs[0, idx].set_yticks([])
        axs[0, idx].set_xticks([1, 2, 3, 4, 5, 6])
        axs[0, idx].set_xlim(left=0.5, right=6.5)
        legend_elements = [
            Patch(
                facecolor="deeppink",
                edgecolor="black",
                label="observed",
            ),
            Patch(
                facecolor="mediumspringgreen",
                edgecolor="black",
                label="prior-sampled",
            ),
        ]
        axs[0, idx].legend(handles=legend_elements, loc="upper right")

        #######################################################################

        position_name = (
            "YMRS_improvement"
            if list(set(data[data["Sub_ID"] == sub_id]["status"]))[0] == "ME"
            else "HDRS_improvement"
        )
        impr = xr.DataArray(np.linspace(0, 1, n_samples), dims=["plot_dim"])
        meds = xr.DataArray(
            plot._get_meds(
                I=list(data[data["Sub_ID"] == sub_id][position_name]),
                M=list(data[data["Sub_ID"] == sub_id]["meds"]),
                num_points=n_samples,
            ),
            dims=["plot_dim"],
        )

        x_true = data[data["Sub_ID"] == sub_id][position_name]
        y_true = np.log(data[data["Sub_ID"] == sub_id]["hrv_rmssd_avg"])
        y = (
            trace.prior[f"{model_name}::beta_0"][..., sub_id_idx]
            + trace.prior[f"{model_name}::beta_impr"][..., sub_id_idx] * impr
            + trace.prior[f"{model_name}::beta_meds"] * meds
        )
        sampled_indices = np.random.choice(
            y.stack(sample=("chain", "draw")).shape[1], size=n_samples, replace=False
        )
        axs[1, idx].plot(
            impr,
            y.stack(sample=("chain", "draw"))[..., sampled_indices],
            c="k",
            alpha=0.1,
            zorder=1,
        )
        axs[1, idx].scatter(
            x_true,
            y_true,
            color="red" if position_name == "YMRS_improvement" else "blue",
            marker="x",
            s=markersize,
            zorder=2,
        )
        axs[1, idx].plot(
            impr,
            np.mean(y.stack(sample=("chain", "draw")).values, axis=1),
            c="mediumspringgreen",
            zorder=2,
            linestyle="--",
        )
        axs[1, idx].legend(
            [
                Line2D([0], [0], color="mediumspringgreen", linestyle="--"),
                Line2D(
                    [0], [0], marker="x", color="red", markersize=10, linestyle="None"
                ),
            ],
            ["Predictive Mean", "Observed"],
            loc="upper left",
        )
        axs[1, idx].set_xticks([0, 0.25, 0.5, 0.75, 1])
        axs[1, idx].set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        axs[1, idx].set_yticks([2, 3, 4, 5])
        axs[1, idx].tick_params(axis="both", which="major", labelsize=ticksize)
        axs[1, idx].tick_params(axis="both", which="minor", labelsize=ticksize)
        axs[1, idx].set_xlabel(
            "Symptoms' Improvement - $I_{i=a,t}$", fontsize=labelsize
        )
        axs[1, idx].set_ylabel("lnRMSSD", fontsize=labelsize)
    fig.savefig(
        os.path.join(args.working_dir, f"prior_checks.{args.format}"), dpi=args.dpi
    )


def sensitivity2prior(
    args, data: pd.DataFrame, trace_posterior_one_disease: InferenceData
):
    _, bias2zero_model = get_models(args, data=data, prior="biased2zero")
    _, bias2positive_model = get_models(args, data=data, prior="biased2positive")
    with bias2zero_model:
        trace_posterior_bias2zero = pm.sample(
            2000,
            tune=2000,
            cores=4,
            target_accept=0.99,
            random_seed=args.seed,
            idata_kwargs={"log_likelihood": True},
        )
    with bias2positive_model:
        trace_posterior_bias2positive = pm.sample(
            2000,
            tune=2000,
            cores=4,
            target_accept=0.99,
            random_seed=args.seed,
            idata_kwargs={"log_likelihood": True},
        )
    az.compare(
        compare_dict={
            "one_disease_model": trace_posterior_one_disease,
            "bias2zero_model": trace_posterior_bias2zero,
            "bias2positive_model": trace_posterior_bias2positive,
        },
        ic="waic",
    ).to_csv(os.path.join(args.working_dir, "sensitivity2prior.csv"))
    plot.slope_figure(
        args,
        data=data,
        model=bias2zero_model,
        posterior_trace=trace_posterior_bias2zero,
    )
    effect_direction, p_direction = probability_of_direction(
        args,
        posterior_samples=trace_posterior_bias2zero.posterior[
            f"{bias2zero_model.name}::beta_impr"
        ].values.flatten(),
    )
    hdi_arr = az.hdi(
        trace_posterior_bias2zero.posterior[
            f"{bias2zero_model.name}::beta_impr"
        ].values.flatten(),
        hdi_prob=0.95,
    )
    print(
        f"{bias2positive_model.name}\n "
        f"Probability of {effect_direction} direction = {p_direction}.\n "
        f"HDI-95=[{hdi_arr[0]}-{hdi_arr[1]}]"
    )
    plot.slope_figure(
        args,
        data=data,
        model=bias2positive_model,
        posterior_trace=trace_posterior_bias2positive,
    )
    effect_direction, p_direction = probability_of_direction(
        args,
        posterior_samples=trace_posterior_bias2positive.posterior[
            f"{bias2positive_model.name}::beta_impr"
        ].values.flatten(),
    )
    hdi_arr = az.hdi(
        trace_posterior_bias2positive.posterior[
            f"{bias2positive_model.name}::beta_impr"
        ].values.flatten(),
        hdi_prob=0.95,
    )
    print(
        f"{bias2positive_model.name}\n "
        f"Probability of {effect_direction} direction = {p_direction}.\n "
        f"HDI-95=[{hdi_arr[0]}-{hdi_arr[1]}]"
    )


def run_simulation(
    args,
    data: pd.DataFrame,
    seed=int,
    num_individuals: int = 23,
    num_time_points: int = 4,
    rope: t.List = [-0.05, 0.05],
):
    random.seed(seed)
    np.random.seed(seed)
    # Generate synthetic data
    age = np.random.normal(data["age"].mean(), data["age"].std(), num_individuals)
    age = (age - age.mean()) / age.std()
    sex = np.random.choice(
        [0, 1], p=[1 - data["sex"].mean(), data["sex"].mean()], size=num_individuals
    )
    baseline_severity = np.random.uniform(
        np.min(data["onset_severity"]), 1, num_individuals
    )
    meds = np.ones((num_individuals, num_time_points))
    meds[:, 0] = np.random.choice(
        [2, 3, 4, 5, 6],
        p=len([2, 3, 4, 5, 6]) * [1 / len([2, 3, 4, 5, 6])],
        size=num_individuals,
    )
    for i in range(num_individuals):
        state = meds[i, 0]
        for t in range(1, num_time_points):
            if np.random.uniform(0, 1) > 0.975:
                if state > 0:
                    state += np.random.choice(a=[-1, 1], p=[0.5, 0.5], size=1)[0]
                else:
                    state += 1
            meds[i, t] = state
    impr = np.zeros((num_individuals, num_time_points))
    for i in range(num_individuals):
        points = np.random.choice(
            np.linspace(0.2, 1, 100), size=num_time_points - 1, replace=False
        )
        points.sort()
        impr[i, 1:] = points
    if num_time_points == 4:
        three_samples_ratio = np.sum(
            np.unique(data["Sub_ID"], return_counts=True)[1] == 3
        ) / len(np.unique(data["Sub_ID"]))
        for i, v in enumerate(
            np.random.choice(
                [False, True],
                size=num_individuals,
                p=[1 - three_samples_ratio, three_samples_ratio],
            )
        ):
            if v:
                col = np.random.choice([1, 2, 3], size=1, p=[1 / 3, 1 / 3, 1 / 3])[0]
                meds[i, col] = np.nan
                impr[i, col] = np.nan

    # Sample parameters from the assumed prior distribution
    alpha_0 = np.random.normal(np.mean(np.log(data["hrv_rmssd_avg"])), 0.1)
    alpha_1 = np.random.normal(-0.1, 0.1)
    alpha_2 = np.random.normal(-0.1, 0.1)
    alpha_3 = np.random.normal(-0.1, 0.1)
    beta_2 = np.random.normal(-0.1, 0.1)

    beta_0_i = np.random.normal(
        alpha_0 + alpha_1 * age + alpha_2 * sex + alpha_3 * baseline_severity, 0.5
    )
    beta_1_i = -1 * halfnorm.rvs(loc=0.25, scale=0.1, size=num_individuals) + 0.5

    # lnRMSSD
    sigma_i = invgamma(3, scale=0.5).rvs(size=num_individuals)

    lnRMSSD_expectation = (
        beta_0_i[:, np.newaxis] + beta_1_i[:, np.newaxis] * impr + beta_2 * meds
    )
    lnRMSSD_with_noise = lnRMSSD_expectation + np.tile(
        np.random.normal(0, sigma_i[:, np.newaxis]), num_time_points
    )
    synth_data = pd.DataFrame(
        data=np.concatenate(
            (
                lnRMSSD_with_noise.flatten(order="C")[..., np.newaxis],
                impr.flatten(order="C")[..., np.newaxis],
                meds.flatten(order="C")[..., np.newaxis],
                np.repeat(age, num_time_points)[..., np.newaxis],
                np.repeat(sex, num_time_points)[..., np.newaxis],
                np.repeat(baseline_severity, num_time_points)[..., np.newaxis],
                np.repeat(np.arange(num_individuals), num_time_points)[..., np.newaxis],
            ),
            axis=1,
        ),
        columns=[
            "log_hrv",
            "impr",
            "meds",
            "age",
            "sex",
            "baseline_severity",
            "sub_id",
        ],
    )

    synth_data = synth_data.dropna(axis=0).reset_index(drop=True)

    log_hrv = synth_data["log_hrv"]
    subjects, sub_levels = synth_data.sub_id.factorize()
    coords = {"sub_ids": sub_levels}
    age = np.array(synth_data.groupby("sub_id").head(1)["age"])
    sex = np.array(synth_data.groupby("sub_id").head(1)["sex"])
    onset_severity = np.array(synth_data.groupby("sub_id").head(1)["baseline_severity"])

    with pm.Model(coords=coords) as simulation_model:
        sub_idx = pm.MutableData("sub_idx", subjects, dims="obs_id")
        meds_num = pm.MutableData("meds", synth_data["meds"], dims="obs_id")
        symptoms_time = pm.MutableData("symptoms", synth_data["impr"], dims="obs_id")
        # Priors
        alpha_0 = pm.Normal("alpha_0", mu=log_hrv.mean(), sigma=0.1)
        alpha_1 = pm.Normal("alpha_age", mu=-0.1, sigma=0.1)
        alpha_2 = pm.Normal("alpha_sex", mu=-0.1, sigma=0.1)
        alpha_3 = pm.Normal("alpha_onset_severity", mu=-0.1, sigma=0.1)

        mu_beta_0 = pm.Deterministic(
            "beta_0_mu",
            alpha_0 + alpha_1 * age + alpha_2 * sex + alpha_3 * onset_severity,
        )
        beta_0 = pm.Normal("beta_0", mu=mu_beta_0, sigma=0.5, dims="sub_ids")
        beta_1 = pm.Uniform("beta_impr", lower=-1, upper=1, dims="sub_ids")
        beta_2 = pm.Normal("beta_meds", mu=-0.1, sigma=0.1)

        # Model error
        y_sd = pm.InverseGamma("hrv_sd", alpha=3, beta=0.5)
        # Expected value
        y_mu = pm.Deterministic(
            "hrv_mu",
            beta_0[sub_idx] + beta_1[sub_idx] * symptoms_time + beta_2 * meds_num,
        )
        y = pm.Normal(
            "hrv_observed",
            mu=y_mu,
            sigma=y_sd,
            observed=log_hrv,
            dims="obs_id",
        )
    with simulation_model:
        trace_posterior_simulation = pm.sample(
            2000,
            tune=2000,
            cores=4,
            target_accept=0.99,
            random_seed=seed,
            idata_kwargs={"log_likelihood": True},
        )
    hdi = az.hdi(
        trace_posterior_simulation.posterior["beta_impr"].values.flatten(),
        hdi_prob=0.95,
    )
    return int(hdi[0] > rope[1])


def alternative_priors(args):
    markersize, ticksize, labelsize, titlesize = 90, 18, 24, 30
    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 5),
        facecolor=None,
        gridspec_kw={"wspace": 0.01, "hspace": 0.08},
        dpi=args.dpi,
    )

    a, b = 5, 2
    samples = 1.5 * beta.rvs(a=a, b=b, size=10**6) - 0.85
    sns.kdeplot(samples, fill=False, ax=axs[0], color="blue")
    axs[0].set_xlabel("Values", fontsize=labelsize)
    axs[0].set_ylabel("Density", fontsize=labelsize)
    axs[0].set_title("(a)", fontsize=labelsize)
    axs[0].set_yticklabels([])
    axs[0].tick_params(axis="both", which="major", labelsize=ticksize)
    axs[0].set_xticks([-1, -0.5, 0, 0.5])

    loc, scale = 0, 1
    x = np.linspace(-2, 2, 1000)
    pdf = norm.pdf(x, loc, scale)
    axs[1].plot(x, pdf, color="blue")
    axs[1].set_xlabel("Values", fontsize=labelsize)
    axs[1].set_ylabel("Density", fontsize=labelsize)
    axs[1].set_title("(b)", fontsize=labelsize)
    axs[1].set_yticklabels([])
    axs[1].tick_params(axis="both", which="major", labelsize=ticksize)
    axs[1].set_xticks([-2, -1, 0, 1, 2])

    loc, scale = 0.25, 0.1
    rope = [-0.05, 0.05]
    distribution = norm(loc=loc, scale=scale)
    x = np.linspace(-3, 3, 1000)
    pdf = norm.pdf(x, loc, scale)
    axs[2].plot(x, pdf, color="blue")
    axs[2].plot(-1 * x + 0.5, pdf)
    axs[2].set_xlabel("Values", fontsize=labelsize)
    axs[2].set_ylabel("Density", fontsize=labelsize)
    axs[2].set_yticklabels([])
    axs[2].tick_params(axis="both", which="major", labelsize=ticksize)
    axs[2].set_xticks([-0.2, 0, 0.2, 0.5, 1])
    axs[2].set_xticklabels(["-0.2", "0", "0.2", "0.5", "1"])

    axs[2].set_xlim(-0.25, 1)
    # Plot ROPE segment
    rope_bottom, rope_top = rope
    axs[2].hlines(
        y=0.43,
        xmin=rope_bottom,
        xmax=rope_top,
        color="black",
        linestyle="-",
        linewidth=2,
    )
    axs[2].text(
        (rope_bottom + rope_top) / 2,
        0.45,
        "ROPE",
        fontsize=labelsize,
        color="black",
        verticalalignment="bottom",
        horizontalalignment="center",
    )
    # Plot HDI segment
    hdi_bottom, hdi_top = compute_hdi(distribution)
    axs[2].hlines(
        y=0.1, xmin=hdi_bottom, xmax=hdi_top, color="black", linestyle="-", linewidth=2
    )
    axs[2].text(
        (hdi_bottom + hdi_top) / 2,
        0.12,
        "HDI-95",
        fontsize=labelsize,
        color="black",
        verticalalignment="bottom",
        horizontalalignment="center",
    )
    axs[2].set_title("(c)", fontsize=labelsize)

    fig.savefig(
        fname=os.path.join(
            args.working_dir,
            f"sensitivity.{args.format}",
        ),
        dpi=args.dpi,
    )


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    data = load_dataset(args)
    two_polarities_model, one_disease_model = get_models(args, data=data)

    plot.plot_model_graph(args, model=two_polarities_model)
    plot.plot_model_graph(args, model=one_disease_model)
    trace_prior_two_polarities = prior_predictive_checks(
        args, data=data, model=two_polarities_model, n_samples=2000
    )
    trace_prior_one_disease = prior_predictive_checks(
        args, data=data, model=one_disease_model, n_samples=2000
    )
    prior_checks_figure(
        args,
        data=data,
        two_polarities_model=two_polarities_model,
        one_disease_model=one_disease_model,
        one_disease_trace=trace_prior_one_disease,
        two_polarities_trace=trace_prior_two_polarities,
    )
    with two_polarities_model:
        trace_posterior_two_polarities = pm.sample(
            2000,
            tune=2000,
            cores=4,
            target_accept=0.99,
            random_seed=args.seed,
            idata_kwargs={"log_likelihood": True},
        )
    with one_disease_model:
        trace_posterior_one_disease = pm.sample(
            2000,
            tune=2000,
            cores=4,
            target_accept=0.99,
            random_seed=args.seed,
            idata_kwargs={"log_likelihood": True},
        )
    posterior_predictive_checks(
        args,
        data=data,
        model=two_polarities_model,
        trace=trace_posterior_two_polarities,
    )
    posterior_predictive_checks(
        args, data=data, model=one_disease_model, trace=trace_posterior_one_disease
    )
    az.summary(trace_posterior_two_polarities).to_csv(
        os.path.join(args.working_dir, f"two_polarities_model_trace_summary.csv")
    )
    az.summary(trace_posterior_one_disease).to_csv(
        os.path.join(args.working_dir, f"one_disease_model_trace_summary.csv")
    )
    az.compare(
        compare_dict={
            "two_polarities_model": trace_posterior_two_polarities,
            "one_disease_model": trace_posterior_one_disease,
        },
        ic="waic",
    ).to_csv(os.path.join(args.working_dir, "model_comparison.csv"))

    effect_direction, p_direction = probability_of_direction(
        args,
        posterior_samples=trace_posterior_one_disease.posterior[
            f"{one_disease_model.name}::beta_impr"
        ].values.flatten(),
    )
    hdi_arr = az.hdi(
        trace_posterior_one_disease.posterior[
            f"{one_disease_model.name}::beta_impr"
        ].values.flatten(),
        hdi_prob=0.95,
    )
    print(
        f"{one_disease_model.name}\n "
        f"Probability of {effect_direction} direction = {p_direction}.\n "
        f"HDI-95=[{hdi_arr[0]}-{hdi_arr[1]}]"
    )
    plot.slope_figure(
        args,
        data=data,
        model=one_disease_model,
        posterior_trace=trace_posterior_one_disease,
    )

    ######################### SUPPLEMENTARY MATERIAL ###########################

    plot.suppl_subjects(
        args,
        data=data,
        model=one_disease_model,
        posterior_trace=trace_posterior_one_disease,
    )

    plot.suppl_covariates(
        args, posterior_trace=trace_posterior_one_disease, model=one_disease_model
    )

    sensitivity2prior(
        args, data=data, trace_posterior_one_disease=trace_posterior_one_disease
    )

    distribution = norm(loc=0.25, scale=0.1)
    print(f"HDI-95 for norm(loc=0.25, scale=0.1) is: {compute_hdi(distribution, 0.95)}")
    seeds = generatePrime(n=args.num_simulations)
    sim_res = {"23_ids": 0, "50_ids": 0, "5_ts": 0}
    sim_subs = {"23_ids": 23, "50_ids": 50, "5_ts": 23}
    sim_ts = {"23_ids": 3, "50_ids": 3, "5_ts": 5}
    for key in sim_res.keys():
        for i in range(args.num_simulations):
            res = run_simulation(
                args,
                data=data,
                num_individuals=sim_subs[key],
                num_time_points=sim_ts[key],
                seed=seeds[i],
            )
            sim_res[key] += res
            print(f"iteration no. {i}")
    sim_res = {k: v / args.num_simulations for k, v in sim_res.items()}
    print(f"percentage success: {sim_res}")
    with open(os.path.join(args.working_dir, "sensitivity.json"), "w") as file:
        json.dump(
            sim_res,
            file,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", type=str, required=True)
    parser.add_argument("--path2preprocessed", type=str, required=True)
    parser.add_argument("--seed", type=int, default=8924)
    parser.add_argument("--num_simulations", type=int, default=100)
    # matplotlib
    parser.add_argument(
        "--format", type=str, default="pdf", choices=["pdf", "png", "svg"]
    )
    parser.add_argument("--dpi", type=int, default=120)
    # misc
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    main(parser.parse_args())
