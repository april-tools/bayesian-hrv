import argparse
import datetime
import pickle
import shutil
import typing as t

import pandas as pd
import pytz

from timebase.data.static import *
from timebase.utils import h5, utils


def _create_dict(keys, values):
    result_dict = {}
    for key, value in zip(keys, values):
        if key in result_dict:
            # If key already exists, append the value to the list
            if isinstance(result_dict[key], list):
                result_dict[key].append(value)
            else:
                result_dict[key] = [result_dict[key], value]
        else:
            # If key is new, add it to the dictionary
            result_dict[key] = value
    return result_dict


def _concat_unique_sessions(session_codes):
    unique_values = "-".join(session_codes.unique())
    return unique_values


def _first_entry(series):
    return series.iloc[0]


def _extract_monotonically_decreasing_indexes(arr):
    result_values = [arr[0]]
    result_indices = [0]

    for i in range(1, len(arr)):
        if arr[i] <= result_values[-1]:
            result_values.append(arr[i])
            result_indices.append(i)

    return np.array(result_indices)


def get_night_sleep_segment_indexes(
    args, data: t.Dict, segments_datetime_t0: np.ndarray
):
    """
    Return  sleep segments indexes
    """

    if os.path.exists(os.path.join(args.output_dir, "indexes.pkl")):
        with open(os.path.join(args.output_dir, "indexes.pkl"), "rb") as file:
            d = pickle.load(file)
    else:
        # subjects whose mood state is in ["MDE_BD", "MDE_MDD", "ME", "MX", "Eu_BD", "Eu_MDD", "HC"]
        # few subjects are seen across multiple states
        ids, states, duplicates = [], [], []
        for sub_id in np.unique(data["sessions_labels"]["Sub_ID"]):
            l = len(
                set(
                    list(
                        data["clinical_info"].loc[
                            data["clinical_info"]["Sub_ID"] == sub_id, "status"
                        ]
                    )
                ).intersection(
                    set(["MDE_BD", "MDE_MDD", "ME", "MX", "Eu_BD", "Eu_MDD", "HC"])
                )
            )
            if l:
                if l > 1:
                    duplicates.append(sub_id)
                else:
                    ids.append(sub_id)

        # from sub_ids from above get indexes for sleep segments per each T on
        # the first night's sleep or (if unavailable) the second night's sleep
        (
            indexes,
            missing_sub_ids,
            missing_rec_ts,
            missing_sessions,
            missing_nights,
            sub_ids,
            rec_ts,
            nights,
        ) = ([], [], [], [], [], [], [], [])
        for sub_id in ids:
            if sub_id in duplicates:
                bool_arr = (data["sessions_labels"]["Sub_ID"] == sub_id) & np.isin(
                    data["sessions_labels"]["status"],
                    list(
                        v
                        for k, v in DICT_STATE.items()
                        if k in ["MDE_BD", "MDE_MDD", "ME", "MX"]
                    ),
                )

            else:
                bool_arr = data["sessions_labels"]["Sub_ID"] == sub_id
            n_t = np.unique(data["sessions_labels"]["time"][np.where(bool_arr)[0]])
            for rec_t in n_t:
                earliest_date = min(
                    segments_datetime_t0[
                        np.where(
                            (bool_arr & (data["sessions_labels"]["time"] == rec_t))
                        )[0]
                    ]
                )
                # 10 pm on the first day of recording for a given T
                first_night_start_time = earliest_date.replace(
                    hour=22, minute=0, second=0
                )
                # 5 am on the following day
                first_night_end_time = (
                    earliest_date + datetime.timedelta(days=1)
                ).replace(hour=5, minute=0, second=0)
                # 10 pm on the second day of recording for a given T
                second_night_start_time = first_night_end_time.replace(
                    hour=22, minute=0, second=0
                )
                # 5 am on the following day
                second_night_end_time = (
                    second_night_start_time + datetime.timedelta(days=1)
                ).replace(hour=5, minute=0, second=0)

                idx_first = np.where(
                    (
                        bool_arr
                        & (data["sessions_labels"]["time"] == rec_t)
                        & (segments_datetime_t0 >= first_night_start_time)
                        & (segments_datetime_t0 <= first_night_end_time)
                        & (data["sessions_sleep_status"] == 1)
                    )
                )[0]
                if len(idx_first) == 0:
                    missing_session = list(
                        np.unique(
                            data["sessions_labels"]["Session_Code"][
                                np.where(
                                    (
                                        (data["sessions_labels"]["Sub_ID"] == sub_id)
                                        & (data["sessions_labels"]["time"] == rec_t)
                                    )
                                )[0]
                            ]
                        )
                    )
                    missing_sessions.append(missing_session)
                    missing_sub_ids.append(sub_id)
                    missing_rec_ts.append(rec_t)
                    missing_nights.append(first_night_start_time)
                else:
                    indexes.extend(list(idx_first))
                    sub_ids.append(sub_id)
                    rec_ts.append(rec_t)
                    nights.append(first_night_start_time)
                idx_second = np.where(
                    (
                        bool_arr
                        & (data["sessions_labels"]["time"] == rec_t)
                        & (segments_datetime_t0 >= second_night_start_time)
                        & (segments_datetime_t0 <= second_night_end_time)
                        & (data["sessions_sleep_status"] == 1)
                    )
                )[0]
                if len(idx_second) == 0:
                    missing_session = list(
                        np.unique(
                            data["sessions_labels"]["Session_Code"][
                                np.where(
                                    (
                                        (data["sessions_labels"]["Sub_ID"] == sub_id)
                                        & (data["sessions_labels"]["time"] == rec_t)
                                    )
                                )[0]
                            ]
                        )
                    )
                    missing_sessions.append(missing_session)
                    missing_sub_ids.append(sub_id)
                    missing_rec_ts.append(rec_t)
                    missing_nights.append(second_night_start_time)
                else:
                    indexes.extend(list(idx_second))
                    sub_ids.append(sub_id)
                    rec_ts.append(rec_t)
                    nights.append(second_night_start_time)
        d = {"indexes": indexes, "sub_ids": sub_ids, "rec_ts": rec_ts, "nights": nights}
        with open(os.path.join(args.output_dir, "indexes.pkl"), "wb") as file:
            pickle.dump(d, file)
        d_missing = pd.DataFrame(
            {
                "sub_id": missing_sub_ids,
                "T": missing_rec_ts,
                "session_code": missing_sessions,
                "date_night": missing_nights,
            }
        )
        with open(os.path.join(args.output_dir, "missing_data_info.pkl"), "wb") as file:
            pickle.dump(d_missing, file)

    return d


def main(args):
    utils.set_random_seed(args.seed, verbose=args.verbose)

    if args.clear_output_dir and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    filename = os.path.join(args.dataset, "metadata.pkl")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find metadata.pkl in {args.dataset}.")
    with open(filename, "rb") as file:
        data = pickle.load(file)

    data["clinical_info"] = data["clinical_info"].replace(
        {"status": {v: k for k, v in DICT_STATE.items()}}
    )

    # retrieve segments' starting time
    madrid_timezone = pytz.timezone("Europe/Madrid")
    segments_datetime_t0 = np.array(
        [
            datetime.datetime.fromtimestamp(ts, tz=pytz.utc).astimezone(madrid_timezone)
            for ts in data["segments_unix_t0"]
        ]
    )
    d = get_night_sleep_segment_indexes(
        args, data=data, segments_datetime_t0=segments_datetime_t0
    )
    df_dict = {k: v[d["indexes"]] for k, v in data["sessions_labels"].items()}
    df_dict["segments_datetime_t0"] = segments_datetime_t0[d["indexes"]]

    # get extracted features
    handcrafted_features = np.array(
        [
            h5.get(filename=filename, name="handcrafted")
            for filename in data["sessions_paths"][d["indexes"]]
        ]
    )
    handcrafted_dict = {
        k: handcrafted_features[:, i]
        for i, k in enumerate(data["handcrafted_features"])
    }
    df_dict["hrv_rmssd"] = handcrafted_dict["hrv_rmssd"]

    if args.verbose:
        missing_rate = np.isnan(df_dict["hrv_rmssd"]).sum() / len(df_dict["hrv_rmssd"])
        print(f"{missing_rate:.02f} percentage nan values in hrv features")
    missing_mask = np.isnan(df_dict["hrv_rmssd"])
    df_dict = {k: v[~missing_mask] for k, v in df_dict.items()}
    lost2nan_sessions = set(
        np.unique(df_dict["Session_Code"][np.isnan(df_dict["hrv_rmssd"])])
    ).difference(np.unique(df_dict["Session_Code"][~np.isnan(df_dict["hrv_rmssd"])]))
    if args.verbose:
        print(f"Sessions lost to nan values in hrv: {list(lost2nan_sessions)}")

    df = pd.DataFrame.from_dict(df_dict)
    if (df.shape[0] - df.dropna(axis=0).shape[0]) > 0:
        print("Missing values not just under rmssd")
    for s in np.unique(df["Sub_ID"]):
        if len(np.unique(df[df["Sub_ID"] == s]["status"])) > 1:
            raise ValueError(f"Subject ID {s} appear across multiple states")
    df["status"] = df["status"].replace({v: k for k, v in DICT_STATE.items()})

    times = pd.DataFrame(
        {
            "Sub_ID": d["sub_ids"],
            "time": d["rec_ts"],
            "nights": [t.date() for t in d["nights"]],
        }
    )
    delta = []
    for s in np.unique(times["Sub_ID"]):
        s_df = times[times["Sub_ID"] == s]
        deltas = [t - s_df.iloc[0, -1] for t in s_df.iloc[:, -1]]
        deltas = [delta.days for delta in deltas]
        delta.extend(deltas)
    times["delta"] = delta

    df["delta"] = np.ones(len(df)) * -9

    for s in np.unique(times["Sub_ID"]):
        for t in np.unique(times[times["Sub_ID"] == s]["time"]):
            for n in np.unique(
                times[(times["Sub_ID"] == s) & (times["time"] == t)]["nights"]
            ):
                start = datetime.datetime.combine(n, datetime.time(22, 0, 0))
                end = (start + datetime.timedelta(days=1)).replace(
                    hour=6, minute=0, second=0
                )
                idx = np.where(
                    (df["segments_datetime_t0"] >= madrid_timezone.localize(start))
                    & (df["segments_datetime_t0"] <= madrid_timezone.localize(end))
                    & (df["Sub_ID"] == s)
                )[0]
                value = times[
                    (times["Sub_ID"] == s)
                    & (times["time"] == t)
                    & (times["nights"] == n)
                ]["delta"].values[0]
                df.loc[idx, ["delta"]] = value
    df = df[df["delta"] != -9].reset_index(drop=True)
    df["Session_Code"] = df["Session_Code"].astype(str)
    df.rename(columns={"delta": "days"}, inplace=True)
    df["days"] = df["days"].astype(int)
    df.drop("segments_datetime_t0", axis=1, inplace=True)

    aggregated_df = (
        df.groupby(["Sub_ID", "days"])
        .agg(
            hrv_rmssd_avg=("hrv_rmssd", "mean"),
            hrv_rmssd_median=("hrv_rmssd", "median"),
            Session_Codes=("Session_Code", _concat_unique_sessions),
            **{
                col: (col, _first_entry)
                for col in df.columns
                if col
                not in [
                    "Sub_ID",
                    "days",
                    "hrv_rmssd",
                    "Session_Code",
                ]
            },
        )
        .reset_index()
    )
    container = []
    for sub_id in np.unique(df["Sub_ID"]):
        for t in np.unique(df[df["Sub_ID"] == sub_id]["time"]):
            v, c = np.unique(
                df[(df["Sub_ID"] == sub_id) & (df["time"] == t)]["days"],
                return_counts=True,
            )
            d = v[np.argmax(c)]
            sub_df = df[
                (df["Sub_ID"] == sub_id) & (df["time"] == t) & (df["days"] == d)
            ].values
            container.append(sub_df)
    times_df = pd.DataFrame(data=np.concatenate(container, axis=0), columns=df.columns)
    times_df.drop("days", axis=1, inplace=True)
    aggregated_df_longest_night = (
        times_df.groupby(["Sub_ID", "time"])
        .agg(
            hrv_rmssd_avg_1_night=("hrv_rmssd", "mean"),
            hrv_rmssd_median_1_night=("hrv_rmssd", "median"),
            Session_Codes=("Session_Code", _concat_unique_sessions),
            **{
                col: (col, _first_entry)
                for col in times_df.columns
                if col
                not in [
                    "Sub_ID",
                    "time",
                    "hrv_rmssd",
                    "Session_Code",
                ]
            },
        )
        .reset_index()
    )
    aggregated_df = pd.merge(
        aggregated_df,
        aggregated_df_longest_night.loc[
            :,
            [
                "Sub_ID",
                "time",
                "hrv_rmssd_avg_1_night",
                "hrv_rmssd_median_1_night",
            ],
        ],
        on=["Sub_ID", "time"],
    )
    sub_ids, times, ymrs_impr, hdrs_impr = [], [], [], []
    for sub_id in np.unique(aggregated_df["Sub_ID"]):
        ts = np.unique(aggregated_df[aggregated_df["Sub_ID"] == sub_id]["time"])
        t0 = ts[0]
        score_t0_ymrs = aggregated_df[
            (aggregated_df["Sub_ID"] == sub_id) & (aggregated_df["time"] == t0)
        ]["YMRS_SUM"].to_numpy()[0]
        score_t0_hdrs = aggregated_df[
            (aggregated_df["Sub_ID"] == sub_id) & (aggregated_df["time"] == t0)
        ]["HDRS_SUM"].to_numpy()[0]
        for t in ts:
            score_t_ymrs = aggregated_df[
                (aggregated_df["Sub_ID"] == sub_id) & (aggregated_df["time"] == t)
            ]["YMRS_SUM"]
            percentage_change_y = (score_t0_ymrs - score_t_ymrs) / score_t0_ymrs
            ymrs_impr.append(list(percentage_change_y)[0])

            score_t_hdrs = aggregated_df[
                (aggregated_df["Sub_ID"] == sub_id) & (aggregated_df["time"] == t)
            ]["HDRS_SUM"]
            percentage_change_h = (score_t0_hdrs - score_t_hdrs) / score_t0_hdrs
            hdrs_impr.append(list(percentage_change_h)[0])
            times.append(t)
            sub_ids.append(sub_id)
    ymrs_impr = [np.nan if not np.isfinite(x) else x for x in ymrs_impr]
    hdrs_impr = [np.nan if not np.isfinite(x) else x for x in hdrs_impr]
    improvement = pd.DataFrame(
        data={
            "Sub_ID": sub_ids,
            "time": times,
            "YMRS_improvement": ymrs_impr,
            "HDRS_improvement": hdrs_impr,
        }
    )
    aggregated_df = pd.merge(aggregated_df, improvement, on=["Sub_ID", "time"])
    aggregated_df["YMRS_position"] = (60 - aggregated_df["YMRS_SUM"]) / 60
    aggregated_df["HDRS_position"] = (52 - aggregated_df["HDRS_SUM"]) / 52

    ids = []
    for sub_id in np.unique(
        aggregated_df[aggregated_df["status"].isin(["MDE_BD", "ME"])]["Sub_ID"]
    ):
        if (
            len(np.unique(aggregated_df[aggregated_df["Sub_ID"] == sub_id]["time"]))
            >= 3
        ):
            ids.append(sub_id)
    bipolar_df = aggregated_df[aggregated_df["Sub_ID"].isin(ids)].reset_index(drop=True)
    bipolar_df.drop(
        columns=["hrv_rmssd_median"],
        inplace=True,
    )

    bipolar_df = (
        bipolar_df.groupby(["Sub_ID", "time"])
        .agg(
            hrv_rmssd_avg=("hrv_rmssd_avg", _first_entry),
            **{
                col: (col, _first_entry)
                for col in bipolar_df.columns
                if col
                not in [
                    "Sub_ID",
                    "time",
                    "days",
                    "hrv_rmssd_avg",
                    "Session_Code",
                ]
            },
        )
        .reset_index()
    )

    # Only monotonic improvement allowed
    ids, stati = [], []
    for sub_id in np.unique(bipolar_df["Sub_ID"]):
        s = list(set(bipolar_df[bipolar_df["Sub_ID"] == sub_id]["status"]))[0]
        scale_name = "YMRS_SUM" if s == "ME" else "HDRS_SUM"
        scores = np.array(bipolar_df[bipolar_df["Sub_ID"] == sub_id][scale_name])
        if all(scores[i] > scores[i + 1] for i in range(len(scores) - 1)):
            ids.append(sub_id)
            stati.append(s)

    # Recover subjects with monotonic improvement upon dropping one out of
    # four measurements
    container = []
    for sub_id in set(np.unique(bipolar_df["Sub_ID"])).difference(ids):
        if len(np.unique(bipolar_df[bipolar_df["Sub_ID"] == sub_id]["time"])) > 3:
            s = list(set(bipolar_df[bipolar_df["Sub_ID"] == sub_id]["status"]))[0]
            scale_name = "YMRS_SUM" if s == "ME" else "HDRS_SUM"
            id_df = bipolar_df[bipolar_df["Sub_ID"] == sub_id]
            scores = np.array(id_df[scale_name])
            indices = _extract_monotonically_decreasing_indexes(scores)
            if len(indices) > 2:
                container.append(id_df.values[indices])
    bipolar_df = pd.DataFrame(
        data=np.concatenate(
            (
                bipolar_df[bipolar_df["Sub_ID"].isin(ids)].values,
                np.concatenate(container, axis=0),
            ),
            axis=0,
        ),
        columns=bipolar_df.columns,
    )
    bipolar_df = bipolar_df.sort_values(by=["Sub_ID", "time"]).reset_index(drop=True)
    bipolar_df.to_csv(os.path.join(args.output_dir, "hrv_bipolar.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--clear_output_dir", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to directory where preprocessed data are stored",
    )

    main(parser.parse_args())
