"""
Helper functions to preprocess CSV files
Reference on data export and formatting of Empatica E4 wristband
https://support.empatica.com/hc/en-us/articles/201608896-Data-export-and-formatting-from-E4-connect-
"""


import re
import shutil
import typing as t
import warnings
from datetime import datetime
from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import concurrent

from timebase.data import utils
from timebase.data.filter_data import scripps_clinic_algorithm
from timebase.data.filter_data import van_hees_algorithm
from timebase.data.static import *
from timebase.utils.utils import get_sequences_boundaries_index

warnings.simplefilter("error", RuntimeWarning)


def read_clinical_info(filename: str):
    """Read clinical EXCEL file"""
    assert os.path.isfile(filename), f"clinical file {filename} does not exists."
    xls = pd.ExcelFile(filename)
    info = pd.read_excel(xls, sheet_name=None)  # read all sheets
    return pd.concat(info)


def split_acceleration(
    channel_data: t.Dict[str, np.ndarray],
    sampling_rates: t.Dict[str, int],
):
    """Split 3D ACC into ACC_x, ACC_y and ACC_z"""
    channel_data["ACC_x"] = channel_data["ACC"][:, 0]
    channel_data["ACC_y"] = channel_data["ACC"][:, 1]
    channel_data["ACC_z"] = channel_data["ACC"][:, 2]
    del channel_data["ACC"]
    sampling_rates["ACC_x"] = sampling_rates["ACC"]
    sampling_rates["ACC_y"] = sampling_rates["ACC"]
    sampling_rates["ACC_z"] = sampling_rates["ACC"]
    del sampling_rates["ACC"]


def load_channel(recording_dir: str, channel: str):
    """Load channel CSV data from file
    Returns
      unix_t0: int, the start time of the recording in UNIX time
      sampling_rate: int, sampling rate of the recording (if exists)
      data: np.ndarray, the raw recording data
    """
    assert channel in CSV_CHANNELS, f"unknown channel {channel}"
    try:
        if channel == "IBI":
            raw_data = pd.read_csv(
                os.path.join(recording_dir, f"{channel}.csv"),
                delimiter=",",
            )
        else:
            raw_data = pd.read_csv(
                os.path.join(recording_dir, f"{channel}.csv"),
                delimiter=",",
                header=None,
            ).values
    except pd.errors.EmptyDataError:
        return np.nan, np.nan, np.nan

    unix_t0, sampling_rate, data = None, -1.0, None
    if channel == "IBI":
        unix_t0 = np.float64(raw_data.columns[0])
        data = raw_data.values
    else:
        unix_t0 = raw_data[0] if raw_data.ndim == 1 else raw_data[0, 0]
        sampling_rate = raw_data[1] if raw_data.ndim == 1 else raw_data[1, 0]
        data = raw_data[2:]
    assert sampling_rate.is_integer(), "sampling rate must be an integer"
    data = np.squeeze(data)
    return int(unix_t0), int(sampling_rate), data.astype(np.float32)


def preprocess_channel(recording_dir: str, channel: str):
    """
    Load and downsample channel using args.downsampling s.t. each time-step
    corresponds to one second in wall-time
    """
    assert channel in CSV_CHANNELS
    unix_t0, sampling_rate, data = load_channel(
        recording_dir=recording_dir, channel=channel
    )
    # transform to g for acceleration
    if channel == "ACC":
        data = data * 2 / 128
    if channel not in ("HR", "IBI"):
        # HR begins at t0 + 10s, remove first 10s from channels other than HR
        data = data[sampling_rate * HR_OFFSET :]
    return data, sampling_rate, unix_t0


def sleep_wake_detection(args, t0: int, session_info: t.Dict, channel_data: t.Dict):
    if (args.sleep_algorithm == "scripps_clinic") and (args.wear_minimum_minutes < 30):
        raise TypeError(
            "Scripps Clinic algorithm requires a minimum of 30 " "minutes observed"
        )
    session_info["sampling_rates"]["SLEEP"] = session_info["sampling_rates"]["ACC"]
    session_info["mask_names"].append("SLEEP")

    acc = channel_data["ACC"].copy()
    timestamps = pd.to_datetime(t0, unit="s", origin="unix") + np.arange(
        len(acc)
    ) * timedelta(seconds=session_info["sampling_rates"]["ACC"] ** -1)
    df_acc = pd.DataFrame(data=acc, columns=["acc_x", "acc_y", "acc_z"])
    df_acc = df_acc.set_index(pd.DatetimeIndex(data=timestamps, tz="UTC", name="time"))

    # Empatica E4 samples EDA at 4 Hz, ACC at 32 Hz, thus up-sample no_wear_mask
    # derived from EDA in order to align it to ACC
    upsampled_no_wear_mask = np.reshape(
        channel_data["WEAR"],
        newshape=(-1, session_info["sampling_rates"]["WEAR"]),
        order="C",
    )
    upsampled_no_wear_mask = np.repeat(
        upsampled_no_wear_mask,
        repeats=session_info["sampling_rates"]["ACC"]
        // session_info["sampling_rates"]["WEAR"],
        axis=1,
    )
    upsampled_no_wear_mask = np.reshape(upsampled_no_wear_mask, newshape=-1, order="C")
    indexes = get_sequences_boundaries_index(arr=upsampled_no_wear_mask, value=1)

    match args.sleep_algorithm:
        case "van_hees":
            sleep_wake_extractor = van_hees_algorithm
        case "scripps_clinic":
            sleep_wake_extractor = scripps_clinic_algorithm

    sleep_wake_mask = sleep_wake_extractor(
        indexes=indexes,
        df_acc=df_acc,
        timestamps=timestamps,
        acc_freq=session_info["sampling_rates"]["ACC"],
    )
    channel_data["SLEEP"] = sleep_wake_mask
    mask_labels, values = np.unique(sleep_wake_mask, return_counts=True)
    session_info["seconds_per_status"] = {
        k: (v / session_info["sampling_rates"]["SLEEP"])
        for k, v in zip(mask_labels, values)
    }


def no_wear_detection(args, t0: int, session_info: t.Dict, channel_data: t.Dict):
    # 1) Values of EDA outside the range 0.05 and 100 are considered as
    # invalid. If a sampling cycle, contains any such value the whole
    # sampling cycle is discarded
    # https://archive.arch.ethz.ch/esum/downloads/manuals/emaptics.pdf
    # https://box.empatica.com/documentation/20141119_E4_TechSpecs.pdf
    session_info["sampling_rates"]["WEAR"] = session_info["sampling_rates"]["EDA"]
    session_info["mask_names"].append("WEAR")
    eda = channel_data["EDA"].copy()
    # wear = 1, no-wear = 0
    mask = np.where(
        np.logical_or(eda < 0.05, eda > 100),
        True,
        False,
    )
    mask = np.reshape(mask, (-1, session_info["sampling_rates"]["EDA"]))
    mask = np.sum(mask, axis=1)
    mask = np.where(mask > 0, 0, 1)
    mask = np.repeat(mask, session_info["sampling_rates"]["EDA"])
    session_info["no_wear_percentage"] = np.sum(mask == 0) / len(mask)

    # 2) Valid sampling cycles are kept only if they occur as a sequence whose
    # length (in wall-time) is over a given time threshold

    timestamps = pd.to_datetime(t0, unit="s", origin="unix") + np.arange(
        len(eda)
    ) * timedelta(seconds=session_info["sampling_rates"]["EDA"] ** -1)
    indexes = get_sequences_boundaries_index(arr=mask, value=1)
    for i in indexes:
        start, stop = i[0], i[1]
        if (timestamps[stop] - timestamps[start]) < pd.Timedelta(
            minutes=args.wear_minimum_minutes
        ):
            mask[start : stop + 1] = 0

    channel_data["WEAR"] = mask


def preprocess_dir(args, recording_dir: str):
    """
    Preprocess channels in recording_dir and return the preprocessed features
    and corresponding label obtained from spreadsheet.
    Returns:
      features: np.ndarray, preprocessed channels in SAVE_CHANNELS format
    """
    durations, channel_data, sampling_rates, unix_t0s = [], {}, {}, {}
    # load and preprocess all channels except IBI
    for channel in CSV_CHANNELS:
        if channel != "IBI":
            channel_data[channel], sampling_rate, unix_t0 = preprocess_channel(
                recording_dir=recording_dir, channel=channel
            )
            durations.append(len(channel_data[channel]) // sampling_rate)
            sampling_rates[channel] = sampling_rate
            unix_t0s[channel] = unix_t0
        else:
            channel_data[channel], _, unix_t0 = preprocess_channel(
                recording_dir=recording_dir, channel=channel
            )
            if isinstance(channel_data[channel], float):
                channel_data[channel] = np.array([channel_data[channel]])
            unix_t0s[channel] = unix_t0

    session_info = {
        "channel_names": utils.get_channel_names(channel_data),
        "sampling_rates": sampling_rates,
        "unix_t0": unix_t0s,
        "mask_names": [],
    }

    # all channels should have the same durations, but as a failsafe, crop
    # each channel to the shortest duration
    min_duration = min(durations)
    short_section = min_duration < (args.minimum_recorded_time * 60)

    if short_section:
        # drop session is shorter than args.shortest_acceptable_duration minutes
        return channel_data, session_info, short_section
    else:
        for channel, recording in channel_data.items():
            if channel != "IBI":
                channel_data[channel] = recording[
                    : min_duration * sampling_rates[channel]
                ]

        # no-wear = 0, wear = 1
        # no_wear_mask is sampled at EDA frequency (4Hz)
        no_wear_detection(
            args,
            channel_data=channel_data,
            t0=unix_t0s["HR"],
            session_info=session_info,
        )
        # wake = 0, sleep = 1, can't tell = 2
        # sleep_wake_mask is sampled at ACC frequency (32Hz)
        sleep_wake_detection(
            args,
            channel_data=channel_data,
            t0=unix_t0s["HR"],
            session_info=session_info,
        )

        split_acceleration(channel_data=channel_data, sampling_rates=sampling_rates)

        return channel_data, session_info, short_section


def get_channel_from_filename(filename: str, channels: t.List):
    # Create a regular expression that matches any of the strings in `strings`.
    regex = re.compile("|".join(channels))

    # Check whether the regular expression matches the string.
    match = regex.search(filename)

    # If the regular expression matches the string, return the matched string.
    if match:
        return match.group()
    else:
        return None
