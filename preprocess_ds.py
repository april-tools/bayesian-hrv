import argparse
import pickle
from functools import partial
from shutil import rmtree

import pandas as pd
from tqdm import tqdm
from tqdm.contrib import concurrent

from timebase.data import preprocessing, spreadsheet, utils
from timebase.data.static import *
from timebase.utils import h5
from timebase.utils.utils import set_random_seed


def get_session_label(clinical_info: pd.DataFrame, session_id: str):
    session = clinical_info[clinical_info.Session_Code == session_id]
    if session.empty:
        return None
    else:
        values = session.values[0]
        values[LABEL_COLS.index("Session_Code")] = float(
            os.path.basename(values[LABEL_COLS.index("Session_Code")])
        )
        return values.astype(np.float32)


def preprocess_session(args, clinical_info: pd.DataFrame, session_id: str):
    recording_dir = utils.unzip_session(
        args.path2data, session_id=os.path.basename(session_id)
    )
    session_label = get_session_label(clinical_info, session_id=session_id)
    if session_label is None:
        raise ValueError(f"Cannot find session {session_id} in spreadsheet.")
    session_data, session_info, short_section = preprocessing.preprocess_dir(
        args,
        recording_dir=recording_dir,
    )
    if short_section:
        return None
    else:
        session_data["labels"] = session_label
        session_output_dir = os.path.join(args.output_dir, session_id)
        if not os.path.isdir(session_output_dir):
            os.makedirs(session_output_dir)
        filename = os.path.join(session_output_dir, "channels.h5")
        h5.write(filename=filename, content=session_data, overwrite=True)

        return session_info


def preprocess_wrapper(args, clinical_info, session_id):
    results = preprocess_session(
        args,
        clinical_info,
        session_id,
    )
    return results


def main(args):
    if not os.path.isdir(args.path2data):
        raise FileNotFoundError(f"Data not found at {args.path2data}.")
    if os.path.isdir(args.output_dir):
        if args.overwrite:
            rmtree(args.output_dir)
        else:
            raise FileExistsError(
                f"output_dir {args.output_dir} already exists. Add --overwrite "
                f" flag to overwrite the existing preprocessed data."
            )
    os.makedirs(args.output_dir)
    set_random_seed(args.seed)

    clinical_info = spreadsheet.read(args)
    args.session_codes = list(clinical_info["Session_Code"])
    clinical_info.replace({"status": DICT_STATE}, inplace=True)
    clinical_info.replace({"time": DICT_TIME}, inplace=True)
    clinical_info.replace({"time_new": DICT_TIME_NEW}, inplace=True)

    results = concurrent.process_map(
        partial(preprocess_wrapper, args, clinical_info),
        args.session_codes,
        max_workers=args.num_workers,
        chunksize=args.chunksize,
        desc="Preprocessing",
    )
    sessions_info, invalid_sessions = {}, []
    for i, session_id in tqdm(
        enumerate(args.session_codes),
        desc="Preprocessing...",
        disable=args.verbose == 0,
    ):
        # session_info = preprocess_session(
        #     args,
        #     session_id=session_id,
        #     clinical_info=clinical_info,
        # )
        session_info = results[i]

        if session_info is None:
            invalid_sessions.append(session_id)
            continue
        sessions_info[session_id] = session_info

    numeric_columns = clinical_info.select_dtypes(include=[np.number]).columns
    clinical_info[numeric_columns] = clinical_info[numeric_columns].astype(np.float32)
    with open(os.path.join(args.output_dir, "metadata.pkl"), "wb") as file:
        pickle.dump(
            {
                "clinical_info": clinical_info,
                "invalid_sessions": invalid_sessions,
                "sessions_info": sessions_info,
                "sleep_algorithm": args.sleep_algorithm,
            },
            file,
        )

    print(f"Saved processed data to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path2data",
        type=str,
        default="data/raw_data",
        help="path to directory with raw data in zip files collected and "
        "annotated in Barcelona, Hospital Clínic",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to directory to store dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing preprocessed directory",
    )
    parser.add_argument(
        "--overwrite_spreadsheet",
        action="store_true",
        help="read from timebase/data/TIMEBASE_database.xlsx",
    )
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=1234)

    # preprocessing configuration
    parser.add_argument(
        "--sleep_algorithm",
        type=str,
        default="van_hees",
        choices=["van_hees", "scripps_clinic"],
        help="algorithm used for sleep-wake detection",
    )
    parser.add_argument(
        "--wear_minimum_minutes",
        type=int,
        default=5,
        help="minimum duration (in minutes) recording periods within a session"
        "marked as on-body have to meet in order to be included in further "
        "analyses",
    )
    parser.add_argument(
        "--minimum_recorded_time",
        type=int,
        default=15,
        help="minimum duration (in minutes) a recording session has to meet in"
        " order to be considered for further analysis",
    )
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--chunksize", type=int, default=1)
    main(parser.parse_args())
