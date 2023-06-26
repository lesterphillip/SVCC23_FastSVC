#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Calculate statistics of feature files."""

import argparse
import logging
import os

import numpy as np
import yaml

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from joblib import dump

from harana.datasets import B02Dataset
from harana.utils import read_hdf5
from harana.utils import write_hdf5


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean and variance of dumped raw features "
            "(See detail in parallel_wavegan/bin/compute_statistics.py)."
        )
    )
    parser.add_argument(
        "--rootdir",
        type=str,
        help=(
            "directory including feature files. "
            "you need to specify either feats-scp or rootdir."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--dumpdir",
        default=None,
        type=str,
        required=True,
        help=(
            "directory to save statistics. if not provided, "
            "stats will be saved in the above root directory. (default=None)"
        ),
    )
    args = parser.parse_args()

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check arguments
    if args.rootdir is None:
        raise ValueError("Please specify --rootdir.")

    # check directory existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir)

    # get dataset
    query = "*.h5"

    dataset = B02Dataset(
        root_dir=args.rootdir,
        query=query,
        return_utt_id=True
    )
    logging.info(f"The number of files = {len(dataset)}.")

    # calculate statistics
    scaler = {}
    features = ["ppg", "mcep", "bap", "lf0", "lft", "spk_emb"]
    for feat in features:
        if feat in ["ppg", "lf0", "lft", "spk_emb"]:
            scaler[feat] = MinMaxScaler()
        else:
            scaler[feat] = StandardScaler()

    for items in tqdm(dataset):
        utt_id, audio, ppg, mcep, bap, lft, lf0, emb, df, f0 = items
        scaler["mcep"].partial_fit(mcep)
        scaler["bap"].partial_fit(bap)
        scaler["ppg"].partial_fit(ppg)
        scaler["lf0"].partial_fit(lf0)
        scaler["lft"].partial_fit(lft)
        scaler["spk_emb"].partial_fit(emb)

    # save scaler file
    dump(
        scaler,
        os.path.join(args.dumpdir, "stats.joblib")
    )
    logging.info(f"Successfully saved statistics file.")

if __name__ == "__main__":
    main()
