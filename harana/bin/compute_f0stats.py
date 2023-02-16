#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Compute F0 mean and variance for conversion.
References:
    - https://github.com/k2kobayashi/sprocket
"""

import argparse
import logging
import os
import yaml

import numpy as np

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from joblib import load

from harana.datasets import FastSVCDataset
from harana.utils import read_hdf5
from harana.utils import write_hdf5
from harana.utils.features import F0Statistics


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=("Compute F0 statistics for conversion during inference")
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        help=(
            "directory including feature files to be normalized. "
            "you need to specify either *-scp or rootdir."
        ),
    )
    parser.add_argument(
        "--wav-scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file. you need to specify either *-scp or rootdir.",
    )
    parser.add_argument(
        "--feats-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file. you need to specify either *-scp or rootdir.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    args = parser.parse_args()

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # check arguments
    if args.rootdir is None:
        raise ValueError("Please specify either --rootdir or --feats-scp.")

    # check directory existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir)
    if not os.path.exists(f"{args.dumpdir}/f0_stats"):
        os.makedirs(f"{args.dumpdir}/f0_stats")

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # get dataset
    dataset = FastSVCDataset(
        root_dir=args.rootdir,
        return_utt_id=True,
    )

    f0class = F0Statistics()

    logging.info(f"The number of files = {len(dataset)}.")
    f0_dict = {}
    contf0_dict = {}

    # process each file
    for items in tqdm(dataset):
        utt_id, audio, f0, ppg, lft, emb = items
        logging.info(f"processing speaker id: {utt_id}")
        spk_id = utt_id.split("_")[0]

        # store F0 in dictionary
        if f0_dict.get(spk_id) is None:
            f0_dict[spk_id] = []
        f0_dict[spk_id].append(f0)

    # compute mean and variance, then write to config file
    for key, value in f0_dict.items():
        logging.info(f"processing f0 speaker id: {key}")
        f0list = value
        f0stats = f0class.estimate(f0list)
        f0 = {
            "spk": key,
            "stats": {"mean": float(f0stats[0]), "std": float(f0stats[1])},
        }

        # save to yaml file
        with open(os.path.join(args.dumpdir, "f0_stats", f"{key}.yml"), "w") as file:
            yaml.dump(f0, file)


if __name__ == "__main__":
    main()
