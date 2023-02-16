#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Normalize feature files and store them to dump.
References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
"""

import argparse
import logging
import os

import numpy as np
import yaml

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from joblib import load

from harana.datasets import FastSVCDataset
from harana.utils import read_hdf5
from harana.utils import write_hdf5


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Normalize dumped raw features (See detail in"
            " parallel_wavegan/bin/normalize.py)."
        )
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
        "--segments",
        default=None,
        type=str,
        help="kaldi-style segments file.",
    )
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump normalized feature files.",
    )
    parser.add_argument(
        "--stats",
        type=str,
        required=True,
        help="statistics file.",
    )
    parser.add_argument(
        "--skip-wav-copy",
        default=False,
        action="store_true",
        help="whether to skip the copy of wav files.",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
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
        raise ValueError("Please specify either --rootdir or --feats-scp.")

    # check directory existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir)

    # get dataset
    dataset = FastSVCDataset(
        root_dir=args.rootdir,
        return_utt_id=True,
    )
    logging.info(f"The number of files = {len(dataset)}.")

    # load scaler
    scaler = load(args.stats)

    # process each file
    for items in tqdm(dataset):
        utt_id, audio, f0, ppg, lft, emb = items

        # normalize and save features to h5 files
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "f0",
            f0.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "ppg",
            scaler["ppg"].transform(ppg).astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "lft",
            lft.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "spk_emb",
            emb.astype(np.float32),
        )

        # write audio file
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "wave",
            audio.astype(np.float32),
        )


if __name__ == "__main__":
    main()
