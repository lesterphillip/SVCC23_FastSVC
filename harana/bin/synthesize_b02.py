#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Waveform generation script for SVCC23 B02 model."""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm

from torch.utils.data import DataLoader
from harana.datasets import USFGANDataset
from harana.utils import load_model
from harana.utils import read_hdf5
from harana.utils.features import SignalGenerator


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description=(
            "Decode dumped features with trained HARANA toolkit."
            "(See detail in harana/bin/decode.py)."
        )
    )
    parser.add_argument(
        "--dumpdir",
        default=None,
        type=str,
        help=(
            "directory including feature files. "
            "you need to specify either feats-scp or dumpdir."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated speech.",
    )
    parser.add_argument(
        "--srcf0stats",
        default=None,
        type=str,
        help=(
            "directory containing the source f0 information. "
        ),
    )
    parser.add_argument(
        "--trgf0stats",
        default=None,
        type=str,
        help=(
            "directory containing the target f0 information. "
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint file to be loaded.",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help=(
            "yaml format configuration file. if not explicitly provided, "
            "it will be searched in the checkpoint directory. (default=None)"
        ),
    )
    parser.add_argument(
        "--normalize-before",
        default=False,
        action="store_true",
        help=(
            "whether to perform feature normalization before input to the model. if"
            " true, it assumes that the feature is de-normalized. this is useful when"
            " text2mel model and vocoder use different feature statistics."
        ),
    )
    args = parser.parse_args()

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check arguments
    if args.dumpdir is None:
        raise ValueError("Please specify either --dumpdir or --feats-scp.")

    # get dataset
    query = "*.h5"
    dataset = USFGANDataset(
        root_dir=args.dumpdir,
        hop_size=config["hop_size"],
        query=query,
        return_utt_id=True,
    )
    logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # get data processor
    signal_generator = SignalGenerator(
        sample_rate=config["sampling_rate"],
        hop_size=config["hop_size"],
        sine_amp=config["signal_generator"].get("sine_amp", 0.1),
        noise_amp=config["signal_generator"].get("noise_amp", 0.003),
        signal_types=config["signal_generator"].get("signal_types", ["sine"]),
    )

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = load_model(args.checkpoint, None, config)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    model.remove_weight_norm()
    model = model.eval().to(device)

    pad_fn = torch.nn.ReplicationPad1d(config["aux_context_window"])
    # start generation
    total_rtf = 0.0

    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for idx, (utt_id, y, aux, df, f0) in enumerate(pbar, 1):

            aux = pad_fn(torch.FloatTensor(aux).unsqueeze(0).transpose(2,1)).to(device)
            df = torch.FloatTensor(df).view(1, 1, -1).to(device)
            f0 = torch.FloatTensor(f0).transpose(2, 1).to(device)
            sine = signal_generator(f0)

            # generate
            start = time.time()
            y, s = model(sine, aux, df)
            rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            # save as PCM 16 bit wav file
            sf.write(
                os.path.join(config["outdir"], f"{utt_id}_gen.wav"),
                y.view(-1).cpu().numpy(),
                config["sampling_rate"],
                "PCM_16",
            )

    # report average RTF
    logging.info(
        f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f})."
    )


if __name__ == "__main__":
    main()
