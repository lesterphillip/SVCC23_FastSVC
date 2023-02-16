#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Decoding script for FastSVC.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/chomeyama/HN-UnifiedSourceFilterGAN
"""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm

from harana.datasets import FastSVCDataset
from harana.utils import load_model
from harana.utils import read_hdf5
from harana.utils.features import SignalGenerator
from harana.utils.features import F0Statistics


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
        "--srcf0stats",
        default=None,
        type=str,
        help=("directory containing the source f0 information. "),
    )
    parser.add_argument(
        "--trgf0stats",
        default=None,
        type=str,
        help=("directory containing the target f0 information. "),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated speech.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint file to be loaded.",
    )
    parser.add_argument(
        "--stats",
        type=str,
        required=True,
        help="stats file for normalization.",
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
        "--spk_emb_path",
        default=None,
        type=str,
        help="path to the spk embedding extractions for all speakers.",
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
    dataset = FastSVCDataset(
        args.dumpdir,
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
    model = load_model(args.checkpoint, args.stats, config)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    model.remove_weight_norm()
    model = model.eval().to(device)
    pad_fn = torch.nn.ReplicationPad1d(config["aux_context_window"])

    # start generation
    total_rtf = 0.0
    f0class = F0Statistics()

    speaker_list = config["convert_to_speakers"]
    for trgspk in speaker_list:
        logging.info(f"converting source to speaker {trgspk}")

        # load speaker embeddings
        if config["generator_params"].get("use_spk_emb"):
            trg_emb = read_hdf5(args.spk_emb_path, trgspk)
            logging.info(type(trg_emb))
            trg_emb = torch.FloatTensor(trg_emb).to(device)
        else:
            trg_emb = None

        # get F0 statistics of target speaker
        with open(f"{args.trgf0stats}/{trgspk}.yml") as file:
            trgyaml = yaml.load(file, Loader=yaml.FullLoader)
        trgstats = np.array([trgyaml["stats"]["mean"], 1])

        with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
            for idx, (utt_id, audio, f0, ppg, lft, src_emb) in enumerate(pbar, 1):
                logging.info(f"processing {utt_id}")
                ppg = torch.FloatTensor(ppg).to(device)
                spk_id = utt_id.split("_")[0]

                # get F0 statistics of source speaker
                with open(f"{args.srcf0stats}/{spk_id}.yml") as file:
                    srcyaml = yaml.load(file, Loader=yaml.FullLoader)
                srcstats = np.array([srcyaml["stats"]["mean"], 1])

                # convert F0 using mean transformation
                f0 = np.squeeze(f0, 1)
                f0 = f0class.convert(f0, srcstats, trgstats)
                f0 = np.expand_dims(f0, 1)
                f0 = torch.FloatTensor(f0).to(device)
                lft = torch.FloatTensor(lft).to(device)

                # generate waveforms
                start = time.time()
                y = model.inference(
                    ppg, f0, lft, signal_generator, pad_fn, trg_emb
                ).view(-1)
                rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
                pbar.set_postfix({"RTF": rtf})
                total_rtf += rtf

                # save as PCM 16 bit wav file
                sf.write(
                    os.path.join(config["outdir"], f"{utt_id}_{trgspk}_gen.wav"),
                    y.cpu().numpy(),
                    config["sampling_rate"],
                    "PCM_16",
                )

    # report average RTF
    logging.info(
        f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f})."
    )


if __name__ == "__main__":
    main()
