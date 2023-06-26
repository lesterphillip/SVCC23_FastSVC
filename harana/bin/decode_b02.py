#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Decoding script for SVCC23 B02 model.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
"""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml
import pyworld
import pysptk

import matplotlib.pyplot as plt
import librosa
import librosa.display

from tqdm import tqdm
from joblib import load
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader
from harana.datasets import Taco2Dataset
from harana.utils import load_model
from harana.utils import read_hdf5, write_hdf5, validate_length
from harana.utils.features import F0Statistics
from harana.bin.preprocess_b02 import interp1d


_c4_hz = 440 * 2 ** (3 / 12 - 1)
_c4_cent = 4800


def _colorbar_wrap(fig, mesh, ax, format="%+2.f dB"):
    try:
        fig.colorbar(mesh, ax=ax, format=format)
    except IndexError as e:
        # In _quantile_ureduce_func:
        # IndexError: index -1 is out of bounds for axis 0 with size 0
        print(str(e))


class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(self):
        """Initialize customized collater for PyTorch DataLoader.

        Args:

        """
        pass

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where
                T = (T' - 2 * aux_context_window) * hop_size.
            Tensor: Target signal batch (B, 1, T).

        """
        sorted_batch = sorted(batch, key=lambda x: -x[1].shape[0])
        bs = len(sorted_batch)

        utt_id_batch = [sorted_batch[i][0] for i in range(bs)]
        input_batch_nopad = [torch.from_numpy(sorted_batch[i][1]) for i in range(bs)]
        input_batch = pad_sequence(input_batch_nopad, batch_first=True)
        input_lengths = torch.from_numpy(np.array([i.size(0) for i in input_batch_nopad]))

        input_lft_nopad = [torch.from_numpy(sorted_batch[i][3]) for i in range(bs)]
        input_lft = pad_sequence(input_lft_nopad, batch_first=True)

        input_logf0_nopad = [torch.from_numpy(sorted_batch[i][4]) for i in range(bs)]
        input_logf0 = pad_sequence(input_logf0_nopad, batch_first=True)
        spk_embs = torch.from_numpy(np.array([sorted_batch[i][5] for i in range(bs)]))

        wave_batch = [sorted_batch[i][6] for i in range(bs)]
        input_f0_nopad = [torch.from_numpy(sorted_batch[i][7]) for i in range(bs)]
        input_f0 = pad_sequence(input_f0_nopad, batch_first=True)

        output_batch_nopad = [torch.from_numpy(sorted_batch[i][2]) for i in range(bs)]
        output_lengths = torch.from_numpy(np.array([o.size(0) for o in output_batch_nopad]))
        output_batch = pad_sequence(output_batch_nopad, batch_first=True)

        return utt_id_batch, (input_batch, input_lengths, input_lft, input_logf0, spk_embs), (input_f0, wave_batch, output_batch)


def transform_f0(f0, f0class, srcstats, trgstats, device, f0_type):
    f0 = f0.cpu().numpy().squeeze(0)
    if f0.shape[-1] == 1:
        f0 = np.squeeze(f0, 1)
    #f0 = f0class.convert(f0, srcstats, trgstats, f0_type)
    #return torch.FloatTensor(f0).to(device)
    return f0class.convert(f0, srcstats, trgstats, f0_type)


def variance_scaling(gv, feats, offset=2, note_frame_indices=None):
    """Variance scaling method to enhance synthetic speech quality
    Method proposed in :cite:t:`silen2012ways`.
    Args:
        gv (tensor): global variance computed over training data
        feats (tensor): input features
        offset (int): offset
        note_frame_indices (tensor): indices of note frames
    Returns:
        tensor: scaled features
    """
    if note_frame_indices is not None:
        utt_gv = feats[note_frame_indices].var(0)
        utt_mu = feats[note_frame_indices].mean(0)
    else:
        utt_gv = feats.var(0)
        utt_mu = feats.mean(0)

    out = feats.copy()
    if note_frame_indices is not None:
        out[note_frame_indices, offset:] = (
            np.sqrt(gv[offset:] / utt_gv[offset:])
            * (feats[note_frame_indices, offset:] - utt_mu[offset:])
            + utt_mu[offset:]
        )
    else:
        out[:, offset:] = (
            np.sqrt(gv[offset:] / utt_gv[offset:])
            * (feats[:, offset:] - utt_mu[offset:])
            + utt_mu[offset:]
        )

    return out


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
        "--f0stats",
        default=None,
        type=str,
        help=(
            "directory containing the f0 information. "
        ),
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
        "--mode",
        default="generator",
        help="which split of the dataset to create the h5 files.",
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

    with open(args.f0stats, 'r') as file:
        f0_file = yaml.load(file, Loader=yaml.FullLoader)

    # query functions for loading the dataset
    query = "*.h5"

    dataset = Taco2Dataset(
        root_dir=args.dumpdir,
        query=query,
        return_utt_id=True,
    )
    logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    scaler = load(args.stats)
    config["stats"] = {
        "mean": torch.from_numpy(scaler["mcep"].mean_).float().to(device),
        "scale": torch.from_numpy(scaler["mcep"].scale_).float().to(device),
    }

    collater = Collater()

    data_loader = DataLoader(
        dataset=dataset,
        collate_fn=collater,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = load_model(args.checkpoint)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    scaler = load(args.stats)

    model = model.eval().to(device)

    hdf5_save_dir = config["outdir"]

    f0class = F0Statistics()
    speaker_list = config["convert_to_speakers"]
    for trgspk in speaker_list:
        logging.info(f"converting source to speaker {trgspk}")

        # load speaker embeddings
        if config["generator_params"].get("multi_speaker"):
            spk_emb = read_hdf5(args.spk_emb_path, trgspk)
            spk_emb = torch.FloatTensor(spk_emb).unsqueeze(0).transpose(2, 1).to(device)
        else:
            spk_emb = None

        with torch.no_grad(), tqdm(data_loader, desc="[decode]") as pbar:
            for idx, (utt_id, x, y) in enumerate(pbar, 1):
                x = tuple([x_.to(device) for x_ in x])
                inputs, ilens, lft, logf0, _ = x
                f0, wave, real_outputs = y

                utt_id = utt_id[0]
                spk_id = utt_id.split("_")[0]

                # convert f0
                tgt_lf0_mean = f0_file[trgspk]["lf0_mean"]
                tgt_lf0_scale = f0_file[trgspk]["lf0_scale"]
                src_lf0_mean = f0_file[spk_id]["lf0_mean"]
                src_lf0_scale = f0_file[spk_id]["lf0_scale"]

                src_mean_cent = 1200 * np.log(np.exp(src_lf0_mean) / _c4_hz) / np.log(2) +_c4_cent
                tgt_mean_cent = 1200 * np.log(np.exp(tgt_lf0_mean) / _c4_hz) / np.log(2) +_c4_cent

                # round in semi-tone
                f0_shift_in_cent = round((tgt_mean_cent - src_mean_cent) / 100)
                logging.info(f"f0 shift: {f0_shift_in_cent}")

                logf0 = logf0 * 2 ** (f0_shift_in_cent / 12)
                f0 = f0 * 2 ** (f0_shift_in_cent / 12)

                predictions, lengths = model(inputs, ilens, lft, logf0, spk_emb, None)
                lengths = lengths.cpu().item()
                hdf5_save_path = os.path.join(hdf5_save_dir, (f"{utt_id}_{trgspk}.h5"))

                # generate h5 files
                mcep, bap = torch.split(predictions, [60, 3], dim=2)
                mcep = mcep.cpu().numpy().squeeze(0)
                bap = bap.cpu().numpy().squeeze(0)

                logging.info("after")
                logging.info(logf0.shape)
                logging.info(inputs.shape)
                logging.info(lft.shape)
                write_hdf5(hdf5_save_path, "mcep", mcep[:lengths].tolist())
                write_hdf5(hdf5_save_path, "bap", bap[:lengths].tolist())
                write_hdf5(hdf5_save_path, "lf0", logf0[:lengths].cpu().numpy().tolist())
                write_hdf5(hdf5_save_path, "lft", lft[:lengths].tolist())
                write_hdf5(hdf5_save_path, "ppg", inputs.squeeze(0)[:lengths].tolist())
                write_hdf5(hdf5_save_path, "f0", f0[:lengths].cpu().numpy().tolist())
                write_hdf5(hdf5_save_path, "wave", wave[0])

    if __name__ == "__main__":
        main()
