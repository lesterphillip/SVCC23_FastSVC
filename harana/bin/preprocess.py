#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Perform feature extraction for FastSVC (F0, loudness, PPG).

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/chomeyama/HN-UnifiedSourceFilterGAN
"""

import argparse
import logging
import os
import sys
import copy

import librosa
import pysptk
import pyworld
import torch
import numpy as np
import yaml
import soundfile as sf
import torch.nn.functional as F

from tqdm import tqdm
from numpy import pi, log10
from scipy.interpolate import interp1d
from scipy.signal import firwin, lfilter

from harana.layers import Stretch2d
from harana.datasets import AudioSCPDataset
from harana.utils import read_hdf5, write_hdf5, validate_length
from harana.ppg import load_ppg_model


def f0_extract(
    audio,
    sampling_rate=24000,
    minf0=70,
    maxf0=340,
    shiftms=5,
):
    # extract discontinuous f0
    f0, t = pyworld.harvest(
        audio,
        fs=sampling_rate,
        f0_floor=minf0,
        f0_ceil=maxf0,
        frame_period=shiftms,
    )

    return f0, t


def loudness_extract(audio, sampling_rate, hop_length):
    stft = librosa.stft(audio, hop_length=hop_length)
    power_spectrum = np.square(np.abs(stft))
    bins = librosa.fft_frequencies(sr=sampling_rate)
    loudness = librosa.perceptual_weighting(power_spectrum, bins)
    loudness = librosa.db_to_amplitude(loudness)
    loudness = np.log(np.mean(loudness, axis=0) + 1e-5)
    loudness = torch.from_numpy(loudness)

    for _ in range(3):
        loudness = loudness.unsqueeze(0)
    scaler = Stretch2d(hop_length, 1)
    loudness = scaler(loudness)
    for _ in range(3):
        loudness = loudness.squeeze(0)
    return loudness.numpy()


def ppg_extract(
    audio,
    device,
    train_config,
    model_file,
):
    model = load_ppg_model(train_config, model_file, device)
    wav_tensor = torch.from_numpy(audio).float().to(device).unsqueeze(0)
    wav_length = torch.LongTensor([audio.shape[0]]).to(device)

    with torch.no_grad():
        bnf = model(wav_tensor, wav_length)

    return bnf


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess audio and then extract features (See detail in"
            " harana/bin/preprocess.py)."
        )
    )
    parser.add_argument(
        "--wav-scp",
        "--scp",
        default=None,
        type=str,
        required=True,
        help="kaldi-style wav.scp file. you need to specify either scp or rootdir.",
    )
    parser.add_argument(
        "--segments",
        default=None,
        type=str,
        help="kaldi-style segments file. if use, you must to specify both scp and segments.",
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
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="device to be used when extracting PPG features.",
    )
    parser.add_argument(
        "--f0_path",
        default=None,
        type=str,
        help="path to the f0 information for each speaker.",
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

    # load config file for feature extraction
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # read f0 min/max file
    with open(args.f0_path, "r") as file:
        f0_file = yaml.load(file, Loader=yaml.FullLoader)

    # get dataset
    dataset = AudioSCPDataset(
        args.wav_scp,
        segments=args.segments,
        return_utt_id=True,
        return_sampling_rate=True,
    )

    # check directory existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)

    # process each data
    for utt_id, (audio, fs) in tqdm(dataset):
        logging.info(f"processing {utt_id}")
        # check
        assert len(audio.shape) == 1, f"{utt_id} seems to be multi-channel signal."
        assert (
            np.abs(audio).max() <= 1.0
        ), f"{utt_id} seems to be different from 16 bit PCM."

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )

        # resample to 24kHz and 16kHz (for PPG extraction)
        audio16k = librosa.resample(audio, fs, 16000)
        if fs != config["sampling_rate"]:
            logging.warn(f"resampling audio from {fs} to {config['sampling_rate']}")
            audio = librosa.resample(audio, fs, config["sampling_rate"])

        sampling_rate = config["sampling_rate"]
        shiftms = config["shiftms"]
        hop_size = config["hop_size"]
        lft_hop_size = config["lft_hop_size"]

        # get min and max f0 information of each speaker
        spk_id = utt_id.split("_")[0]
        minf0 = f0_file[spk_id]["minf0"]
        maxf0 = f0_file[spk_id]["maxf0"]

        # read speaker embedding files
        try:
            spk_emb = read_hdf5(args.spk_emb_path, spk_id)
            spk_emb = np.array(spk_emb.reshape(-1, 1))
        except:
            logging.error(f"cannot find spk emb file, make sure to run stage 0 first.")
            sys.exit(1)

        logging.info(f"input_audio {audio.shape}")

        audio = np.array(audio, dtype=np.float)
        audio16k = np.array(audio16k, dtype=np.float)  # for PPG extractor

        # extract F0 using WORLD
        f0, t = f0_extract(audio, sampling_rate, minf0, maxf0, shiftms)

        if np.all(f0 <= 0):
            logging.warn(f"contains negative f0, may cause an error later")

        f0 = np.expand_dims(f0, axis=-1)

        # extract loudness features using A-weighting
        lft = loudness_extract(audio, sampling_rate, lft_hop_size)
        lft = np.expand_dims(lft, axis=-1)

        # ppg extraction and linear interpolation to 24kHz
        raw_ppg = ppg_extract(
            audio16k, args.device, config["ppg_conf_path"], config["ppg_model_path"]
        )
        raw_ppg = raw_ppg.permute(0, 2, 1)
        ppg = F.interpolate(raw_ppg, scale_factor=1.5)
        ppg = ppg.permute(0, 2, 1).squeeze(0).cpu().numpy()

        # sanity check to ensure feature lengths are being correctly cut
        logging.info("BEFORE LENGTH ADJUSTMENT")
        logging.info(f"PPG: {ppg.shape}")
        logging.info(f"F0: {f0.shape}")
        logging.info(f"lft: {lft.shape}")
        logging.info(f"audio: {audio.shape}")

        audio, f0 = validate_length(audio, f0, hop_size)
        audio, lft = validate_length(audio, lft)
        f0, ppg = validate_length(f0, ppg)

        logging.info("AFTER LENGTH ADJUSTMENT")
        logging.info(f"PPG: {ppg.shape}")
        logging.info(f"F0: {f0.shape}")
        logging.info(f"lft: {lft.shape}")
        logging.info(f"audio: {audio.shape}")
        logging.info(f"spk_emb: {spk_emb.shape}")

        # save to hdf5 files
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "wave",
            audio.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "f0",
            f0,
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "lft",
            lft,
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"), "ppg", ppg.astype(np.float32)
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "spk_emb",
            spk_emb.astype(np.float32),
        )


if __name__ == "__main__":
    main()
