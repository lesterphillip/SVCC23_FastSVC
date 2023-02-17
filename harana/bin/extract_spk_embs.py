#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Speaker embedding extraction using x-vectors.
References:
    - https://github.com/s3prl/s3prl/tree/master/s3prl/downstream/a2a-vc-vctk
"""

import argparse
import logging
import os
import copy

import librosa
import torch
import numpy as np
import yaml
import soundfile as sf

from tqdm import tqdm

from harana.datasets import AudioSCPDataset
from harana.utils import write_hdf5
from speechbrain.pretrained import EncoderClassifier


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

    spk_emb_extractor = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb",
    )
    spk_dict = {}
    sampling_rate = config["sampling_rate"]

    # process each data
    for utt_id, (audio, fs) in tqdm(dataset):
        # check
        assert len(audio.shape) == 1, f"{utt_id} seems to be multi-channel signal."
        assert (
            np.abs(audio).max() <= 1.0
        ), f"{utt_id} seems to be different from 16 bit PCM."
        if fs != 16000:
            logging.warn(f"resampling audio from {fs} to {16000}")
            audio = librosa.resample(audio, fs, 16000)

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )

        logging.info(f"processing {utt_id}")

        # NOTE: speaker id needs to be {id}_{any} format
        spk_id = utt_id.split("_")[0]
        logging.info(f"input_audio {audio.shape}")

        # extract speaker embedding and save to dictionary
        audio = np.array(audio, dtype=np.float)
        spk_emb = spk_emb_extractor.encode_batch(torch.from_numpy(audio))
        if spk_dict.get(spk_id) is None:
            spk_dict[spk_id] = []
        spk_dict[spk_id].append(spk_emb.cpu().squeeze(0).numpy().tolist())

    # combine speaker embeddings, then write to h5 file
    for key, value in spk_dict.items():
        logging.info(f"processing speaker id: {key}")
        stack_spk_emb = np.array(value)
        mean_spk_emb = np.mean(stack_spk_emb, axis=0)

        # save to hdf5 files
        write_hdf5(
            os.path.join(args.dumpdir, "spk_embs.h5"),
            key,
            mean_spk_emb.astype(np.float32),
        )


if __name__ == "__main__":
    main()
