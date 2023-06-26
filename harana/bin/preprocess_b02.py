#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Perform feature extraction for SVCC23 B02: Decomposed FastSVC (F0, loudness, PPG, mcep, bap)

References:
https://github.com/kan-bayashi/ParallelWaveGAN
https://github.com/chomeyama/HN-UnifiedSourceFilterGAN
https://github.com/nnsvs/nnsvs/

"""

import argparse
import logging
import os
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
from scipy import interpolate
from speechbrain.pretrained import EncoderClassifier

from harana.layers import Stretch2d
from harana.datasets import AudioSCPDataset
from harana.utils import read_hdf5, write_hdf5, validate_length


def check_nan(array):
    tmp = np.sum(array)
    if np.isnan(tmp) or np.isinf(tmp):
        logging.warning('NaN or Inf found in input tensor.')
        return True
    return False


def interp1d(f0, kind="slinear"):
    """Coutinuous F0 interpolation from discontinuous F0 trajectory
    This function generates continuous f0 from discontinuous f0 trajectory
    based on :func:`scipy.interpolate.interp1d`. This is meant to be used for
    continuous f0 modeling in statistical speech synthesis
    (e.g., see [1]_, [2]_).
    If ``kind`` = ``'slinear'``, then this does same thing as Merlin does.
    Args:
        f0 (ndarray): F0 or log-f0 trajectory
        kind (str): Kind of interpolation that :func:`scipy.interpolate.interp1d`
            supports. Default is ``'slinear'``, which means linear interpolation.
    Returns:
        1d array (``T``, ) or 2d (``T`` x 1) array: Interpolated continuous f0
        trajectory.
    Examples:
        >>> from nnmnkwii.preprocessing import interp1d
        >>> import numpy as np
        >>> from nnmnkwii.util import example_audio_file
        >>> from scipy.io import wavfile
        >>> import pyworld
        >>> fs, x = wavfile.read(example_audio_file())
        >>> f0, timeaxis = pyworld.dio(x.astype(np.float64), fs, frame_period=5)
        >>> continuous_f0 = interp1d(f0, kind="slinear")
        >>> assert f0.shape == continuous_f0.shape
    .. [1] Yu, Kai, and Steve Young. "Continuous F0 modeling for HMM based
        statistical parametric speech synthesis." IEEE Transactions on Audio,
        Speech, and Language Processing 19.5 (2011): 1071-1079.
    .. [2] Takamichi, Shinnosuke, et al. "The NAIST text-to-speech system for
        the Blizzard Challenge 2015." Proc. Blizzard Challenge workshop. 2015.
    """
    ndim = f0.ndim
    if len(f0) != f0.size:
        raise RuntimeError("1d array is only supported")
    continuous_f0 = f0.flatten()
    nonzero_indices = np.where(continuous_f0 > 0)[0]

    # Nothing to do
    if len(nonzero_indices) <= 0:
        return f0

    # Need this to insert continuous values for the first/end silence segments
    continuous_f0[0] = continuous_f0[nonzero_indices[0]]
    continuous_f0[-1] = continuous_f0[nonzero_indices[-1]]

    # Build interpolation function
    nonzero_indices = np.where(continuous_f0 > 0)[0]
    interp_func = interpolate.interp1d(
        nonzero_indices, continuous_f0[continuous_f0 > 0], kind=kind
    )

    # Fill silence segments with interpolated values
    zero_indices = np.where(continuous_f0 <= 0)[0]
    continuous_f0[zero_indices] = interp_func(zero_indices)

    if ndim == 2:
        return continuous_f0[:, None]
    return continuous_f0


def loudness_extract(
    audio,
    sampling_rate,
    hop_length,
):
    stft = librosa.stft(audio, hop_length=hop_length)
    power_spectrum = np.square(np.abs(stft))
    bins = librosa.fft_frequencies(sr=sampling_rate)
    loudness = librosa.perceptual_weighting(power_spectrum, bins)
    loudness = librosa.db_to_amplitude(loudness)
    lft_noint = np.log(np.mean(loudness, axis=0) + 1e-5)

    return lft_noint


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
        help="kaldi-style wav.scp file. you need to specify either scp or rootdir.",
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        help=(
            "directory including wav files. you need to specify either scp or rootdir."
        ),
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
        "--extractordir",
        default=None,
        type=str,
        help="directory to the espnet pretrained model",
    )
    parser.add_argument(
        "--f0_path",
        default=None,
        type=str,
        help="path to the f0 information for each speaker.",
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
    with open(args.f0_path, 'r') as file:
        f0_file = yaml.load(file, Loader=yaml.FullLoader)

    # check arguments
    if (args.wav_scp is not None and args.rootdir is not None) or (
        args.wav_scp is None and args.rootdir is None
    ):
        raise ValueError("Please specify either --rootdir or --wav-scp.")

    # get dataset
    dataset = AudioSCPDataset(
        args.wav_scp,
        segments=args.segments,
        return_utt_id=True,
        return_sampling_rate=True,
    )

    # initialize hubert soft extractor
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").to(args.device)
    hubert.feature_extractor.conv6.stride = (1,) # change hop size to 10ms

    # speaker embedding extractor
    spk_emb_extractor = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb",
    )

    # check directory existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)

    # process each data
    for utt_id, (audio, fs) in tqdm(dataset):
        logging.info(f"processing {utt_id}")

        sr = config["sampling_rate"]
        shiftms = config["shiftms"]
        hop_size = config["hop_size"]
        lft_hop_size = config["lft_hop_size"]

        # checks
        assert (
            np.abs(audio).max() <= 1.0
        ), f"{utt_id} seems to be different from 16 bit PCM."
        if len(audio.shape) != 1:
            logging.warn(f"converting to mono")
            audio = librosa.to_mono(audio.T)

        # trim silence
        if config["trim_silence"]:
            logging.warn(f"trimming audio")
            audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )

        audio16k = librosa.resample(audio, fs, 16000)
        if fs != config["sampling_rate"]:
            logging.warn(f"resampling audio from {fs} to {config['sampling_rate']}")
            audio = librosa.resample(audio, fs, sr)

        # get min and max f0 information of each speaker
        spk_id = utt_id.split("_")[0]

        try:
            minf0 = f0_file[spk_id]["minf0"]
            maxf0 = f0_file[spk_id]["maxf0"]
        except:
            logging.warn(f"cannot find f0 information, using default values")
            minf0 = 100
            maxf0 = 1000

        # extract speaker embedding
        spk_emb = spk_emb_extractor.encode_batch(torch.from_numpy(audio16k))
        spk_emb = spk_emb.squeeze(0).transpose(1, 0)

        audio = np.array(audio, dtype=np.float)
        audio16k = np.array(audio16k, dtype=np.float)

        f0, timeaxis = pyworld.harvest(
            audio,
            fs=sr,
            f0_floor=minf0,
            f0_ceil=maxf0,
            frame_period=shiftms,
        )

        spectrogram = pyworld.cheaptrick(audio, f0, timeaxis, sr)
        aperiodicity = pyworld.d4c(audio, f0, timeaxis, sr)
        mcep = pysptk.sp2mc(spectrogram, order=config["mcep_dim"], alpha=pysptk.util.mcepalpha(sr))
        bap = pyworld.code_aperiodicity(aperiodicity, sr)

        if check_nan(bap):
            logging.warning(f'removing {utt_id} from preprocessing')
            continue
        if check_nan(mcep):
            logging.warning(f'removing {utt_id} from preprocessing')
            continue

        # compute continuous logf0
        f0 = f0[:, None]
        lf0 = f0.copy()
        nonzero_indices = np.nonzero(f0)
        lf0[nonzero_indices] = np.log(f0[nonzero_indices])
        vuv = (lf0 != 0).astype(np.float32)
        lf0 = interp1d(lf0, kind="slinear")

        # extract loudness features using A-weighting
        # NOTE: This lft is not interpolated like the one in FastSVC
        lft = loudness_extract(audio, sr, hop_size)
        lft = np.expand_dims(lft, axis=-1)

        # ppg extraction and linear interpolation to 24kHz (of 5ms hop size)
        hubert = hubert.double()
        raw_ppg = hubert.units(torch.from_numpy(audio16k.astype(np.double)).to(args.device).unsqueeze(0).unsqueeze(0).double())
        raw_ppg = raw_ppg.permute(0, 2, 1)
        ppg = F.interpolate(raw_ppg, scale_factor=1.5)
        ppg = ppg.permute(0, 2, 1).squeeze(0).cpu().numpy()

        logging.info("BEFORE LENGTH ADJUSTMENT")
        logging.info(f"raw PPG: {raw_ppg.shape}")
        logging.info(f"PPG: {ppg.shape}")
        logging.info(f"F0: {f0.shape}")
        logging.info(f"cont F0: {lf0.shape}")
        logging.info(f"lft: {lft.shape}")
        logging.info(f"audio: {audio.shape}")
        logging.info(f"mcep: {mcep.shape}")
        logging.info(f"bap: {bap.shape}")

        audio, lf0 = validate_length(audio, lf0, hop_size)
        lf0, lft = validate_length(lf0, lft)
        lft, ppg = validate_length(lft, ppg)
        ppg, mcep = validate_length(ppg, mcep)
        mcep, bap = validate_length(mcep, bap)

        audio, f0 = validate_length(audio, f0, hop_size)

        logging.info("AFTER LENGTH ADJUSTMENT")
        logging.info(f"PPG: {ppg.shape}")
        logging.info(f"F0: {f0.shape}")
        logging.info(f"cont F0: {lf0.shape}")
        logging.info(f"lft: {lft.shape}")
        logging.info(f"audio: {audio.shape}")
        logging.info(f"spk_emb: {spk_emb.shape}")
        logging.info(f"mcep: {mcep.shape}")
        logging.info(f"bap: {bap.shape}")

        # save to hdf5 files
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "wave",
            audio.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "f0",
            f0.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "lf0",
            lf0.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "vuv",
            vuv.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "mcep",
            mcep.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "bap",
            bap.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "lft",
            lft.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "ppg",
            ppg.astype(np.float32)
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "spk_emb",
            spk_emb.cpu().numpy().astype(np.float32),
        )

if __name__ == "__main__":
    main()
