# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules for FastSVC.
References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
"""

import logging
import os
import random
import yaml
import numpy as np

from multiprocessing import Manager
from torch.utils.data import Dataset
from harana.utils import find_files, read_hdf5


class FastSVCDataset(Dataset):
    """PyTorch compatible audio and mel dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*.h5",
        feats_query="*.h5",
        audio_load_fn=lambda x: read_hdf5(x, "wave"),
        f0_load_fn=lambda x: read_hdf5(x, "f0"),
        ppg_load_fn=lambda x: read_hdf5(x, "ppg"),
        lft_load_fn=lambda x: read_hdf5(x, "lft"),
        emb_load_fn=lambda x: read_hdf5(x, "spk_emb"),
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            feats_query (str): Query to find feature files in root_dir.
            audio_load_fn (func): Function to load audio file.
            f0_load_fn (func): Function to load f0 feature file.
            ppg_load_fn (func): Function to load ppg feature file.
            lft_load_fn (func): Function to load loudness feature file.
            emb_load_fn (func): Function to load speaker embeddings feature file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query))
        feats_files = sorted(find_files(root_dir, feats_query))

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."
        assert len(audio_files) == len(feats_files), (
            f"Number of audio and mel files are different ({len(audio_files)} vs"
            f" {len(mel_files)})."
        )

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.f0_load_fn = f0_load_fn
        self.ppg_load_fn = ppg_load_fn
        self.lft_load_fn = lft_load_fn
        self.emb_load_fn = emb_load_fn
        self.feats_files = feats_files
        self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in audio_files]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])
        f0 = self.f0_load_fn(self.feats_files[idx])
        ppg = self.ppg_load_fn(self.feats_files[idx])
        lft = self.lft_load_fn(self.feats_files[idx])
        emb = self.emb_load_fn(self.feats_files[idx])

        if self.return_utt_id:
            items = utt_id, audio, f0, ppg, lft, emb
        else:
            items = audio, f0, ppg, lft, emb

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)
