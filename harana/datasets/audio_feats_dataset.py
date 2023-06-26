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
from harana.utils import find_files, dilated_factor, read_hdf5, validate_length


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


class B02Dataset(Dataset):
    """PyTorch compatible audio and auxiliary features dataset."""

    def __init__(
        self,
        root_dir,
        sample_rate=16000,
        hop_size=160,
        dense_factor=4,
        query="*.h5",
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        h5_files = sorted(find_files(root_dir, query))

        # assert the number of files
        assert len(h5_files) != 0, f"No files found in ${root_dir}."

        self.h5_files = h5_files
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.dense_factor = dense_factor
        self.audio_load_fn = lambda x: read_hdf5(x, "wave")
        self.ppg_load_fn = lambda x: read_hdf5(x, "ppg")
        self.mcep_load_fn = lambda x: read_hdf5(x, "mcep")
        self.bap_load_fn = lambda x: read_hdf5(x, "bap")
        self.f0_load_fn = lambda x: read_hdf5(x, "f0")
        self.lft_load_fn = lambda x: read_hdf5(x, "lft")
        self.logf0_load_fn = lambda x: read_hdf5(x, "lf0")
        self.embs_load_fn = lambda x: read_hdf5(x, "spk_emb")

        self.utt_ids = [
            os.path.splitext(os.path.basename(f))[0] for f in h5_files
        ]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(h5_files))]

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
        wave_files = self.audio_load_fn(self.h5_files[idx])
        ppg_features = self.ppg_load_fn(self.h5_files[idx])
        mcep_features = self.mcep_load_fn(self.h5_files[idx])
        bap_features = self.bap_load_fn(self.h5_files[idx])

        aux_features = []
        aux_features += [mcep_features]
        aux_features += [bap_features]
        aux_features = np.concatenate(aux_features, axis=1)

        lft_features = self.lft_load_fn(self.h5_files[idx])
        logf0_features = self.logf0_load_fn(self.h5_files[idx])

        embs_features = self.embs_load_fn(self.h5_files[idx])

        f0_features = self.f0_load_fn(self.h5_files[idx])
        df = dilated_factor(
            np.squeeze(f0_features.copy()), self.sample_rate, self.dense_factor
        )
        df_features = df.repeat(self.hop_size, axis=0)

        if self.return_utt_id:
            items = utt_id, wave_files, ppg_features, mcep_features, bap_features, lft_features, logf0_features, embs_features, df_features, f0_features
        else:
            items = wave_files, ppg_features, mcep_features, bap_features, lft_features, logf0_features, embs_features, df_features, f0_features

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.h5_files)


class USFGANDataset(Dataset):
    """PyTorch compatible audio and auxiliary features dataset."""

    def __init__(
        self,
        root_dir,
        aux_features="world",
        sample_rate=24000,
        hop_size=160,
        dense_factor=4,
        query="*.h5",
        return_utt_id=False,
        allow_cache=False,
        decode_mode=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        h5_files = sorted(find_files(root_dir, query))

        # assert the number of files
        assert len(h5_files) != 0, f"No files found in ${root_dir}."

        self.h5_files = h5_files
        self.aux_features = aux_features
        self.decode_mode = decode_mode
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.dense_factor = dense_factor
        self.audio_load_fn = lambda x: read_hdf5(x, "wave")
        self.mcep_load_fn = lambda x: read_hdf5(x, "mcep")
        self.bap_load_fn = lambda x: read_hdf5(x, "bap")
        self.f0_load_fn = lambda x: read_hdf5(x, "f0")
        self.utt_ids = [
            os.path.splitext(os.path.basename(f))[0] for f in h5_files
        ]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(h5_files))]

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
        audio_features = self.audio_load_fn(self.h5_files[idx])
        mcep_features = self.mcep_load_fn(self.h5_files[idx])
        bap_features = self.bap_load_fn(self.h5_files[idx])

        f0_features = self.f0_load_fn(self.h5_files[idx])

        aux_features = []
        aux_features += [mcep_features]
        aux_features += [bap_features]
        aux_features = np.concatenate(aux_features, axis=1)

        if f0_features.shape[0] == 1:
            f0_features = np.squeeze(f0_features, axis=0)
        elif f0_features.shape[0] != 1:
            f0_features = np.expand_dims(f0_features, axis=1)
        f0_features, aux_features = validate_length(f0_features, aux_features)
        audio_features, f0_features = validate_length(audio_features, f0_features, 160)
        f0_features = np.expand_dims(f0_features, axis=0)

        df = dilated_factor(
            np.squeeze(f0_features.copy()), self.sample_rate, self.dense_factor
        )
        df_features = df.repeat(self.hop_size, axis=0)

        if self.return_utt_id:
            items = utt_id, audio_features, aux_features, df_features, f0_features
        else:
            items = audio_features, aux_features, df_features, f0_features

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.h5_files)


class Taco2Dataset(Dataset):
    """PyTorch compatible audio and auxiliary features dataset."""

    def __init__(
        self,
        root_dir,
        input_feat_type="ppg",
        aux_features="logmel",
        query="*.h5",
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        h5_files = sorted(find_files(root_dir, query))

        # assert the number of files
        assert len(h5_files) != 0, f"No files found in ${root_dir}."

        self.h5_files = h5_files
        self.input_feat_type = input_feat_type
        self.aux_features = aux_features
        self.input_load_fn = lambda x: read_hdf5(x, input_feat_type)
        self.mcep_load_fn = lambda x: read_hdf5(x, "mcep")
        self.bap_load_fn = lambda x: read_hdf5(x, "bap")
        self.lft_load_fn = lambda x: read_hdf5(x, "lft")
        self.logf0_load_fn = lambda x: read_hdf5(x, "lf0")
        self.f0_load_fn = lambda x: read_hdf5(x, "f0")
        self.wave_load_fn = lambda x: read_hdf5(x, "wave")
        self.spk_emb_load_fn = lambda x: read_hdf5(x, "spk_emb")
        self.utt_ids = [
            os.path.splitext(os.path.basename(f))[0] for f in h5_files
        ]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(h5_files))]

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
        input_features = self.input_load_fn(self.h5_files[idx])
        mcep_features = self.mcep_load_fn(self.h5_files[idx])
        bap_features = self.bap_load_fn(self.h5_files[idx])
        lft_features = self.lft_load_fn(self.h5_files[idx])
        logf0_features = self.logf0_load_fn(self.h5_files[idx])

        # validate lengths
        input_features, logf0_features = validate_length(input_features, logf0_features)
        logf0_features, lft_features = validate_length(logf0_features, lft_features)

        aux_features = []
        aux_features += [mcep_features]
        aux_features += [bap_features]
        aux_features = np.concatenate(aux_features, axis=1)

        f0_features = self.f0_load_fn(self.h5_files[idx])
        wave_features = self.wave_load_fn(self.h5_files[idx])
        spk_emb_features = self.spk_emb_load_fn(self.h5_files[idx])

        if self.return_utt_id:
            items = utt_id, input_features, aux_features, lft_features, logf0_features, spk_emb_features, wave_features, f0_features
        else:
            items = input_features, aux_features, lft_features, logf0_features, spk_emb_features, wave_features, f0_features

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.h5_files)

