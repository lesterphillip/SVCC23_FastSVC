# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
# MIT License (https://opensource.org/licenses/MIT)

"""
HN-Unified Source-Filter GAN.
References:
    - https://github.com/chomeyama/HN-UnifiedSourceFilterGAN
"""

from logging import getLogger

import copy
import numpy as np
from logging import getLogger
from tkinter import W

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as WN

from harana.layers import Conv1d1x1, ResidualBlocks, upsample
from harana.layers.residual_block import PeriodicityEstimator
from harana.utils import index_initial
from harana.layers import Conv1d, Conv2d
from torchaudio.functional import spectrogram

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# A logger for this file
logger = getLogger(__name__)


class HNUSFGANGenerator(nn.Module):
    """Parallel hn-uSFGAN Generator module."""

    def __init__(
        self,
        harmonic_network_params={
            "blockA": 20,
            "cycleA": 4,
            "blockF": 0,
            "cycleF": 0,
            "cascade_mode": 0,
        },
        noise_network_params={
            "blockA": 0,
            "cycleA": 0,
            "blockF": 5,
            "cycleF": 5,
            "cascade_mode": 0,
        },
        filter_network_params={
            "blockA": 0,
            "cycleA": 0,
            "blockF": 30,
            "cycleF": 3,
            "cascade_mode": 0,
        },
        periodicity_estimator_params={
            "conv_blocks": 3,
            "kernel_size": 5,
            "dilation": 1,
            "padding_mode": "replicate",
        },
        in_channels=1,
        out_channels=1,
        residual_channels=64,
        gate_channels=128,
        skip_channels=64,
        aux_channels=62,
        aux_context_window=2,
        use_weight_norm=True,
        upsample_params={"upsample_scales": [5, 4, 3, 2]},
    ):
        """Initialize ParallelHnUSFGANGenerator module.
        Args:
            harmonic_network_params (dict): Periodic source generation network parameters.
            noise_network_params (dict): Aperiodic source generation network parameters.
            filter_network_params (dict): Filter network parameters.
            periodicity_estimator_params (dict): Periodicity estimation network parameters.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            upsample_params (dict): Upsampling network parameters.
        """
        super(HNUSFGANGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.n_ch = residual_channels

        # define first convolution
        self.conv_first_sine = Conv1d1x1(in_channels, residual_channels)
        self.conv_first_noise = Conv1d1x1(in_channels, residual_channels)

        # define upsampling network
        self.upsample_net = getattr(upsample, "ConvInUpsampleNetwork")(
            **upsample_params,
            aux_channels=aux_channels,
            aux_context_window=aux_context_window,
        )

        # define harmonic/noise/filter networks
        for params in [
            harmonic_network_params,
            noise_network_params,
            filter_network_params,
        ]:
            params.update(
                {
                    "residual_channels": residual_channels,
                    "gate_channels": gate_channels,
                    "skip_channels": skip_channels,
                    "aux_channels": aux_channels,
                }
            )
        self.harmonic_network = ResidualBlocks(**harmonic_network_params)
        self.noise_network = ResidualBlocks(**noise_network_params)
        self.filter_network = ResidualBlocks(**filter_network_params)

        # define periodicity estimator
        self.periodicity_estimator = PeriodicityEstimator(
            **periodicity_estimator_params, in_channels=aux_channels
        )

        # convert hidden representation to output signal
        self.conv_last = nn.Sequential(
            nn.ReLU(),
            Conv1d1x1(skip_channels, skip_channels),
            nn.ReLU(),
            Conv1d1x1(skip_channels, out_channels),
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, c, d):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').
            d (Tensor): Input pitch-dependent dilated factors (B, 1, T).
        Returns:
            Tensor: Output tensor (B, 1, T)
        """
        # index initialization
        batch_index, ch_index = index_initial(x.size(0), self.n_ch)

        # upsample auxiliary features
        c = self.upsample_net(c)
        assert c.size(-1) == x.size(-1), f"{c.size(-1)} vs {x.size(-1)}"

        # estimate periodicity
        a = self.periodicity_estimator(c)

        # assume the first channel is sine and the other is noise
        sine, noise = torch.chunk(x, 2, 1)

        # encode to hidden representation
        h = self.conv_first_sine(sine)
        n = self.conv_first_noise(noise)

        # generate periodic and aperiodic source latent features
        h = self.harmonic_network(h, c, d, None, batch_index, ch_index)
        n = self.noise_network(n, c, d, None, batch_index, ch_index)

        # merge periodic and aperiodic latent features
        h = a * h
        n = (1.0 - a) * n
        s = h + n

        # resonance filtering
        x = self.filter_network(s, c, d, None, batch_index, ch_index)
        x = self.conv_last(x)

        # convert to 1d signal for regularization loss
        s = self.conv_last(s)

        # just for debug
        with torch.no_grad():
            h = self.conv_last(h)
            n = self.conv_last(n)

        return x, s

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logger.debug(f"Weight norm is removed from {m}.")
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
                logger.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)


