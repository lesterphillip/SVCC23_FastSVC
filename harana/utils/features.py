# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Feature-related functions.

References:
    - https://github.com/k2kobayashi/sprocket
    - https://github.com/chomeyama/HN-UnifiedSourceFilterGAN
"""

import sys
from logging import getLogger

import numpy as np
import torch
from torch.nn.functional import interpolate

# file logger
logger = getLogger(__name__)


class F0Statistics(object):
    """F0 statistics class
    Estimate F0 statistics and convert F0
    """

    def __init__(self):
        pass

    def estimate(self, f0list):
        """Estimate F0 statistics from list of f0
        Parameters
        ---------
        f0list : list, shape('f0num')
            List of several F0 sequence
        Returns
        ---------
        f0stats : array, shape(`[mean, std]`)
            Values of mean and standard deviation for logarithmic F0
        """

        n_files = len(f0list)
        for i in range(n_files):
            f0 = f0list[i]
            nonzero_indices = np.nonzero(f0)
            if i == 0:
                f0s = np.log(f0[nonzero_indices])
            else:
                f0s = np.r_[f0s, np.log(f0[nonzero_indices])]

        f0stats = np.array([np.mean(f0s), np.std(f0s)])
        return f0stats

    def convert(self, f0, orgf0stats, tarf0stats):
        """Convert F0 based on F0 statistics
        Parameters
        ---------
        f0 : array, shape(`T`, `1`)
            Array of F0 sequence
        orgf0stats, shape (`[mean, std]`)
            Vector of mean and standard deviation of logarithmic F0 for original speaker
        tarf0stats, shape (`[mean, std]`)
            Vector of mean and standard deviation of logarithmic F0 for target speaker
        Returns
        ---------
        cvf0 : array, shape(`T`, `1`)
            Array of converted F0 sequence
        """

        # get length and dimension
        T = len(f0)

        # perform f0 conversion
        cvf0 = np.zeros(T)

        nonzero_indices = f0 > 0
        cvf0[nonzero_indices] = np.exp(
            (tarf0stats[1] / orgf0stats[1])
            * (np.log(f0[nonzero_indices]) - orgf0stats[0])
            + tarf0stats[0]
        )

        return cvf0


class SignalGenerator:
    """Input signal generator module."""

    def __init__(
        self,
        sample_rate=16000,
        hop_size=640,
        sine_amp=0.1,
        noise_amp=0.003,
        signal_types=["sine", "noise"],
    ):
        """Initialize WaveNetResidualBlock module.

        Args:
            sample_rate (int): Sampling rate.
            hop_size (int): Hop size of input F0.
            sine_amp (float): Sine amplitude for NSF-based sine generation.
            noise_amp (float): Noise amplitude for NSF-based sine generation.
            signal_types (list): List of input signal types for generator.

        """
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.signal_types = signal_types
        self.sine_amp = sine_amp
        self.noise_amp = noise_amp

        for signal_type in signal_types:
            if not signal_type in ["noise", "sine", "uv"]:
                logger.info(f"{signal_type} is not supported type for generator input.")
                sys.exit(0)
        logger.info(f"Use {signal_types} for generator input signals.")

    @torch.no_grad()
    def __call__(self, f0):
        signals = []
        for typ in self.signal_types:
            if "noise" == typ:
                signals.append(self.random_noise(f0))
            if "sine" == typ:
                signals.append(self.sinusoid(f0))
            if "uv" == typ:
                signals.append(self.vuv_binary(f0))

        input_batch = signals[0]
        for signal in signals[1:]:
            input_batch = torch.cat([input_batch, signal], axis=1)

        return input_batch

    @torch.no_grad()
    def random_noise(self, f0):
        """Calculate noise signals.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: Gaussian noise signals (B, 1, T).

        """
        B, _, T = f0.size()
        noise = torch.randn((B, 1, T * self.hop_size), device=f0.device)

        return noise

    @torch.no_grad()
    def sinusoid(self, f0):
        """Calculate sine signals.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: Sines generated following NSF (B, 1, T).

        """
        B, _, T = f0.size()
        vuv = interpolate((f0 > 0) * torch.ones_like(f0), T * self.hop_size)
        radious = (interpolate(f0, T * self.hop_size) / self.sample_rate) % 1
        sine = vuv * torch.sin(torch.cumsum(radious, dim=2) * 2 * np.pi) * self.sine_amp
        if self.noise_amp > 0:
            noise_amp = vuv * self.noise_amp + (1.0 - vuv) * self.noise_amp / 3.0
            noise = torch.randn((B, 1, T * self.hop_size), device=f0.device) * noise_amp
            sine = sine + noise

        return sine

    @torch.no_grad()
    def vuv_binary(self, f0):
        """Calculate V/UV binary sequences.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: V/UV binary sequences (B, 1, T).

        """
        _, _, T = f0.size()
        uv = interpolate((f0 > 0) * torch.ones_like(f0), T * self.hop_size)

        return uv
