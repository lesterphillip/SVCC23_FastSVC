# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""FastSVC modules.

Please refer to the original paper:
    - FastSVC: Fast Cross-Domain Singing Voice Conversion with Feature-wise Linear Modulation
    - Songxiang Liu, Yuewen Cao, Na Hu, Dan Su, Helen Meng
    - https://arxiv.org/abs/2011.05731

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
"""

import logging
import math

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import load

from harana.layers import Conv1d1x1, Conv1d1x3, Conv2d1x3
from harana.layers import Squeeze2d, Stretch2d
from harana.layers import upsample
from harana import models
from harana.utils import read_hdf5


class FastSVCUpsampleNet(torch.nn.Module):
    """FastSVC upsampling block (Fig. 4a)"""

    def __init__(
        self,
        in_channels,
        mid_channels,
        upsampling_scale,
        spk_emb_size=512,
        use_spk_emb=True,
    ):
        """Initialize FastSVC upsampling block.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of output channels.
            upsampling_scale (int): Rate to upsample linguistic features.
            spk_emb_size (int): Dimension of speaker embedding.
            use_spk_emb (bool): Whether or not to condition the network with the speaker embedding.

        """
        super(FastSVCUpsampleNet, self).__init__()
        self.conv_first = Conv2d1x3(in_channels, mid_channels, (0, 1), 1)
        self.upsample_block0 = nn.Sequential(
            nn.LeakyReLU(0.2),
            Stretch2d(upsampling_scale, 1),
            Conv2d1x3(mid_channels, mid_channels, (0, 1), 1),
            nn.LeakyReLU(0.2),
        )
        self.conv_block1 = nn.Sequential(
            nn.LeakyReLU(0.2), Conv2d1x3(mid_channels, mid_channels, (0, 3), 3)
        )
        self.conv_block2 = nn.Sequential(
            nn.LeakyReLU(0.2), Conv2d1x3(mid_channels, mid_channels, (0, 9), 9)
        )
        self.conv_block3 = nn.Sequential(
            nn.LeakyReLU(0.2), Conv2d1x3(mid_channels, mid_channels, (0, 27), 27)
        )
        self.residual_block = nn.Sequential(
            Stretch2d(upsampling_scale, 1),
            Conv2d1x3(mid_channels, mid_channels, (0, 1), 1),
        )
        self.instance_norm = nn.InstanceNorm2d(mid_channels)
        if use_spk_emb:
            self.emb_projector = nn.Linear(spk_emb_size, mid_channels)

    def forward(self, x, s, l, spk_emb=None):
        """Calculate upsampling block's forward propagation.

        Args:
            x (Tensor): Input linguistic features (B, C', T).
            s (Tuple of Tensors): Downsampled sine wave (B, C' ,T').
            l (Tuple of Tensors): Downsampled loudness features (B, C',T').

        Returns:
            Tensor: Output tensor (B, out_channels, T)

        """
        x = x.unsqueeze(2)  # B, C, 1, T
        x = self.conv_first(x)
        xr = self.residual_block(x)

        # compute first half of upsampling block
        x = self.upsample_block0(x)
        x = self._feature_affine(x, s, l, spk_emb)
        x = self.conv_block1(x)

        # apply residuals
        x_ = x + xr

        # compute second half of upsampling block
        x = self._feature_affine(x_, s, l, spk_emb)
        x = self.conv_block2(x)
        x = self._feature_affine(x, s, l, spk_emb)
        x = self.conv_block3(x)

        # apply skip connection
        x = x + x_
        x = x.squeeze(2)  # B, C', T'
        return x

    def _feature_affine(self, x, sine, lft, spk_emb=None):
        """Fuse in sine wave and linguistic information.

        Args:
            x (Tensor): Input linguistic features (B, C, 1, T).
            s (Tuple of Tensors): Downsampled sine wave after FiLM (B, 1 ,T').
            l (Tuple of Tensors): Downsampled loudness features after FiLM (B, 1 ,T').
            spk_emb (Tensors): Speaker embeddings (B, S).

        Returns:
            Tensor: Output tensor (B, out_channels, T)
        """
        s_scale, s_shift = sine
        l_scale, l_shift = lft
        scale = s_scale + l_scale
        shift = s_shift.unsqueeze(2) + l_shift.unsqueeze(2)
        x = torch.mul(scale.unsqueeze(2), x)
        x = x + shift

        if spk_emb is not None:
            spk_emb = (
                self.emb_projector(F.normalize(spk_emb)).unsqueeze(2).unsqueeze(3)
            )  # B, S, 1, 1
            x = self.instance_norm(x)
            x = x + spk_emb
        return x


class FastSVCDownsampleNet(torch.nn.Module):
    """FastSVC downsampling block (Fig. 4b)"""

    def __init__(
        self,
        in_channels=1,
        mid_channels=[12, 24, 48, 96, 192],
        downsampling_scales=[1, 5, 4, 4, 4],
    ):
        """Initialize FastSVC downsampling block.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of output channels.
            downsampling_scale (int): Rate to downsample features.
                Computed by upsampling_scales.pop().insert(1, 0)

        """
        super(FastSVCDownsampleNet, self).__init__()

        # compute residual block
        self.residual_block = nn.Sequential(
            Conv1d1x1(in_channels, mid_channels),
            Squeeze2d(downsampling_scales),
        )

        # compute downsampling block
        self.downsample_block = nn.Sequential(
            Squeeze2d(downsampling_scales),
            nn.LeakyReLU(0.2),
            Conv1d1x3(in_channels, mid_channels, 1, 1),
            nn.LeakyReLU(0.2),
            Conv1d1x3(mid_channels, mid_channels, 2, 2),
            nn.LeakyReLU(0.2),
            Conv1d1x3(mid_channels, mid_channels, 4, 4),
        )

    def forward(self, x):
        """Calculate downsampling block's forward propagation.

        Args:
            x (Tensor): Input sine wave or loudness features (B, 1, T').

        Returns:
            Tensor: Output tensor (B, 1, T')
        """

        r = self.residual_block(x)
        x = self.downsample_block(x)
        x = x + r
        return x


class FastSVCFiLMNet(torch.nn.Module):
    """FastSVC FiLM block (Fig. 4c)"""

    def __init__(self, mid_channels):
        """Initialize FastSVC FiLM block.

        Args:
            mid_channels (int): Number of output channels in current upsampling block.

        """
        super(FastSVCFiLMNet, self).__init__()
        dilation = 1
        padding = (3 - 1) // 2 * dilation
        self.conv = Conv1d1x3(
            mid_channels, mid_channels, padding=padding, dilation=dilation
        )
        self.relu = nn.LeakyReLU(0.2)
        self.conv_scale = Conv1d1x3(
            mid_channels, mid_channels, padding=padding, dilation=dilation
        )
        self.conv_shift = Conv1d1x3(
            mid_channels, mid_channels, padding=padding, dilation=dilation
        )

    def forward(self, x):
        """Fuse in sine wave and linguistic information.

        Args:
            x (Tensor): Input sine wave or loudness features (B, C', T').

        Returns:
            Tensor: Output tensor (B, C', T')
        """
        x = self.relu(self.conv(x))
        scale = self.conv_scale(x)
        shift = self.conv_shift(x)
        return scale, shift


class FastSVCGenerator(torch.nn.Module):
    """FastSVC waveform generator (Fig. 3)"""

    def __init__(
        self,
        in_channels=144,
        mid_channels=[192, 96, 48, 24],
        upsampling_scales=[2, 4, 4, 5],
        out_channels=1,
        spk_emb_size=512,
        use_spk_emb=True,
    ):
        """Initialize FastSVC waveform generator.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of output channels for each block.
            upsampling_scales (int): Rate to upsample linguistic features.
            spk_emb_size (int): Dimension of speaker embedding.
            use_spk_emb (bool): Whether or not to condition the network with the speaker embedding.

        """
        super(FastSVCGenerator, self).__init__()
        self.in_channels = in_channels
        self.upsampling_scales = upsampling_scales
        self.mid_channels = mid_channels
        self.upsampling_nets = torch.nn.ModuleList()
        for idx, (scale, channel) in enumerate(zip(upsampling_scales, mid_channels)):
            upsample_net = FastSVCUpsampleNet(
                in_channels, channel, scale, spk_emb_size, use_spk_emb
            )
            self.upsampling_nets += [upsample_net]
            in_channels = channel

        # downsampling networks
        downsampling_scales = upsampling_scales[::-1]
        downsampling_scales.pop()
        downsampling_scales.insert(0, 1)

        downsampling_lft_layers = []
        downsampling_sine_layers = []
        in_channels = 1
        for scale, channel in zip(downsampling_scales, mid_channels[::-1]):
            downsampling_lft_layers += [
                FastSVCDownsampleNet(in_channels, channel, scale)
            ]
            downsampling_sine_layers += [
                FastSVCDownsampleNet(in_channels, channel, scale)
            ]
            in_channels = channel

        self.downsampling_lft = nn.Sequential(*downsampling_lft_layers)
        self.downsampling_sine = nn.Sequential(*downsampling_sine_layers)

        # FiLM blocks
        self.film_lft = torch.nn.ModuleList()
        self.film_sine = torch.nn.ModuleList()
        for channel in mid_channels[::-1]:
            # film block for downsampled loudness features
            film_lft_block = FastSVCFiLMNet(channel)
            self.film_lft += [film_lft_block]

            # film block for downsampled sine features
            film_sine_block = FastSVCFiLMNet(channel)
            self.film_sine += [film_sine_block]

        self.conv_last = Conv1d1x1(mid_channels[-1], out_channels)

        self.apply_weight_norm()

    def forward(self, x, s, l, spk_emb=None):
        """Calculate forward propagation

        Args:
            x (Tensor): Input linguistic features (B, C, T').
            s (Tuple): Input sine wave (B, C, T').
            l (Tuple): Input loudness features (B, C, T').

        Returns:
            Tensor: Output tensor (B, out_channels, T')
        """

        for idx, upsample_net in enumerate(self.upsampling_nets):
            # compute film of the downsampled loudness and sine
            didx = len(self.upsampling_scales) - idx - 1

            # compute downsampled
            lft_downsampled = self.downsampling_loop(l, didx, self.downsampling_lft)
            lft = self.film_lft[didx](lft_downsampled)

            sine_downsampled = self.downsampling_loop(s, didx, self.downsampling_sine)
            sine = self.film_sine[didx](sine_downsampled)

            x = upsample_net(x, sine, lft, spk_emb)

        x = self.conv_last(x)

        return x

    def downsampling_loop(self, x, didx, nets):
        """Calculate the output until the didx index."""
        for idx, net in enumerate(nets):
            x = net(x)
            if idx == didx:
                return x
        raise ValueError("index went over the length of the network")

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def inference(self, x, f0, l, signal_generator, pad_fn, spk_emb=None):
        """Perform inference.

        Args:
            x (Union[Tensor, ndarray]): Linguistic features (T' ,C).
            f0 (Union[Tensor, ndarray]): Input f0 signal (T, 1).
            l (Union[Tensor, ndarray]): Input loudness features (T, 1).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T, out_channels)

        """

        x = pad_fn(x.transpose(1, 0).unsqueeze(0))
        l = l.transpose(1, 0).unsqueeze(0)
        f0 = f0.transpose(1, 0).unsqueeze(0)
        s = signal_generator(f0)

        return self.forward(x, s, l, spk_emb).squeeze(0).transpose(1, 0)


class HiFiGANPeriodDiscriminator(nn.Module):
    """HiFiGAN period discriminator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        period=3,
        kernel_sizes=[5, 3],
        channels=32,
        downsample_scales=[3, 3, 3, 3, 1],
        max_downsample_channels=1024,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        use_spectral_norm=False,
    ):
        """Initialize HiFiGANPeriodDiscriminator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            period (int): Period.
            kernel_sizes (list): Kernel sizes of initial conv layers and the final conv layer.
            channels (int): Number of initial channels.
            downsample_scales (list): List of downsampling scales.
            max_downsample_channels (int): Number of maximum downsampling channels.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_sizes[1] % 2 == 1, "Kernel size must be odd number."

        self.period = period
        self.convs = nn.ModuleList()
        in_chs = in_channels
        out_chs = channels
        for downsample_scale in downsample_scales:
            self.convs += [
                nn.Sequential(
                    nn.Conv2d(
                        in_chs,
                        out_chs,
                        (kernel_sizes[0], 1),
                        (downsample_scale, 1),
                        padding=((kernel_sizes[0] - 1) // 2, 0),
                    ),
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                )
            ]
            in_chs = out_chs
            # NOTE(kan-bayashi): Use downsample_scale + 1?
            out_chs = min(out_chs * 4, max_downsample_channels)
        self.output_conv = nn.Conv2d(
            out_chs,
            out_channels,
            (kernel_sizes[1] - 1, 1),
            1,
            padding=((kernel_sizes[1] - 1) // 2, 0),
        )

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, in_channels, T).
            return_fmaps (bool): Whether to return feature maps.
        Returns:
            list: List of each layer's tensors.
        """
        # transform 1d to 2d -> (B, C, T/P, P)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        # forward conv
        fmap = []
        for f in self.convs:
            x = f(x)
            if return_fmaps:
                fmap.append(x)
        x = self.output_conv(x)
        out = torch.flatten(x, 1, -1)

        if return_fmaps:
            return out, fmap
        else:
            return out

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """Apply spectral normalization module from all of the layers."""

        def _apply_spectral_norm(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.spectral_norm(m)
                logging.debug(f"Spectral norm is applied to {m}.")

        self.apply(_apply_spectral_norm)


class HiFiGANMultiPeriodDiscriminator(nn.Module):
    """HiFiGAN multi-period discriminator module."""

    def __init__(
        self,
        periods=[2, 3, 5, 7, 11],
        discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):
        """Initialize HiFiGANMultiPeriodDiscriminator module.
        Args:
            periods (list): List of periods.
            discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.
        """
        super().__init__()
        self.discriminators = nn.ModuleList()
        for period in periods:
            params = copy.deepcopy(discriminator_params)
            params["period"] = period
            self.discriminators += [HiFiGANPeriodDiscriminator(**params)]

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.
        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs, fmaps = [], []
        for f in self.discriminators:
            if return_fmaps:
                out, fmap = f(x, return_fmaps)
                fmaps.extend(fmap)
            else:
                out = f(x)
            outs.append(out)

        if return_fmaps:
            return outs, fmaps
        else:
            return outs


class HiFiGANScaleDiscriminator(nn.Module):
    """HiFi-GAN scale discriminator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_sizes=[15, 41, 5, 3],
        channels=128,
        max_downsample_channels=1024,
        max_groups=16,
        bias=True,
        downsample_scales=[2, 2, 4, 4, 1],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        use_spectral_norm=False,
    ):
        """Initilize HiFiGAN scale discriminator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of four kernel sizes. The first will be used for the first conv layer,
                and the second is for downsampling part, and the remaining two are for output layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()
        self.layers = nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 4
        for ks in kernel_sizes:
            assert ks % 2 == 1

        # add first layer
        self.layers += [
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    channels,
                    # NOTE(kan-bayashi): Use always the same kernel size
                    kernel_sizes[0],
                    bias=bias,
                    padding=(kernel_sizes[0] - 1) // 2,
                ),
                getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers
        in_chs = channels
        out_chs = channels
        # NOTE(kan-bayashi): Remove hard coding?
        groups = 4
        for downsample_scale in downsample_scales:
            self.layers += [
                nn.Sequential(
                    nn.Conv1d(
                        in_chs,
                        out_chs,
                        kernel_size=kernel_sizes[1],
                        stride=downsample_scale,
                        padding=(kernel_sizes[1] - 1) // 2,
                        groups=groups,
                        bias=bias,
                    ),
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                )
            ]
            in_chs = out_chs
            # NOTE(kan-bayashi): Remove hard coding?
            out_chs = min(in_chs * 2, max_downsample_channels)
            # NOTE(kan-bayashi): Remove hard coding?
            groups = min(groups * 4, max_groups)

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            nn.Sequential(
                nn.Conv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_sizes[2],
                    stride=1,
                    padding=(kernel_sizes[2] - 1) // 2,
                    bias=bias,
                ),
                getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.last_layer = nn.Conv1d(
            out_chs,
            out_channels,
            kernel_size=kernel_sizes[3],
            stride=1,
            padding=(kernel_sizes[3] - 1) // 2,
            bias=bias,
        )

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.
        Returns:
            List: List of output tensors of each layer.
        """
        fmap = []
        for f in self.layers:
            x = f(x)
            if return_fmaps:
                fmap.append(x)
        out = self.last_layer(x)

        if return_fmaps:
            return out, fmap
        else:
            return out

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """Apply spectral normalization module from all of the layers."""

        def _apply_spectral_norm(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.spectral_norm(m)
                logging.debug(f"Spectral norm is applied to {m}.")

        self.apply(_apply_spectral_norm)


class HiFiGANMultiScaleDiscriminator(nn.Module):
    """HiFi-GAN multi-scale discriminator module."""

    def __init__(
        self,
        scales=3,
        downsample_pooling="AvgPool1d",
        # follow the official implementation setting
        downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm=False,
    ):
        """Initilize HiFiGAN multi-scale discriminator module.
        Args:
            scales (int): Number of multi-scales.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementaion. The first discriminator uses spectral norm and the other
                discriminators use weight norm.
        """
        super().__init__()
        self.discriminators = nn.ModuleList()

        # add discriminators
        for i in range(scales):
            params = copy.deepcopy(discriminator_params)
            if follow_official_norm:
                if i == 0:
                    params["use_weight_norm"] = False
                    params["use_spectral_norm"] = True
                else:
                    params["use_weight_norm"] = True
                    params["use_spectral_norm"] = False
            self.discriminators += [HiFiGANScaleDiscriminator(**params)]
        self.pooling = getattr(nn, downsample_pooling)(**downsample_pooling_params)

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.
        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs, fmaps = [], []
        for f in self.discriminators:
            if return_fmaps:
                out, fmap = f(x, return_fmaps)
                fmaps.extend(fmap)
            else:
                out = f(x)
            outs.append(out)
            x = self.pooling(x)

        if return_fmaps:
            return outs, fmaps
        else:
            return outs


class HiFiGANMultiScaleMultiPeriodDiscriminator(nn.Module):
    """HiFi-GAN multi-scale + multi-period discriminator module."""

    def __init__(
        self,
        # Multi-scale discriminator related
        scales=3,
        scale_downsample_pooling="AvgPool1d",
        scale_downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        scale_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm=True,
        # Multi-period discriminator related
        periods=[2, 3, 5, 7, 11],
        period_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):
        """Initilize HiFiGAN multi-scale + multi-period discriminator module.
        Args:
            scales (int): Number of multi-scales.
            scale_downsample_pooling (str): Pooling module name for downsampling of the inputs.
            scale_downsample_pooling_params (dict): Parameters for the above pooling module.
            scale_discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementaion. The first discriminator uses spectral norm and the other
                discriminators use weight norm.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.
        """
        super().__init__()
        self.msd = HiFiGANMultiScaleDiscriminator(
            scales=scales,
            downsample_pooling=scale_downsample_pooling,
            downsample_pooling_params=scale_downsample_pooling_params,
            discriminator_params=scale_discriminator_params,
            follow_official_norm=follow_official_norm,
        )
        self.mpd = HiFiGANMultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=period_discriminator_params,
        )

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.
        Returns:
            List: List of list of each discriminator outputs,
                which consists of each layer output tensors.
                Multi scale and multi period ones are concatenated.
        """
        if return_fmaps:
            msd_outs, msd_fmaps = self.msd(x, return_fmaps)
            mpd_outs, mpd_fmaps = self.mpd(x, return_fmaps)
            outs = msd_outs + mpd_outs
            fmaps = msd_fmaps + mpd_fmaps
            return outs, fmaps
        else:
            msd_outs = self.msd(x)
            mpd_outs = self.mpd(x)
            outs = msd_outs + mpd_outs
            return outs
