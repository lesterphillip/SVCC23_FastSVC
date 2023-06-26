# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Tacotron2 Modules."""

import logging
import math
import six

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from joblib import load

from harana.layers import Conv1d1x1, Conv1d1x3, Conv2d1x3
from harana.layers import Squeeze2d, Stretch2d
from harana.layers import upsample
from harana.models.fastsvc import FastSVCFiLMNet
from harana import models
from harana.utils import read_hdf5, validate_length

def encoder_init(m):
    """Initialize encoder parameters."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))

class Taco2Encoder(torch.nn.Module):
    """Encoder module of the Tacotron2 TTS model.
    Reference:
    _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_
       https://arxiv.org/abs/1712.05884
    """

    def __init__(
        self,
        idim,
        elayers=1,
        eunits=512,
        econv_layers=3,
        econv_chans=512,
        econv_filts=5,
        use_batch_norm=True,
        use_residual=False,
        dropout_rate=0.5,
    ):
        """Initialize Tacotron2 encoder module.
        Args:
            idim (int) Dimension of the inputs.
            elayers (int, optional) The number of encoder blstm layers.
            eunits (int, optional) The number of encoder blstm units.
            econv_layers (int, optional) The number of encoder conv layers.
            econv_filts (int, optional) The number of encoder conv filter size.
            econv_chans (int, optional) The number of encoder conv filter channels.
            use_batch_norm (bool, optional) Whether to use batch normalization.
            use_residual (bool, optional) Whether to use residual connection.
            dropout_rate (float, optional) Dropout rate.
        """
        super(Taco2Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.use_residual = use_residual

        # define network layer modules
        self.input_layer = torch.nn.Linear(idim, econv_chans)

        if econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for layer in range(econv_layers):
                ichans = econv_chans
                if use_batch_norm:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.BatchNorm1d(econv_chans),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
                else:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
        else:
            self.convs = None
        if elayers > 0:
            iunits = econv_chans if econv_layers != 0 else embed_dim
            self.blstm = torch.nn.LSTM(
                iunits, eunits // 2, elayers, batch_first=True, bidirectional=True
            )
        else:
            self.blstm = None

        # initialize
        self.apply(encoder_init)

    def forward(self, xs, ilens=None):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of the padded acoustic feature sequence (B, Lmax, idim)
        """
        xs = self.input_layer(xs).transpose(1, 2)

        if self.convs is not None:
            for i in range(len(self.convs)):
                if self.use_residual:
                    xs += self.convs[i](xs)
                else:
                    xs = self.convs[i](xs)
        if self.blstm is None:
            return xs.transpose(1, 2)
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens)
        xs = pack_padded_sequence(xs.transpose(1, 2), ilens.cpu(), batch_first=True)
        self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Lmax, C)
        xs, hlens = pad_packed_sequence(xs, batch_first=True)

        return xs, hlens

class Taco2Prenet(torch.nn.Module):
    """Prenet module for decoder of Tacotron2.
    The Prenet preforms nonlinear conversion
    of inputs before input to auto-regressive lstm,
    which helps alleviate the exposure bias problem.
    Note:
        This module alway applies dropout even in evaluation.
        See the detail in `Natural TTS Synthesis by
        Conditioning WaveNet on Mel Spectrogram Predictions`_.
    _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_
       https://arxiv.org/abs/1712.05884
    """

    def __init__(self, idim, n_layers=2, n_units=256, dropout_rate=0.5):
        super(Taco2Prenet, self).__init__()
        self.dropout_rate = dropout_rate
        self.prenet = torch.nn.ModuleList()
        for layer in range(n_layers):
            n_inputs = idim if layer == 0 else n_units
            self.prenet += [
                torch.nn.Sequential(torch.nn.Linear(n_inputs, n_units), torch.nn.ReLU())
            ]

    def forward(self, x):
        # Make sure at least one dropout is applied.
        if len(self.prenet) == 0:
            return F.dropout(x, self.dropout_rate)

        for i in range(len(self.prenet)):
            x = F.dropout(self.prenet[i](x), self.dropout_rate)
        return x


class RNNCell(nn.Module):
    ''' RNN cell wrapper'''

    def __init__(self, input_dim, module, dim, dropout, layer_norm, proj):
        super(RNNCell, self).__init__()
        # Setup
        rnn_out_dim = dim
        self.out_dim = rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.proj = proj

        # Recurrent cell
        self.cell = getattr(nn, module.upper()+"Cell")(input_dim, dim)

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)

    def forward(self, input_x, z, c):

        # Forward RNN cell
        new_z, new_c = self.cell(input_x, (z, c))

        # Normalizations
        if self.layer_norm:
            new_z = self.ln(new_z)
        if self.dropout > 0:
            new_z = self.dp(new_z)

        if self.proj:
            new_z = torch.tanh(self.pj(new_z))

        return new_z, new_c


class RNNLayer(nn.Module):
    ''' RNN wrapper, includes time-downsampling'''

    def __init__(self, input_dim, module, bidirection, dim, dropout, layer_norm, sample_rate, proj):
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2 * dim if bidirection else dim
        self.out_dim = rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
        self.proj = proj

        # Recurrent layer
        self.layer = getattr(nn, module.upper())(
            input_dim, dim, bidirectional=bidirection, num_layers=1, batch_first=True)

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)

    def forward(self, input_x, x_len):

        # Forward RNN
        if not self.training:
            self.layer.flatten_parameters()

        input_x = pack_padded_sequence(input_x, x_len, batch_first=True, enforce_sorted=False)
        output, _ = self.layer(input_x)
        output, x_len = pad_packed_sequence(output, batch_first=True)

        # Normalizations
        if self.layer_norm:
            output = self.ln(output)
        if self.dropout > 0:
            output = self.dp(output)

        # Perform Downsampling
        if self.sample_rate > 1:
            output, x_len = downsample(output, x_len, self.sample_rate, 'drop')

        if self.proj:
            output = torch.tanh(self.pj(output))

        return output, x_len


class Taco2Postnet(torch.nn.Module):
    """Postnet module for Spectrogram prediction network.
    This is a module of Postnet in Spectrogram prediction network,
    which described in `Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The Postnet predicts refines the predicted
    Mel-filterbank of the decoder,
    which helps to compensate the detail structure of spectrogram.
    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884
    """

    def __init__(
        self,
        idim,
        odim,
        n_layers=5,
        n_chans=512,
        n_filts=5,
        dropout_rate=0.5,
        use_batch_norm=True,
    ):
        """Initialize postnet module.
        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            n_layers (int, optional): The number of layers.
            n_filts (int, optional): The number of filter size.
            n_units (int, optional): The number of filter channels.
            use_batch_norm (bool, optional): Whether to use batch normalization..
            dropout_rate (float, optional): Dropout rate..
        """
        super(Taco2Postnet, self).__init__()
        self.postnet = torch.nn.ModuleList()
        for layer in six.moves.range(n_layers - 1):
            ichans = odim if layer == 0 else n_chans
            ochans = odim if layer == n_layers - 1 else n_chans
            if use_batch_norm:
                self.postnet += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias=False,
                        ),
                        torch.nn.BatchNorm1d(ochans),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(dropout_rate),
                    )
                ]
            else:
                self.postnet += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias=False,
                        ),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(dropout_rate),
                    )
                ]
        ichans = n_chans if n_layers != 1 else odim
        if use_batch_norm:
            self.postnet += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                    torch.nn.BatchNorm1d(odim),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        else:
            self.postnet += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                    torch.nn.Dropout(dropout_rate),
                )
            ]

    def forward(self, xs):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of the sequences of padded input tensors (B, idim, Tmax).
        Returns:
            Tensor: Batch of padded output tensor. (B, odim, Tmax).
        """
        for i in six.moves.range(len(self.postnet)):
            xs = self.postnet[i](xs)
        return xs


class Tacotron2(nn.Module):
    def __init__(self,
                 #stats,
                 input_dim, # input dimensions (ppg: 144?)
                 output_dim, # mel dimensions (80)
                 hidden_dim=1024,
                 enc_layers=1,
                 dec_layers=2,
                 dec_dropout_rate=0.2,
                 dec_proj_dim=256,
                 dec_layernorm=False,
                 prenet_layers=2,
                 prenet_dim=256,
                 prenet_dropout_rate=0.5,
                 multi_speaker=False,
                 spk_emb_dim=512,
                 integrate_logf0=False,
                 use_postnet=False,
                 ar_mode=True,
                 **kwargs):
        super(Tacotron2, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoder = Taco2Encoder(
            idim=input_dim,
            elayers=enc_layers,
            eunits=hidden_dim
        )

        # define prenet
        self.prenet = Taco2Prenet(
            idim=output_dim,
            n_layers=prenet_layers,
            n_units=prenet_dim,
            dropout_rate=prenet_dropout_rate,
        )

        # define decoder (LSTMPs)
        self.decs = nn.ModuleList()
        self.ar_mode = ar_mode
        for i in range(dec_layers):
            if ar_mode:
                prev_dim = output_dim if prenet_layers == 0 else prenet_dim
                rnn_input_dim = hidden_dim + prev_dim if i == 0 else hidden_dim
                rnn_layer = RNNCell(
                    rnn_input_dim,
                    "LSTM",
                    hidden_dim,
                    dec_dropout_rate,
                    dec_layernorm,
                    proj=True,
                )
            else:
                rnn_input_dim = hidden_dim
                rnn_layer = RNNLayer(
                    rnn_input_dim,
                    "LSTM",
                    False,
                    hidden_dim,
                    dec_dropout_rate,
                    dec_layernorm,
                    sample_rate=1,
                    proj=True,
                )
            self.decs.append(rnn_layer)

        self.use_postnet = use_postnet
        if use_postnet:
            self.postnet = Taco2Postnet(
                idim=output_dim,
                odim=output_dim
            )

        self.integrate_logf0 = integrate_logf0
        if integrate_logf0:
            self.logf0_film_net = FastSVCFiLMNet(1)
            self.lft_film_net = FastSVCFiLMNet(1)

        # projection layer
        self.proj = torch.nn.Linear(hidden_dim, output_dim)

        self.multi_speaker = multi_speaker
        if multi_speaker:
            self.spk_emb_projection = torch.nn.Linear(hidden_dim + spk_emb_dim, hidden_dim)

        self.instance_norm = nn.InstanceNorm2d(hidden_dim)
        self.bap_instance_norm = nn.InstanceNorm2d(hidden_dim)
        bap_dim = 258
        self.bap_decoder = torch.nn.Sequential(
            nn.Conv1d(bap_dim, int(bap_dim/3), kernel_size=3, padding=2, dilation=2),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(int(bap_dim/3), int(bap_dim/6), kernel_size=3, padding=2, dilation=2),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(int(bap_dim/6), int(bap_dim/12), kernel_size=1, padding=1, dilation=1),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(int(bap_dim/12), int(bap_dim/84), kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(3)
        )


    def feature_affine(self, x, logf0, lft):
        """Fuse in sine wave and linguistic information.

        Args:
            x (Tensor): Hidden dimension linguistic features (B, Lmax, idim).
            s (Tuple of Tensors): Downsampled sine wave (B, Lmax, idim).

        Returns:
            Tensor: Output tensor (B, out_channels, T)
        """

        s_scale, s_shift = logf0
        l_scale, l_shift = lft
        scale = s_scale + l_scale
        shift = s_shift + l_shift
        x = torch.mul(scale, x)
        x = x + shift

        return x

    def forward(self, features, lens, lft, logf0, spk_embs=None, targets=None):
        """Calculate forward propagation.
            Args:
            features: Batch of the sequences of input features (B, Lmax, idim).
            targets: Batch of the sequences of padded target features (B, Lmax, odim).
        """
        B = features.shape[0]
        if targets is not None:
            targets, bap_ = torch.split(targets, [60, 3], dim=2)

        # encoder
        encoder_states, lens = self.encoder(features, lens) # (B, Lmax, hidden_dim)

        # bap multi-stream
        bap_features = torch.concat([features, logf0, lft], dim=2)

        # conv layers
        bap_out = self.bap_decoder(bap_features.transpose(2, 1))
        bap_out = self.bap_instance_norm(bap_out)
        encoder_states = self.instance_norm(encoder_states)

        # logF0 and lft integration with FiLM
        if self.integrate_logf0:
            logf0 = self.logf0_film_net(logf0.transpose(2, 1))
            lft = self.lft_film_net(lft.transpose(2, 1))
            encoder_states = self.feature_affine(encoder_states.transpose(2, 1), logf0, lft)
            encoder_states = encoder_states.transpose(2, 1)

        # Integrate spk embeddings
        if self.multi_speaker:
            spk_embs = F.normalize(spk_embs).transpose(2, 1).expand(-1, encoder_states.size(1), -1)
            encoder_states = self.spk_emb_projection(torch.cat([encoder_states, spk_embs], dim=-1))

        # decoder: LSTMP layers & projection
        if self.ar_mode:

            if targets is not None:
                targets = targets.transpose(0, 1) # (Lmax, B, output_dim)

            predicted_list = []

            # initialize hidden states
            c_list = [encoder_states.new_zeros(B, self.hidden_dim)]
            z_list = [encoder_states.new_zeros(B, self.hidden_dim)]
            for _ in range(1, len(self.decs)):
                c_list += [encoder_states.new_zeros(B, self.hidden_dim)]
                z_list += [encoder_states.new_zeros(B, self.hidden_dim)]
            prev_out = encoder_states.new_zeros(B, self.output_dim)

            # step-by-step loop for autoregressive decoding
            for t, encoder_state in enumerate(encoder_states.transpose(0, 1)):
                concat = torch.cat([encoder_state, self.prenet(prev_out)], dim=1) # each encoder_state has shape (B, hidden_dim)
                for i, lstmp in enumerate(self.decs):
                    lstmp_input = concat if i == 0 else z_list[i-1]
                    z_list[i], c_list[i] = lstmp(lstmp_input, z_list[i], c_list[i])
                predicted_list += [self.proj(z_list[-1]).view(B, self.output_dim, -1)] # projection is done here to ensure output dim
                prev_out = targets[t] if targets is not None else predicted_list[-1].squeeze(-1) # targets not None = teacher-forcing

                prev_out = prev_out.float()
            predicted = torch.cat(predicted_list, dim=2)
        else:
            predicted = encoder_states
            for i, lstmp in enumerate(self.decs):
                predicted, lens = lstmp(predicted, lens)

            # projection layer
            raw_predicted = self.proj(predicted).transpose(1, 2)

        value = min(predicted.shape[-1], bap_out.shape[-1])
        predicted = predicted.narrow(-1, 0, value)
        bap_out = bap_out.narrow(-1, 0, value)
        predicted = torch.concat([predicted, bap_out], dim=1)
        predicted = predicted.transpose(1, 2)  # (B, hidden_dim, Lmax) -> (B, Lmax, hidden_dim)

        return predicted, lens

class Tacotron2Wrapper(torch.nn.Module):

    def __init__(
        self,
        #stats,
        input_dim, # input dimensions (ppg: 144?)
        output_dim, # mel dimensions (80)
        hidden_dim=1024,
        enc_layers=1,
        dec_layers=2,
        dec_dropout_rate=0.2,
        dec_proj_dim=256,
        dec_layernorm=False,
        prenet_layers=2,
        prenet_dim=256,
        prenet_dropout_rate=0.5,
        multi_speaker=True,
        spk_emb_dim=512,
        integrate_logf0=True,
        use_postnet=True,
        ar_mode=True,
    ):
        super(Tacotron2Wrapper, self).__init__()
        self.acoustic_network = Tacotron2(
             #stats,
             input_dim=input_dim, # input dimensions (ppg: 144?)
             output_dim=output_dim, # mel dimensions (80)
             hidden_dim=hidden_dim,
             enc_layers=enc_layers,
             dec_layers=dec_layers,
             dec_dropout_rate=dec_dropout_rate,
             dec_proj_dim=dec_proj_dim,
             dec_layernorm=dec_layernorm,
             prenet_layers=prenet_layers,
             prenet_dim=prenet_dim,
             prenet_dropout_rate=prenet_dropout_rate,
             multi_speaker=multi_speaker,
             spk_emb_dim=spk_emb_dim,
             integrate_logf0=integrate_logf0,
             use_postnet=use_postnet,
             ar_mode=ar_mode,
        )

    def forward(self, features, lens, lft, logf0, spk_embs=None, targets=None):
        return self.acoustic_network(features, lens, lft, logf0, spk_embs, targets)

class SubFreqDiscriminator(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        layers=4,
        kernel_size=9,
        channels=64,
        ):
        super(SubFreqDiscriminator, self).__init__()

        self.layers = torch.nn.ModuleList()

        for index in range(layers):
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        padding=4,
                        dilation=1,
                        bias=False
                    ),
                    torch.nn.LeakyReLU(
                        negative_slope=0.2,
                        inplace= True
                    )
            )]
            in_channels=channels

        self.layers += [
            torch.nn.Conv2d(
                in_channels=channels,
                out_channels=1,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=True
            )]


    def forward(self, x):
        x = x.unsqueeze(1)
        for f in self.layers:
            x = f(x)
        return x.squeeze(1)


class MultiSubFreqDiscriminator(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        layers=4,
        kernel_size=9,
        channels=64,
        batch_max_frames=75,
        ):
        super(MultiSubFreqDiscriminator, self).__init__()

        self.batch_max_frames = batch_max_frames
        self.low_discriminator = SubFreqDiscriminator()
        self.mid_discriminator = SubFreqDiscriminator()
        self.high_discriminator = SubFreqDiscriminator()


    def unpack_sequence(self, packed_sequence, lengths):
        """ Removes the padding from the batch tensor.
        """
        assert isinstance(packed_sequence, PackedSequence)
        head = 0
        trailing_dims = packed_sequence.data.shape[1:]
        unpacked_sequence = [torch.zeros(l, *trailing_dims) for l in lengths]
        # l_idx - goes from 0 - maxLen-1
        for l_idx, b_size in enumerate(packed_sequence.batch_sizes):
            for b_idx in range(b_size):
                unpacked_sequence[b_idx][l_idx] = packed_sequence.data[head]
                head += 1
        return unpacked_sequence


    def slice_dataset(self, x, lengths, batch_max_frames):
        """ Randomly selects a window of each batch.
        """
        device = x.device
        x_nopad = pack_padded_sequence(x, lengths, batch_first=True)
        x_nopad = self.unpack_sequence(x_nopad, lengths)

        new_tensor_batch = []
        for x in x_nopad:
            interval_end = len(x) - batch_max_frames
            try:
                start_frame = np.random.randint(0, interval_end)
            except:
                continue
            x = x[start_frame : start_frame + batch_max_frames]
            new_tensor_batch.append(x.unsqueeze(0))

        return torch.concat(new_tensor_batch, dim=0).to(device)

    def forward(self, x, lengths):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, T, D).
        Returns:
            Tensor: Output tensor (B, T, D)
        """

        x = self.slice_dataset(x, lengths, self.batch_max_frames)
        # B, T, D
        x = x.unfold(dimension=2, size=30, step=15).transpose(3, 2)
        # B, T, D', 3
        x_high, x_mid, x_low = torch.split(x, [1, 1, 1], dim=3)
        # B, T, D', 1
        outs_low = self.low_discriminator(x_low.squeeze(3))
        outs_mid = self.mid_discriminator(x_mid.squeeze(3))
        outs_high = self.high_discriminator(x_high.squeeze(3))

        return [outs_low, outs_mid, outs_high]
