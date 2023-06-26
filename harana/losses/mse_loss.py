# Copyright 2023 Lester Violeta
#  MIT License (https://opensource.org/licenses/MIT)

"""MSE loss modules."""

from distutils.version import LooseVersion

import librosa
import torch
import torch.nn.functional as F

from harana.utils import make_non_pad_mask

class MSELoss(torch.nn.Module):
    """
    L1 loss module supporting (1) loss calculation in the normalized target feature space
                              (2) masked loss calculation
    """
    def __init__(self):
        super(MSELoss, self).__init__()
        self.objective = torch.nn.MSELoss(reduction="mean")

    def forward(self, x, y, x_lens, y_lens, device="cuda"):
        # match the input feature length to acoustic feature length to calculate the loss
        if x.shape[1] > y.shape[1]:
            x = x[:, :y.shape[1]]
            masks = make_non_pad_mask(y_lens).unsqueeze(-1).to(device)
        if x.shape[1] <= y.shape[1]:
            y = y[:, :x.shape[1]]
            masks = make_non_pad_mask(x_lens).unsqueeze(-1).to(device)

        # calculate masked loss
        x_masked = x.masked_select(masks)
        y_masked = y.masked_select(masks)
        loss = self.objective(x_masked, y_masked)
        return loss
