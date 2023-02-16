#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Training script for FastSVC.
References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/chomeyama/HN-UnifiedSourceFilterGAN
"""

import argparse
import logging
import os
import sys

from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch
import yaml

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import harana
import harana.models
import harana.optimizers

from harana.datasets import FastSVCDataset
from harana.losses import DiscriminatorAdversarialLoss
from harana.losses import GeneratorAdversarialLoss
from harana.losses import MultiResolutionSTFTLoss
from harana.utils import read_hdf5
from harana.utils.features import SignalGenerator

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class Trainer(object):
    """Customized trainer module for HARANA training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        sampler,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.
        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.
        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = {
            "generator": self.model["generator"].state_dict(),
            "discriminator": self.model["discriminator"].state_dict(),
        }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model["generator"].load_state_dict(state_dict["model"]["generator"])
        self.model["discriminator"].load_state_dict(
            state_dict["model"]["discriminator"]
        )
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(
                state_dict["optimizer"]["generator"]
            )
            self.optimizer["discriminator"].load_state_dict(
                state_dict["optimizer"]["discriminator"]
            )
            self.scheduler["generator"].load_state_dict(
                state_dict["scheduler"]["generator"]
            )
            self.scheduler["discriminator"].load_state_dict(
                state_dict["scheduler"]["discriminator"]
            )

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        x, y = batch
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)

        #######################
        #      Generator      #
        #######################
        if self.steps > self.config.get("generator_train_start_steps", 0):
            y_ = self.model["generator"](*x)

            # initialize
            gen_loss = 0.0

            # multi-resolution sfft loss
            if self.config["use_stft_loss"]:
                sc_loss, mag_loss = self.criterion["stft"](y_, y)
                gen_loss += sc_loss + mag_loss
                self.total_train_loss[
                    "train/spectral_convergence_loss"
                ] += sc_loss.item()
                self.total_train_loss[
                    "train/log_stft_magnitude_loss"
                ] += mag_loss.item()

            # weighting aux loss
            gen_loss *= self.config.get("lambda_aux", 1.0)

            # adversarial loss
            if self.steps > self.config["discriminator_train_start_steps"]:
                p_ = self.model["discriminator"](y_)
                adv_loss = self.criterion["gen_adv"](p_)
                self.total_train_loss["train/adversarial_loss"] += adv_loss.item()

                # add adversarial loss to generator loss
                gen_loss += self.config["lambda_adv"] * adv_loss

            self.total_train_loss["train/generator_loss"] += gen_loss.item()

            # update generator
            self.optimizer["generator"].zero_grad()
            gen_loss.backward()
            if self.config["generator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["generator"].parameters(),
                    self.config["generator_grad_norm"],
                )
            self.optimizer["generator"].step()
            self.scheduler["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                y_ = self.model["generator"](*x)

            # discriminator loss
            p = self.model["discriminator"](y)
            p_ = self.model["discriminator"](y_.detach())
            real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
            dis_loss = real_loss + fake_loss
            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/discriminator_loss"] += dis_loss.item()

            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["discriminator_grad_norm"],
                )
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            self._check_log_interval()
            self._check_eval_interval()
            self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x, y = batch
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)

        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](*x)

        # initialize
        aux_loss = 0.0

        # multi-resolution stft loss
        if self.config["use_stft_loss"]:
            sc_loss, mag_loss = self.criterion["stft"](y_, y)
            aux_loss += sc_loss + mag_loss
            self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
            self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()

        # weighting stft loss
        aux_loss *= self.config.get("lambda_aux", 1.0)

        # adversarial loss
        p_ = self.model["discriminator"](y_)
        adv_loss = self.criterion["gen_adv"](p_)
        gen_loss = aux_loss + self.config["lambda_adv"] * adv_loss

        #######################
        #    Discriminator    #
        #######################
        p = self.model["discriminator"](y)
        p_ = self.model["discriminator"](y_)

        # discriminator loss
        real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
        dis_loss = real_loss + fake_loss

        # add to total eval loss
        self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
        self.total_eval_loss["eval/real_loss"] += real_loss.item()
        self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
        self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["dev"], desc="[eval]"), 1
        ):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._generate_and_save_intermediate_result(batch)

        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        # generate
        x_batch, y_batch = batch
        x_batch = tuple([x.to(self.device) for x in x_batch])
        y_batch = y_batch.to(self.device)
        y_batch_ = self.model["generator"](*x_batch)

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 1):
            # convert to ndarray
            y, y_ = y.view(-1).cpu().numpy(), y_.view(-1).cpu().numpy()

            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(
                figname.replace(".png", "_ref.wav"),
                y,
                self.config["sampling_rate"],
                "PCM_16",
            )
            sf.write(
                figname.replace(".png", "_gen.wav"),
                y_,
                self.config["sampling_rate"],
                "PCM_16",
            )

            if idx >= self.config["num_save_intermediate_results"]:
                break

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(
        self,
        batch_length=19200,
        sample_rate=16000,
        hop_size=640,
        aux_context_window=0,
        use_noise_input=False,
        sine_amp=0.1,
        noise_amp=0.003,
        sine_f0_type="f0",
        signal_types=["sine"],
        use_spk_emb=False,
    ):
        """Initialize customized collater for PyTorch DataLoader.
        Args:
            batch_length (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.
        """
        if batch_length % hop_size != 0:
            batch_length += -(batch_length % hop_size)
        assert batch_length % hop_size == 0
        self.batch_length = batch_length
        self.batch_frames = batch_length // hop_size
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.use_noise_input = use_noise_input

        self.sine_f0_type = sine_f0_type
        self.signal_generator = SignalGenerator(
            sample_rate=sample_rate,
            hop_size=hop_size,
            sine_amp=sine_amp,
            noise_amp=noise_amp,
            signal_types=signal_types,
        )
        # set useful values in random cutting
        self.start_offset = aux_context_window
        self.end_offset = -(self.batch_frames + aux_context_window)
        self.mel_threshold = self.batch_frames + 2 * aux_context_window
        self.use_spk_emb = use_spk_emb

    def __call__(self, batch):
        """Convert into batch tensors.
        Args:
            batch (list): list of tuple of the pair of audio and features.
        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where
                T = (T' - 2 * aux_context_window) * hop_size.
            Tensor: Target signal batch (B, 1, T).
        """
        # time resolution check
        y_batch, f0_batch, ppg_batch, lft_batch, emb_batch = [], [], [], [], []
        for idx in range(len(batch)):
            x, f0, ppg, lft, emb = batch[idx]
            self._check_length(x, f0, ppg, lft, 0)

            # randomly cut audio with length of batch_frames
            if len(ppg) - 2 * self.aux_context_window > self.batch_frames:
                interval_start = self.aux_context_window
                interval_end = len(ppg) - self.batch_frames - self.aux_context_window
                start_frame = np.random.randint(interval_start, interval_end)
                start_step = start_frame * self.hop_size
                y = x[start_step : start_step + self.batch_length]
                f0 = f0[start_frame : start_frame + self.batch_frames]
                ppg = ppg[
                    start_frame
                    - self.aux_context_window : start_frame
                    + self.aux_context_window
                    + self.batch_frames
                ]
                lft = lft[start_step : start_step + self.batch_length]
                self._check_length(
                    y,
                    f0,
                    ppg,
                    lft,
                    self.aux_context_window,
                )
            # the audio sample is smaller than your batch_size
            else:
                logging.warn(
                    f"Audio sample is smaller than the specified batch length. Omitting sample."
                )
                continue
            y_batch += [y.astype(np.float32).reshape(-1, 1)]  # [(T, 1), ...]
            f0_batch += [f0.astype(np.float32).reshape(-1, 1)]  # [(T', 1), ...]
            ppg_batch += [
                ppg.astype(np.float32)
            ]  # [(T' + 2 * aux_context_window, D), ...]
            lft_batch += [lft.astype(np.float32).reshape(-1, 1)]  # [(T, 1), ...]
            emb_batch += [emb.astype(np.float32).squeeze(1)]  # [(T, ), ...]

        # convert each batch to tensor, asuume that each item in batch has the same length
        y_batch = torch.FloatTensor(np.array(y_batch)).transpose(2, 1)  # (B, 1, T)
        f0_batch = torch.FloatTensor(np.array(f0_batch)).transpose(2, 1)  # (B, 1, T')
        ppg_batch = torch.FloatTensor(np.array(ppg_batch)).transpose(
            2, 1
        )  # (B, C, T' + 2 * aux_context_window)
        lft_batch = torch.FloatTensor(np.array(lft_batch)).transpose(2, 1)  # (B, 1, T)
        emb_batch = torch.FloatTensor(np.array(emb_batch))  # (B, S)

        # make input signal batch tensor
        sine_batch = self.signal_generator(f0_batch)

        if self.use_spk_emb:
            return (ppg_batch, sine_batch, lft_batch, emb_batch), y_batch

        return (ppg_batch, sine_batch, lft_batch), y_batch

    def _check_length(self, x, f0, ppg, lft, context_window):
        """Assert the audio and feature lengths are correctly cut for upsampling."""
        assert len(x) == (len(ppg) - 2 * context_window) * self.hop_size
        assert len(x) == len(lft)
        assert len(x) == len(f0) * self.hop_size, f"{len(x)} vs {len(f0)}"


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description=("Train using HARANA toolkit harana/bin/train.py).")
    )
    parser.add_argument(
        "--train-dumpdir",
        default=None,
        type=str,
        help=(
            "directory including training data. "
            "you need to specify either train-*-scp or train-dumpdir."
        ),
    )
    parser.add_argument(
        "--dev-dumpdir",
        default=None,
        type=str,
        help=(
            "directory including development data. "
            "you need to specify either dev-*-scp or dev-dumpdir."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--pretrain",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to load pretrained params. (default="")',
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # check arguments
    if args.train_dumpdir is None:
        raise ValueError("Please specify either --train-dumpdir.")
    if args.dev_dumpdir is None:
        raise ValueError("Please specify either --dev-dumpdir.")

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    if config["remove_short_samples"]:
        mel_length_threshold = config["batch_length"] // config[
            "hop_size"
        ] + 2 * config["generator_params"].get("aux_context_window", 0)
    else:
        mel_length_threshold = None

    # query functions for loading the dataset
    train_dataset = FastSVCDataset(root_dir=args.train_dumpdir)
    logging.info(f"The number of training files = {len(train_dataset)}.")
    dev_dataset = FastSVCDataset(root_dir=args.dev_dumpdir)
    logging.info(f"The number of development files = {len(dev_dataset)}.")
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    # get data loader
    collater = Collater(
        batch_length=config["batch_length"],
        hop_size=config["hop_size"],
        use_spk_emb=config["generator_params"].get("use_spk_emb", False),
        # keep compatibility
        aux_context_window=config["aux_context_window"],
        # keep compatibility
        sine_amp=config["signal_generator"].get("sine_amp", 0.1),
        noise_amp=config["signal_generator"].get("noise_amp", 0.003),
        sine_f0_type=config["signal_generator"].get("sine_f0_type", "f0"),
        signal_types=config["signal_generator"].get("signal_types", ["sine"]),
    )

    sampler = {"train": None, "dev": None}
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["train"],
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["dev"],
            pin_memory=config["pin_memory"],
        ),
    }

    # define models
    generator_class = getattr(
        harana.models,
        # keep compatibility
        config.get("generator_type", "FastSVCGenerator"),
    )
    discriminator_class = getattr(
        harana.models,
        # keep compatibility
        config.get("discriminator_type", "MelGANMultiScaleDiscriminator"),
    )
    model = {
        "generator": generator_class(
            **config["generator_params"],
        ).to(device),
        "discriminator": discriminator_class(
            **config["discriminator_params"],
        ).to(device),
    }

    # define criterions
    criterion = {
        "gen_adv": GeneratorAdversarialLoss(
            # keep compatibility
            **config.get("generator_adv_loss_params", {})
        ).to(device),
        "dis_adv": DiscriminatorAdversarialLoss(
            # keep compatibility
            **config.get("discriminator_adv_loss_params", {})
        ).to(device),
    }

    config["use_stft_loss"] = True
    criterion["stft"] = MultiResolutionSTFTLoss(
        **config["stft_loss_params"],
    ).to(device)

    # define optimizers and schedulers
    generator_optimizer_class = getattr(
        harana.optimizers,
        # keep compatibility
        config.get("generator_optimizer_type", "RAdam"),
    )
    discriminator_optimizer_class = getattr(
        harana.optimizers,
        # keep compatibility
        config.get("discriminator_optimizer_type", "RAdam"),
    )
    optimizer = {
        "generator": generator_optimizer_class(
            model["generator"].parameters(),
            **config["generator_optimizer_params"],
        ),
        "discriminator": discriminator_optimizer_class(
            model["discriminator"].parameters(),
            **config["discriminator_optimizer_params"],
        ),
    }
    generator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("generator_scheduler_type", "StepLR"),
    )
    discriminator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("discriminator_scheduler_type", "StepLR"),
    )
    scheduler = {
        "generator": generator_scheduler_class(
            optimizer=optimizer["generator"],
            **config["generator_scheduler_params"],
        ),
        "discriminator": discriminator_scheduler_class(
            optimizer=optimizer["discriminator"],
            **config["discriminator_scheduler_params"],
        ),
    }

    # show settings
    logging.info(model["generator"])
    logging.info(model["discriminator"])
    logging.info(optimizer["generator"])
    logging.info(optimizer["discriminator"])
    logging.info(scheduler["generator"])
    logging.info(scheduler["discriminator"])
    for criterion_ in criterion.values():
        logging.info(criterion_)

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.pretrain) != 0:
        trainer.load_checkpoint(args.pretrain, load_only_params=True)
        logging.info(f"Successfully load parameters from {args.pretrain}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # run training loop
    try:
        trainer.run()
    finally:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
