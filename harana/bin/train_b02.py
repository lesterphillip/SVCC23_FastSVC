#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Training script for SVCC23 B02 model.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
"""

import argparse
import logging
import os
import sys

from collections import defaultdict
from joblib import load

import matplotlib
import numpy as np
import soundfile as sf
import torch
import yaml

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import harana
import harana.models
import harana.optimizers

from harana.datasets import Taco2Dataset
from harana.losses import DiscriminatorAdversarialLoss
from harana.losses import GeneratorAdversarialLoss
from harana.losses import MSELoss
from harana.utils import read_hdf5, validate_length

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
        y = tuple([y_.to(self.device) for y_ in y])

        inputs, ilens, lft, logf0, spk_embs = x
        outputs, olens = y

        gen_loss = 0.0
        #######################
        #      Generator      #
        #######################
        if self.steps > self.config.get("generator_train_start_steps", 0):

            # generate predicted aux feature
            y_ = self.model["generator"](
                inputs,
                ilens,
                lft,
                logf0,
                spk_embs,
                outputs
            )

            # calculate loss
            loss = 0.0
            mse_loss = self.criterion["mse_loss"](
                y_[0], #outs,
                y[0], #trg,
                y_[1],
                y[1]
            )
            gen_loss += self.config.get("lambda_l1", 1.0) * mse_loss
            self.total_train_loss["train/mse_loss"] += mse_loss.item()

            # adversarial loss
            if self.steps > self.config["discriminator_train_start_steps"]:
                p_ = self.model["discriminator"](y_[0].detach(), y_[1].detach().to("cpu"))
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
                y_ = self.model["generator"](
                    inputs,
                    ilens,
                    lft,
                    logf0,
                    spk_embs,
                    outputs
                )

            # discriminator loss
            p = self.model["discriminator"](y[0], y[1].to("cpu"))
            p_ = self.model["discriminator"](y_[0].detach(), y_[1].detach().to("cpu"))
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
            if self.config["rank"] == 0:
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
        y = tuple([y_.to(self.device) for y_ in y])

        inputs, ilens, lft, logf0, spk_embs = x
        outputs, olens = y

        #######################
        #      Generator      #
        #######################
        aux_loss = 0.0
        if self.steps > self.config.get("generator_train_start_steps", 0):

            # generate predicted aux feature
            y_ = self.model["generator"](
                inputs,
                ilens,
                lft,
                logf0,
                spk_embs,
                outputs
            )

            # calculate loss
            loss = 0.0
            mse_loss = self.criterion["mse_loss"](
                y_[0], #outs,
                y[0], #trg,
                y_[1],
                y[1]
            )
            aux_loss += self.config.get("lambda_l1", 1.0) * mse_loss
            self.total_train_loss["train/mse_loss"] += mse_loss.item()

        # weighting stft loss
        aux_loss *= self.config.get("lambda_aux", 1.0)
        self.total_eval_loss["eval/aux_loss"] += aux_loss.item()

        # adversarial loss
        if self.steps > self.config["discriminator_train_start_steps"]:

            p_ = self.model["discriminator"](y_[0].detach(), y_[1].detach().to("cpu"))
            adv_loss = self.criterion["gen_adv"](p_)
            gen_loss = aux_loss + self.config["lambda_adv"] * adv_loss


        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            p = self.model["discriminator"](y[0], y[1].to("cpu"))
            #p_ = self.model["discriminator"](y_)
            p_ = self.model["discriminator"](y_[0].detach(), y_[1].detach().to("cpu"))

            # discriminator loss
            real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
            dis_loss = real_loss + fake_loss

            # add to total eval loss
            self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
            self.total_eval_loss["eval/real_loss"] += real_loss.item()
            self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
            self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()
            self.total_eval_loss["eval/generator_loss"] += gen_loss.item()

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
        aux_features="world",
    ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_length (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.

        """
        self.aux_features = aux_features

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
        sorted_batch = sorted(batch, key=lambda x: -x[1].shape[0])
        bs = len(sorted_batch)

        input_batch_nopad = [torch.from_numpy(sorted_batch[i][0]) for i in range(bs)]
        input_batch = pad_sequence(input_batch_nopad, batch_first=True)
        input_lengths = torch.from_numpy(np.array([i.size(0) for i in input_batch_nopad]))
        input_lft_nopad = [torch.from_numpy(sorted_batch[i][2]) for i in range(bs)]
        input_lft = pad_sequence(input_lft_nopad, batch_first=True)

        input_logf0_nopad = [torch.from_numpy(sorted_batch[i][3]) for i in range(bs)]
        input_logf0 = pad_sequence(input_logf0_nopad, batch_first=True)

        output_batch_nopad = [torch.from_numpy(sorted_batch[i][1]) for i in range(bs)]
        output_lengths = torch.from_numpy(np.array([o.size(0) for o in output_batch_nopad]))
        output_batch = pad_sequence(output_batch_nopad, batch_first=True)
        spk_embs = torch.from_numpy(np.array([sorted_batch[i][4] for i in range(bs)]))

        return (input_batch, input_lengths, input_lft, input_logf0, spk_embs), (output_batch, output_lengths)



def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description=(
            "Train using HARANA toolkit harana/bin/train.py)."
        )
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
        "--stats",
        type=str,
        required=True,
        help="statistics file.",
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
    parser.add_argument(
        "--rank",
        "--local_rank",
        default=0,
        type=int,
        help="rank for distributed training. no need to explictly specify.",
    )
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

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

    mel_length_threshold = None

    # query functions for loading the dataset

    scaler = load(args.stats)
    config["stats"] = {
        "mean": torch.from_numpy(scaler["mcep"].mean_).float().to(device),
        "scale": torch.from_numpy(scaler["mcep"].scale_).float().to(device),
    }

    #mcep_scaler = scaler["mcep"]
    #bap_scaler = scaler["mcep"]

    query = "*.h5"
    train_dataset = Taco2Dataset(
        root_dir=args.train_dumpdir,
    )
    logging.info(f"The number of training files = {len(train_dataset)}.")
    dev_dataset = Taco2Dataset(
        root_dir=args.dev_dumpdir,
    )
    logging.info(f"The number of development files = {len(dev_dataset)}.")

    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    # get data loader
    collater = Collater(
        aux_features=config["aux_features"]
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
        config.get("generator_type", "Tacotron2"),
    )
    discriminator_class = getattr(
        harana.models,
        # keep compatibility
        config.get("discriminator_type", "MultiSubFreqDiscriminator"),
    )
    model = {
        "generator": generator_class(
            #stats=config["stats"],
            **config["generator_params"],
        ).to(device),
        "discriminator": discriminator_class(
            **config["discriminator_params"],
        ).to(device),
    }

    # define criterions
    criterion = {
        "mse_loss": MSELoss(
            # keep compatibility
        ).to(device),
        "gen_adv": GeneratorAdversarialLoss(
            # keep compatibility
            **config.get("generator_adv_loss_params", {})
        ).to(device),
        "dis_adv": DiscriminatorAdversarialLoss(
            # keep compatibility
            **config.get("discriminator_adv_loss_params", {})
        ).to(device),
    }
    if config.get("use_feat_match_loss", False):  # keep compatibility
        criterion["feat_match"] = FeatureMatchLoss(
            # keep compatibility
            **config.get("feat_match_loss_params", {}),
        ).to(device)
    else:
        config["use_feat_match_loss"] = False

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
