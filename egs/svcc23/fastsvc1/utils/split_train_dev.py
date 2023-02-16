#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Splits the source scp file into a train/dev file while ensuring that all speakers are added to the dev file.
"""

import os
import argparse


def split_wav_scp(filename, train_filename, dev_filename, dev_count):
    with open(filename, "r") as f, open(train_filename, "w") as train_f, open(
        dev_filename, "w"
    ) as dev_f:
        speaker_counts = {}
        for line in f:
            values = line.strip().split(" ")
            if len(values) != 2:
                print("Skipping invalid line: ", line)
                continue
            utt_id, wav_path = values
            speaker_id = utt_id.split("_")[0]

            if speaker_id not in speaker_counts:
                speaker_counts[speaker_id] = 0

            if speaker_counts[speaker_id] < dev_count:
                dev_f.write("{} {}\n".format(utt_id, wav_path))
                speaker_counts[speaker_id] += 1
            else:
                train_f.write("{} {}\n".format(utt_id, wav_path))


def main():
    """Run process."""
    parser = argparse.ArgumentParser(
        description="file name",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source_scp", type=str, help="Input source file scp")
    parser.add_argument(
        "--train_scp_out", type=str, help="Output file path for train scp"
    )
    parser.add_argument("--dev_scp_out", type=str, help="Output file path for dev scp")
    parser.add_argument(
        "--dev_count",
        type=int,
        help="Number of utterances for each speaker in dev file",
    )

    args = parser.parse_args()
    filename = args.source_scp
    train_scp = args.train_scp_out
    dev_scp = args.dev_scp_out
    dev_count = args.dev_count
    split_wav_scp(filename, train_scp, dev_scp, dev_count)


if __name__ == "__main__":
    main()
