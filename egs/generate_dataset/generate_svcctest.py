# Copyright 2023 Nagoya University (Lester Violeta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import pandas as pd
import argparse

from pydub import AudioSegment
from tqdm import tqdm


def change_id(spk):
    if spk == "M04":
        return "IDM1"
    elif spk == "F01":
        return "IDF1"
    elif spk == "M03":
        return "CDM1"
    elif spk == "F02":
        return "CDF1"
    elif spk == "M02":
        return "SM1"
    elif spk == "F04":
        return "SF1"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate SVCC23 test set"
        )
    )
    parser.add_argument(
        "--rootdir",
        required=True,
        type=str,
        help="NHSS dataset path",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=str,
        help=(
            "Output directory for generated wav files"
        ),
    )
    parser.add_argument(
        "--testcsv",
        required=True,
        type=str,
        help=(
            "Location of csv for generating test files"
        ),
    )
    args = parser.parse_args()
    df = pd.read_csv(args.testcsv)

    # create directories
    if not os.path.exists(f"{args.outdir}/ground_truth"):
       os.makedirs(f"{args.outdir}/ground_truth")
    if not os.path.exists(f"{args.outdir}/evaluation"):
       os.makedirs(f"{args.outdir}/evaluation")

    for index, row in tqdm(df.iterrows()):

        # parse information from csv
        spk = row['id'].split("_")[0]
        song = row['id'].split("_")[1]
        audio = AudioSegment.from_wav(f"{args.rootdir}/{spk}/{song}/song.wav")

        # cut files from the raw data
        start = row['start'] * 1000
        stop = row['stop'] * 1000

        audio_chunk = audio[float(start):float(stop)]
        new_spk = change_id(spk)

        # save files
        # evaluation set
        if new_spk == "SM1" or new_spk == "SF1":
            if not os.path.exists(f"{args.outdir}/evaluation/{new_spk}"):
               os.makedirs(f"{args.outdir}/evaluation/{new_spk}")
            audio_chunk.export(f"{args.outdir}/evaluation/{new_spk}/3{str(row['idx']).zfill(4)}.wav", format="wav")
        # ground truth data, used for objective evaluations
        else:
            if not os.path.exists(f"{args.outdir}/ground_truth/{new_spk}"):
               os.makedirs(f"{args.outdir}/ground_truth/{new_spk}")
            audio_chunk.export(f"{args.outdir}/ground_truth/{new_spk}/3{str(row['idx']).zfill(4)}.wav", format="wav")

if __name__ == "__main__":
    main()
