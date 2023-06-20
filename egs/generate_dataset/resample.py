# Copyright 2023 Nagoya University (Lester Violeta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import glob
import os
import librosa
import argparse
import soundfile as sf
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="file name",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dir", type=str, help="Source file directory")
    parser.add_argument("--src_fs", type=int, help="Source sampling freq")
    parser.add_argument("--trg_fs", type=int, help="Target sampling freq")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    input_fs = args.src_fs
    target_fs = args.trg_fs
    all_files = glob.glob(args.dir, recursive=True)
    print(f"num of files: {len(all_files)}")
    for wav in tqdm(all_files, total=len(all_files)):
        y, sr = librosa.load(wav, sr=input_fs)
        if not os.path.exists(wav.rsplit("/", 1)[0]):
            os.system(f"mkdir -p {wav.rsplit('/', 1)[0]}")

        y_res = librosa.resample(y, orig_sr=sr, target_sr=target_fs)
        sf.write(wav, y_res, target_fs)

