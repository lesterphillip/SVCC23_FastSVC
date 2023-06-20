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
    parser.add_argument("--old_path", type=str, help="Keyword to be changed")
    parser.add_argument("--new_path", type=str, help="Keyword to change to")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    trim_threshold_in_db = 30 # Need to tune carefully if the recording is not good.
    trim_frame_size = 2048    # Frame size in trimming.
    trim_hop_size = 512       # Hop size in trimming

    all_files = glob.glob(args.dir, recursive=True)
    print(f"num of files: {len(all_files)}")
    for wav in tqdm(all_files, total=len(all_files)):
        realfs = librosa.get_samplerate(wav)
        y, sr = librosa.load(wav, sr=realfs)
        audio, _ = librosa.effects.trim(
                y,
                top_db=trim_threshold_in_db,
                frame_length=trim_frame_size,
                hop_length=trim_hop_size,
            )
        if not os.path.exists(wav.rsplit("/", 1)[0]):
            os.system(f"mkdir -p {wav.rsplit('/', 1)[0]}")

        sf.write(wav, audio, realfs)
