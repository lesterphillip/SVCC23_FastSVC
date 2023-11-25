import os

def make_wav_scp(directory, output_file, spk_ids=None):
    with open(output_file, "w") as f:
        for spk_id in os.listdir(directory):
            # just check if the directory exists
            spk_dir = os.path.join(directory, spk_id)
            if spk_ids is not None:
                assert os.path.isdir(spk_dir), f"speaker id in: {spk_dir} does not exist"

            for filename in os.listdir(spk_dir):
                if filename.split(".")[-1] == "wav":
                    utt_id = os.path.splitext(filename)[0]
                    file_path_dir = os.path.join(spk_dir, filename)
                    f.write(f"{spk_id}_{utt_id} {file_path_dir}\n")

make_wav_scp("/path/to/SVCC2023Dataset/Data", "data/traindev/wav.scp") # replace with the absolute path of your dataset