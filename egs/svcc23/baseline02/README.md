# SVCC23 B02 Reimplementation

This recipe runs a reimplementation of the SVCC23 B02 system.

Before running this, please make sure that you have already setup your environment. You can refer to the README file in the root folder.

---

## HOW TO RUN THIS REPO

The code follows a kaldi-style recipe.

### 1. Download the dataset and place it in any directory.

```sh
cd <any-where>
git clone https://github.com/lesterphillip/SVCC23_FastSVC.git
cd SVCC23_FastSVC/egs/svcc23/fastsvc1
```

### 2. Specify the following parameters for dataset preprocessing.

2a. Create the wav.scp files in the `data/` directory.

We use a kaldi-style data directory. To make things easier, you can just simply replace the `/path/to/dir` with the directory where you placed the dataset and the repository. 

---

2b. Set the F0 search range in `conf/f0.yaml`

You can simply use default values in the F0 search range. However, if you want to make the F0 extraction more precise, you need to change the values in the `f0.yaml` file. You can refer to this [document](https://github.com/k2kobayashi/sprocket/blob/master/docs/vc_example.md#3-modify-f0-search-range) for a guide on how to determine the F0 search range.
We provide our used calculations in `conf/f0.yaml`

---

2c. Split the wav.scp file into train and test.

To split a wav.scp file, you can simply use the following script. This script ensures that all of the speakers are added into the test set. 

Please note that you need to set the utterance ids as `{speaker_id}_{utterance_details}`.

```sh
$ . ./path.sh
# source_scp: path of input wav.scp file
# train_scp_out: path of output train set wav.scp file
# dev_scp_out: path of output dev set wav.scp file
# dev_count: number of utterances from EACH SPEAKER to be added 
$ python3 utils/split_train_dev.py --source_scp data/svcc23_full/wav.scp --train_scp_out data/train/wav.scp --dev_scp_out data/dev/wav.scp --dev_count 0
```


### 3. Run the feature extraction, statistics calculation, and normalization scripts.

```sh
$ . ./path.sh
# Stage 0: x-vector extraction using SpeechBrain
# Stage 1: PPG, F0, and loudness feature extraction (PPG extraction uses a GPU by default, but you can switch it off through `device_feat_extract`)
# Stage 2: Compute statistics and normalize features
$ ./run.sh --stop_stage 2
```

### 3. Train the model

You can simply start training the model just by the following command:

```sh
# You can sort your experiments by the --tag option.
$ ./run.sh --stage 3 --stop_stage 3 --tag svccb02_experiment1
```

There is also a lot of flexibility in the code, as you can initialize weights from a pretrained model or resume from a checkpoint.

3b (Optional). You can also simply download a pretrained model and proceed to the next step.

Download the model from the link below, then place them in the `exp/` folder.
You can rename them as `exp/{train_set}_{tag}`

### 4. Synthesize waveforms

Next, you can start synthesizing waveforms using the model just by the following command:

```sh
# Make sure that you specify the --tag option to name the experiment directory
$ ./run.sh --stage 4 --tag svccb02_experiment1
```

If you are using a pretrained model, you can also specify the checkpoint to load using the `--checkpoint` option

## Pretrained Models

Acoustic model Taco2 + AR / Trained on SVCC23 + VCTK + opencpop + multisinger + opensinger + m4singer / 50k steps
- [Link](https://drive.google.com/file/d/15jXHAA0SKF4nqD6kRDYjVXezBnhcJbH-/view?usp=drive_link)

Vocoder model HN-uSFGAN / Trained on SVCC23 + VCTK + opencpop + multisinger + opensinger + m4singer / 600k steps
- [Link](https://drive.google.com/file/d/1z61kNRHNz9eELi97re1nFNMvY3ym8zH9/view?usp=drive_link)

