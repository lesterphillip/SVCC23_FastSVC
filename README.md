# Singing Voice Conversion Challenge 2023 Starter Kit: FastSVC and B02 System

Official homepage: [http://www.vc-challenge.org/](http://www.vc-challenge.org/)

## Introduction

<p align="justify"> 
This repository provides the <strong>UNOFFICIAL</strong> reimplementation of FastSVC for the Singing Voice Conversion Challenge 2023 (SVCC23) starter kit. SVCC23 includes two tasks: in-domain and cross-domaing singing voice conversion (SVC). In-domain SVC is trained while having access to the singing data of the target speaker, while cross-domain SVC is trained while only having access to the speech data of the target speaker.
</p>

---
## News
- **2023/06/26** We are releasing the B02 system reimplementation.
- **2023/02/17** We are releasing the first version of the repository and some generated samples at [this site](http://www.vc-challenge.org/samples/index.html).

---
## Description of the FastSVC System
- FastSVC: Fast Cross-Domain Singing Voice Conversion with Feature-wise Linear Modulation
- Songxiang Liu, Yuewen Cao, Na Hu, Dan Su, Helen Meng
- [ArXiv paper](https://arxiv.org/abs/2011.05731)

<p align="justify"> 
This system uses phonetic posteriorgrams (PPGs) extracted by a pretrained ASR model, loudness, pitch, and speaker embeddings (x-vectors). The PPGs are upsampled by a scaling factor, and are fused with the downsampled loudness and pitch features after being processed by a FiLM block. The speaker embeddings are added to the fused PPG, loudness, and pitch afterwards.
</p>

Please note that there are some differences between this system and the official paper. The original FastSVC system was made for 16kHz generation, but we made changes to accommodate 24kHz generation.

Specific changes:
1. Instead of using three different pitch extractors, we only used the Harvest from the PyWorld toolkit as the pitch extractor.
2. The speaker embeddings are fixed (not learnable) and are only extracted beforehand by averaging the extractions from each utterance. 
3. We replace the discriminator with the one used in HiFiGAN as recommended by the authors.
4. The PPG extractor we [used](https://github.com/liusongxiang/ppg-vc/tree/main/conformer_ppg_model) has a hop size of 160, so the upsampling scales are changed to [5, 4, 4, 2].

### Reimplementation Samples
You can see some samples of the original paper's reimplementation [here.](https://drive.google.com/drive/folders/1VDlyQDsvZZ2UujfY3axnUoKeCBM-h-Kx?usp=share_link)

## Description of the FastSVC System
To help people get started with SVC, we also developed a decomposed version of FastSVC to improve training time.
Please refer to recipe in `egs/svcc23/baseline02`

---
## Pretrained models and demo for SVCC23 dataset

Please refer to the README files in `egs/svcc23/fastsvc1/README.md`

---
## Corpus

Please note that we will only give access to people who have signed the dataset's license agreement. To gain access to the SVCC23 dataset, please sign the license agreement in the registration form and submit it there. 

Other external datasets can also be used; however, it has to be publicly available in order to encourage reproducible research. You can find more details about the specific rules of the challenge [here.](http://www.vc-challenge.org/rules.html)

---
## Installation via virtualenv

```bash
$ git clone https://github.com/lesterphillip/SVCC23_FastSVC.git
$ cd SVCC23_FastSVC
$ python3 -m virtualenv venv
$ . ./venv/bin/activate
$ pip install -e .
$ ...
```

---
## Usage

Please check `egs/svcc23/fastsvc1/README.md` for instructions on how to run the repository.

---
## Citation

If you find the code helpful, please cite the following.

```
@inproceedings{liu2021fastsvc,
  title={{FastSVC: Fast Cross-Domain Singing Voice Conversion with Feature-wise Linear Modulation}},
  author={Liu, Songxiang and Cao, Yuewen and Hu, Na and Su, Dan and Meng, Helen},
  booktitle={IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```

---
## Authors

#### Code Development:
Lester Phillip Violeta @ Nagoya University ([@lesterphillip](https://github.com/lesterphillip))

#### FastSVC author:
Songxiang Liu @ Tencent AI Lab ([@liusongxiang](https://github.com/liusongxiang))

#### SVCC 2023 co-organizers:
Wen-Chin Huang @ Nagoya University ([@unilight](https://github.com/unilight))

Lester Phillip Violeta @ Nagoya University ([@lesterphillip](https://github.com/lesterphillip))

Jiatong Shi @ Carnegie Mellon University ([@ftshijt](https://github.com/ftshijt))

Songxiang Liu @ Tencent AI Lab ([@liusongxiang](https://github.com/liusongxiang))

#### Advisor:
Tomoki Toda @ Nagoya University

---

## Acknowledgements
We would like to thank Ryuichi Yamamoto ([@r9y9](https://github.com/r9y9)) for his valuable insights in developing this repository.

The skeleton of this repository was also greatly based on @kan-bayashi's awesome [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) repository. Please check it out if you need to train any sort of vocoder. 

We also used @chomeyama's [HN-uSFGAN](https://github.com/chomeyama/HN-UnifiedSourceFilterGAN) repository as a reference.

---
## Questions

Please submit an issue if you encounter any bugs or have any questions about the repository. You may also contact the organizing team through this e-mail if you have any questions about SVCC itself.

- `svcc2023__at__vc-challenge.org`  (replace the `__at__` with `@`)
