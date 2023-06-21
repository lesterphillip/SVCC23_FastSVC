# Singing Voice Conversion Challenge 2023 Dataset Generation Script

Official homepage: [http://www.vc-challenge.org/](http://www.vc-challenge.org/)

## Corpus

This script reproduces the SVCC 2023 dataset from the NUS-HLT Speak-Sing (NHSS) dataset. Please note that in order to gain access to the NHSS dataset, you need to sign and submit a license agreement. You can find more details about the process here: https://hltnus.github.io/NHSSDatabase/download.html

---
## Usage

1. Make sure you have installed the required dependencies. Please check the README file at root for more details.
2. In the `generate_svcc.sh` script, change the `rootdir` to the directory path of the NHSS dataset,  and `outdir` to where you want the SVCC 2023 dataset to be generated.
3. Run ./generate_svcc.sh

---
## Details
The script would generate the training set into folders IDM1, IDF1, CDM1, and CDF1. This would contain the wav files and text labels.

The evaluation set would be generated into the `evaluation` folder. To conduct objective evaluations, the parallel utterances are generated into the `ground_truth` folder. The text labels are found at `test_set.csv`.

---
## Citation

If you find the code helpful, please cite the following:
"Bidisha Sharma, Xiaoxue Gao, Karthika Vijayan, Xiaohai Tian, and Haizhou Li. "NHSS: A Speech and Singing Parallel Database." arXiv preprint arXiv:2012.00337 (2020)."

(TODO: SVCC 2023 paper link)

---
## Authors

#### Code Development:
Lester Phillip Violeta @ Nagoya University ([@lesterphillip](https://github.com/lesterphillip))

#### SVCC 2023 co-organizers:
Wen-Chin Huang @ Nagoya University ([@unilight](https://github.com/unilight))

Lester Phillip Violeta @ Nagoya University ([@lesterphillip](https://github.com/lesterphillip))

Jiatong Shi @ Carnegie Mellon University ([@ftshijt](https://github.com/ftshijt))

Songxiang Liu @ Tencent AI Lab ([@liusongxiang](https://github.com/liusongxiang))

#### Advisor:
Tomoki Toda @ Nagoya University

---
## Questions

Please submit an issue if you encounter any bugs or have any questions about the script. You may also contact the organizing team through this e-mail if you have any questions about SVCC itself.

- `svcc2023__at__vc-challenge.org`  (replace the `__at__` with `@`)
