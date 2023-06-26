#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Setup HARANA library."""

import os
import pip
import sys

from distutils.version import LooseVersion
from setuptools import find_packages
from setuptools import setup

if LooseVersion(sys.version) < LooseVersion("3.7"):
    raise RuntimeError(
        "HARANA requires Python>=3.7, "
        "but your Python is {}".format(sys.version)
    )
if LooseVersion(pip.__version__) < LooseVersion("19"):
    raise RuntimeError(
        "pip>=19.0.0 is required, but your pip is {}. "
        'Try again after "pip install -U pip"'.format(pip.__version__)
    )

requirements = {
    "install": [
        "torch==1.12.0",
        "numpy==1.22.4",
        "setuptools>=38.5.1",
        "librosa==0.8.1",
        "soundfile>=0.10.2",
        "tensorboardX==2.6",
        "matplotlib>=3.1.0",
        "PyYAML>=3.12",
        "tqdm>=4.26.1",
        "kaldiio>=2.14.1",
        "h5py>=2.9.0",
        "yq>=2.10.0",
        "filelock",
        "protobuf<=3.20.1",
        "pysptk",
        "pyworld==0.3.0",
        "humanfriendly",
        "torch_complex",
        "speechbrain",
        "espnet==202207",
        "pydub",
        #"torchaudio>=1.8.0",
    ],
    "setup": [
        "pytest-runner",
    ],
    "test": [
        "pytest>=3.3.0",
        "hacking>=4.1.0",
        "flake8-docstrings>=1.3.1",
        "black",
    ],
}
entry_points = {
    "console_scripts": [
        "harana-extract-speakers=harana.bin.extract_spk_embs:main",
        "harana-preprocess=harana.bin.preprocess_fastsvc:main",
        "harana-preprocess-b02=harana.bin.preprocess_b02:main",
        "harana-compute-statistics=harana.bin.compute_statistics_fastsvc:main",
        "harana-compute-statistics-b02=harana.bin.compute_statistics_b02:main",
        "harana-compute-f0stats=harana.bin.compute_f0stats:main",
        "harana-normalize=harana.bin.normalize_fastsvc:main",
        "harana-normalize-b02=harana.bin.normalize_b02:main",
        "harana-train-fastsvc=harana.bin.train_fastsvc:main",
        "harana-train-b02=harana.bin.train_b02:main",
        "harana-decode-fastsvc=harana.bin.decode_fastsvc:main",
        "harana-decode-b02=harana.bin.decode_b02:main",
        "harana-synthesize-b02=harana.bin.synthesize_b02:main",
    ]
}

install_requires = requirements["install"]
setup_requires = requirements["setup"]
tests_require = requirements["test"]
extras_require = {
    k: v for k, v in requirements.items() if k not in ["install", "setup"]
}

dirname = os.path.dirname(__file__)
setup(
    name="harana",
    url="http://github.com/lesterphillip/HARANA",
    author="Lester Violeta",
    author_email="violeta.lesterphillip@g.sp.m.is.nagoya-u.ac.jp",
    description="SVCC23 Baseline using FastSVC",
    long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT License",
    packages=find_packages(include=["harana*"]),
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    entry_points=entry_points,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
