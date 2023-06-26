#!/bin/bash

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=0        # stage to start
stop_stage=100 # stage to stop
n_gpus=1       # number of gpus in training
n_jobs=2       # number of parallel jobs in feature extraction

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
conf=conf/taco2_hubertsoft.yaml
f0_path=conf/f0.yml
pretrain=""
vocoder_file=downloads/vocoder_hnusfgan/checkpoint-600000steps.pkl

dumpdir=dump # directory to dump features

# training related settings
tag=""     # tag for directory to save model
resume=""
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)

# decoding related settings
decode_steps=
checkpoint="" # checkpoint path to be used for decoding
              # if not provided, the latest one will be used
              # (e.g. <path>/<to>/checkpoint-400000steps.pkl)


# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

train_set="train_p225" # name of training data directory
dev_set="dev_p225"           # name of development data direcotry
eval_set="eval_full"     # name of evaluation data direcotry

set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Speaker embedding extraction"
    # extract raw features
    pids=()
    for name in "${train_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/embs" ] && mkdir -p "${dumpdir}/${name}/embs"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/embs/speaker_extraction.*.log."
        utils/make_subset_data.sh "data/${name}" "1" "${dumpdir}/${name}/embs"
        ${train_cmd} JOB=1:1 "${dumpdir}/${name}/embs/speaker_extraction.JOB.log" \
            harana-extract-speakers \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/embs/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/embs"
        echo "Successfully finished speaker embedding extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished speaker embedding extraction all sets."
    mkdir "${dumpdir}/${dev_set}"
    cp -r "${dumpdir}/${train_set}/embs" "${dumpdir}/${dev_set}"
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    pids=()
    echo "Stage 1: Feature extraction"
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        if [[ ${name} = ${eval_set} ]]; then device="cpu"; else device="cuda"; fi
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            harana-preprocess-b02 \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --extractordir "../../../harana/extractor/exp" \
                --device "${device}" \
                --f0_path "${f0_path}"
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction all sets."
fi


if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Statistics computation."
    echo "Statistics computation start. See the progress via ${dumpdir}/${train_set}/compute_statistics.log."
    ${train_cmd} "${dumpdir}/${train_set}/compute_statistics.log" \
        harana-compute-statistics-b02 \
            --config "${conf}" \
            --rootdir "${dumpdir}/${train_set}/raw" \
            --dumpdir "${dumpdir}/${train_set}"
    echo "Successfully finished calculation of statistics."
    cp -r "${dumpdir}/${train_set}/stats.joblib" "${dumpdir}/${eval_set}"
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Feature normalization"
    # normalize features and dump them
    pids=()
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/norm" ] && mkdir -p "${dumpdir}/${name}/norm"
        echo "Normalization start. See the progress via ${dumpdir}/${name}/norm/normalize.*.log."
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/norm/normalize.JOB.log" \
            harana-normalize-b02 \
                --config "${conf}" \
                --stats "${dumpdir}/${train_set}/stats.joblib" \
                --rootdir "${dumpdir}/${name}/raw/dump.JOB" \
                --dumpdir "${dumpdir}/${name}/norm/dump.JOB"
        echo "Successfully finished normalization of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished normalization."
    cp -r "${dumpdir}/${train_set}/stats.joblib" "${dumpdir}/${eval_set}"
fi

if [ -z "${tag}" ]; then
    expdir="exp/${train_set}_$(basename "${conf}" .yaml)"
else
    expdir="exp/${train_set}_${tag}"
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Taco2 + AR network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    cp "${dumpdir}/${train_set}/stats.joblib" "${expdir}"
    train="harana-train-b02"
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --stats "${dumpdir}/${train_set}/stats.joblib" \
            --train-dumpdir "${dumpdir}/${train_set}/norm" \
            --dev-dumpdir "${dumpdir}/${dev_set}/norm" \
            --outdir "${expdir}" \
            --resume "${resume}"
    echo "Successfully finished training."
fi
[ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
outdir="${expdir}/decoding/$(basename "${checkpoint}" .pkl)/hdf5"

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    echo "Stage 5: Taco2 network decoding for HN-USFGAN inputs"
    # shellcheck disable=SC2012
    pids=()
    echo "Decoding using ${outdir}."
    for name in  "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            harana-decode-b02 \
                --dumpdir "${dumpdir}/${name}/norm" \
                --stats "${dumpdir}/${eval_set}/stats.joblib" \
                --checkpoint "${checkpoint}" \
                --outdir "${outdir}/${name}" \
                --spk_emb_path "${dumpdir}/${train_set}/embs/spk_embs.h5" \
                --f0stats "${f0_path}"
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished decoding."
fi

if [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 7 ]; then
    echo "Stage 6: HN-USFGAN waveform generation"
    # shellcheck disable=SC2012
    outdir2="${expdir}/decoding/$(basename "${checkpoint}" .pkl)/wav"
    pids=()
    echo "Decoding using ${outdir2}."
    for name in  "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${outdir2}/${name}" ] && mkdir -p "${outdir2}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir2}/${name}/decode.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir2}/${name}/decode.log" \
            harana-synthesize-b02 \
                --dumpdir "${outdir}/${name}" \
                --srcf0stats "${dumpdir}/${name}/f0_stats" \
                --trgf0stats "${dumpdir}/${train_set}/f0_stats" \
                --checkpoint "${vocoder_file}" \
                --outdir "${outdir2}/${name}"
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished decoding."
fi

echo "Finished."
