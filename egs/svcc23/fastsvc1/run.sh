#!/bin/bash
# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=0        # stage to start
stop_stage=100 # stage to stop
n_gpus=1       # number of gpus in training
n_jobs=4       # number of parallel jobs in feature extraction

conf=conf/fastsvc.yaml
f0_path=conf/f0.yml
device_feat_extract="cuda"

# directory path settings
dumpdir=dump # directory to dump features

# training related settings
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)

# decoding related settings
checkpoint="" # checkpoint path to be used for decoding
              # if not provided, the latest one will be used
              # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

train_set="train" # name of training data directory
dev_set="dev"           # name of development data direcotry
eval_set="eval"     # name of evaluation data direcotry

set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Speaker embedding extraction"
    pids=()
    for name in "${train_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/embs" ] && mkdir -p "${dumpdir}/${name}/embs"
        echo "Speaker embedding extraction start. See the progress via ${dumpdir}/${name}/embs/speaker_extraction.*.log."
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
    [ ! -e "${dumpdir}/${dev_set}" ] && mkdir -p "${dumpdir}/${dev_set}"
    cp -r ${dumpdir}/${train_set}/embs ${dumpdir}/${dev_set}
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: PPG, F0, and loudness extraction"
    pids=()
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            harana-preprocess \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --device "${device_feat_extract}" \
                --spk_emb_path "${dumpdir}/${name}/embs/spk_embs.h5" \
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
    echo "Stage 2: Extract F0 mean and deviation"
    for name in "${train_set}" "${eval_set}"; do
    echo "F0 Statistics computation start. See the progress via ${dumpdir}/${name}/compute_f0stats.log."
        ${train_cmd} "${dumpdir}/${name}/compute_f0stats.log" \
            harana-compute-f0stats \
                --config "${conf}" \
                --rootdir "${dumpdir}/${name}/raw" \
                --dumpdir "${dumpdir}/${name}"
    done
    echo "Successfully finished calculation of f0 statistics."
    cp -r ${dumpdir}/${train_set}/f0_stats ${dumpdir}/${dev_set}
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Feature normalization."
    echo "Statistics computation start. See the progress via ${dumpdir}/${train_set}/compute_statistics.log."
    ${train_cmd} "${dumpdir}/${train_set}/compute_statistics.log" \
        harana-compute-statistics \
            --config "${conf}" \
            --rootdir "${dumpdir}/${train_set}/raw" \
            --dumpdir "${dumpdir}/${train_set}"
    echo "Successfully finished calculation of statistics."

    # normalize features and dump them
    pids=()
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/norm" ] && mkdir -p "${dumpdir}/${name}/norm"
        echo "Nomalization start. See the progress via ${dumpdir}/${name}/norm/normalize.*.log."
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/norm/normalize.JOB.log" \
            harana-normalize \
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
fi

if [ -z "${tag}" ]; then
    expdir="exp/${train_set}_$(basename "${conf}" .yaml)"
else
    expdir="exp/${train_set}_${tag}"
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    cp "${dumpdir}/${train_set}/stats.joblib" "${expdir}"
    train="harana-train-fastsvc"
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --train-dumpdir "${dumpdir}/${train_set}/norm" \
            --dev-dumpdir "${dumpdir}/${dev_set}/norm" \
            --outdir "${expdir}" \
            --resume "${resume}"
    echo "Successfully finished training."
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    echo "Stage 5: Waveform generation"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    pids=()
    echo "Decoding using ${outdir}."
    for name in "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            harana-decode-fastsvc \
                --config "${conf}" \
                --dumpdir "${dumpdir}/${name}/norm" \
                --srcf0stats "${dumpdir}/${name}/f0_stats" \
                --trgf0stats "${dumpdir}/${train_set}/f0_stats" \
                --stats "${dumpdir}/${train_set}/stats.joblib" \
                --checkpoint "${checkpoint}" \
                --spk_emb_path "${dumpdir}/${train_set}/embs/spk_embs.h5" \
                --outdir "${outdir}/${name}"
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished decoding."
fi
echo "Finished."
