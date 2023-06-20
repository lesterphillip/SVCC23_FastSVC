#!/bin/bash

# Copyright 2023 Nagoya University (Lester Violeta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

rootdir="/path/to/NHSS/Data"
outdir="./SVCC23"
testcsv="./test_set.csv"

target_spks=("M04" "M03" "F01" "F02")

echo "generating training set files..."
# create SVCC23 training set from the NHSS dataset
for spk in "${target_spks[@]}"; do
    echo "creating files for ${spk}"
    idx=1

    # transfer temporarily to avoid rewriting the original dataset
    cp -r "${rootdir}/${spk}" "${outdir}"

    # songs used in test set
    rm -rf "${outdir}/${spk}/S05"
    rm -rf "${outdir}/${spk}/S06"
    rm -rf "${outdir}/${spk}/S09"
    rm -rf "${outdir}/${spk}/S15"
    rm ${outdir}/${spk}/S*/*.wav
    rm ${outdir}/${spk}/S*/*.TextGrid

    # remove song/speech depending on speaker id
    if [ ${spk} == "M04" ] || [ ${spk} == "F01" ]; then
        rm -rf ${outdir}/${spk}/S*/Speech
    elif [ ${spk} == "M03" ] || [ ${spk} == "F02" ]; then
        rm -rf ${outdir}/${spk}/S*/Song
    fi
    
    # rename files
    find "${outdir}/${spk}" -follow -name "*.wav" | sort | while read -r filename;do
        label=${filename/wav/lab}
        printf -v j "%04d" $idx
        [ ! -e "${outdir}/${spk}" ] && mkdir -p "${outdir}/${spk}"
        cp ${filename} "${outdir}/${spk}/1${j}.wav"
        cp ${label} "${outdir}/${spk}/1${j}.lab"
        idx=$((idx+1))
    done

    # clean text files
    for file in ${outdir}/${spk}/*.lab; do
      awk '!/<SIL>/{print $3}' "$file" | tr '\n' ' ' | sed 's/\ $/\n/' > "${file%.*}.txt"
    done

    # remove temporary files
    rm -rf ${outdir}/${spk}/S*
    rm ${outdir}/${spk}/*.lab
done

# rename to SVCC23 IDs
mv ${outdir}/{M04,IDM1}
mv ${outdir}/{M03,CDM1}
mv ${outdir}/{F01,IDF1}
mv ${outdir}/{F02,CDF1}

# generate test set
echo "generating test set files..."
# you can refer to the testcsv file for the text labels
python3 generate_svcctest.py \
    --rootdir ${rootdir} \
    --outdir ${outdir} \
    --testcsv ${testcsv}

# resample to 24kHz
echo "resampling to 24kHz..."
python3 resample.py \
    --dir "${outdir}/*/*.wav" \
    --src_fs 48000 \
    --trg_fs 24000

# remove silences from target_spks
echo "removing silences..."
python3 remove_silences.py \
    --dir "${outdir}/*/*.wav"
