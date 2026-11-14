#!/bin/bash
# Single-read m6A calling for one RNA004 direct RNA run.
# usage: rna004_predict.sh <pod5_dir> <genome.fa> <model_dir> <out_dir>
set -euo pipefail

POD5=$(realpath "$1")
GENOME=$(realpath "$2")
MODELS=$(realpath "$3")
OUT=$4
THREADS=${THREADS:-16}
KMER_MODEL=${KMER_MODEL:-rna004.nucleotide.5mer.model}   # from the f5c repository
REPO=$(cd "$(dirname "$0")/.." && pwd)
MOTIFS=$(python -c "from nanogtr.motifs import DRACH; print(','.join(DRACH))")

mkdir -p "$OUT/fastq"
cd "$OUT"

dorado basecaller fast@v5.1.0 "$POD5" --min-qscore 7 > calls.bam
samtools fastq calls.bam > fastq/calls.fastq

# splice-aware alignment; plus- and minus-strand reads go to separate folders
python -c "from nanospa.alignment import align; align('fastq/', '$GENOME', '')"

blue-crab p2s "$POD5" -o reads.blow5
slow5tools index reads.blow5

for strand in plus minus; do
    S=alignment/${strand}_strand
    mkdir -p "$S/events" "$S/features"

    f5c index --slow5 reads.blow5 "$S/collect.fastq"
    f5c eventalign -t "$THREADS" --rna --kmer-model "$KMER_MODEL" --slow5 reads.blow5 \
        -b "$S/collect.sorted.bam" -r "$S/collect.fastq" -g "$S/reference.fa" \
        --signal-index --scale-events --print-read-name > "$S/eventalign.tsv"
    awk -F'\t' -v dir="$S/events" -v list="$MOTIFS" '
        BEGIN { n = split(list, m, ","); for (i = 1; i <= n; i++) keep[m[i]] = 1 }
        $3 in keep { print > (dir "/eventalign_with_" $3 ".txt") }' "$S/eventalign.tsv"
    rm "$S/eventalign.tsv"

    # sam2tsv needs a sequence dictionary and fails on records without SEQ or CIGAR
    samtools dict "$S/reference.fa" -o "$S/reference.dict"
    samtools view -h "$S/collect.sorted.bam" \
        | awk '/^@/ || ($10 != "*" && $6 != "*")' \
        | sam2tsv -R "$S/reference.fa" \
        | awk '$9 != "N"' > "$S/single_read.txt"
    python "$REPO/preprocess/split_genome_features.py" "$S/single_read.txt" "$MOTIFS" -o "$S/features"
    rm "$S/single_read.txt"

    python "$REPO/predict/predict_single_read_rna004.py" \
        -f "$S/features" -e "$S/events" -m "$MODELS" -o "m6A_$strand" \
        --strand "$strand" --chrom-sizes "$GENOME.fai"
done
