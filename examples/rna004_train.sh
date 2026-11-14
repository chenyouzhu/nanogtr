#!/bin/bash
# Training inputs and models for RNA004, from a GTR run and an IVT mRNA run that were
# basecalled, aligned and event-aligned as in rna004_predict.sh.
#   GTR: reads aligned to the 32 GTR sequences (A_1..A_16, m6A_1..m6A_16)
#   IVT: plus-strand reads re-aligned to the transcriptome (minimap2 -ax splice -uf -k14)
set -euo pipefail

GTR=gtr      # holds collect.sorted.bam, ref_GTR.fa, eventalign.tsv
IVT=ivt      # holds output.sorted.bam, transcripts.fa, eventalign.tsv
REPO=$(cd "$(dirname "$0")/.." && pwd)

# GTR events: keep the 16 designed NNANN sites of each construct.
# The designed A sits at 0-based 27 + 15k (k = 0..15), i.e. the 5-mer starts at 25 + 15k.
mkdir -p "$GTR/events" "$GTR/features"
awk -F'\t' -v dir="$GTR/events" '
    $1 ~ /^(A|m6A)_/ && $3 ~ /^[ACGT][ACGT]A[ACGT][ACGT]$/ &&
    $2 >= 25 && $2 <= 250 && ($2 - 25) % 15 == 0 {
        print > (dir "/eventalign_with_" $3 ".txt")
    }' "$GTR/eventalign.tsv"

samtools view -b -F 2308 "$GTR/collect.sorted.bam" > "$GTR/primary.bam"
sam2tsv -R "$GTR/ref_GTR.fa" "$GTR/primary.bam" > "$GTR/single_read.txt"
python "$REPO/preprocess/split_genome_features.py" "$GTR/single_read.txt" all -o "$GTR/features"

# IVT events at every NNANN 5-mer; base features from a 10% read subsample
mkdir -p "$IVT/events" "$IVT/features"
awk -F'\t' -v dir="$IVT/events" '
    $3 ~ /^[ACGT][ACGT]A[ACGT][ACGT]$/ { print > (dir "/eventalign_with_" $3 ".txt") }' "$IVT/eventalign.tsv"

samtools view -b -F 2308 -s 0.1 "$IVT/output.sorted.bam" > "$IVT/sample.bam"
sam2tsv -R "$IVT/transcripts.fa" "$IVT/sample.bam" > "$IVT/single_read.txt"
python "$REPO/preprocess/split_genome_features.py" "$IVT/single_read.txt" all -o "$IVT/features"

# GTR only
python "$REPO/train/train_single_read_rna004.py" \
    -s "$GTR/features" -e "$GTR/events" -o models_gtr_only --motifs all -w 10

# GTR + IVT, at most 60,000 IVT reads per motif
python "$REPO/train/train_single_read_rna004.py" \
    -s "$GTR/features" -e "$GTR/events" -o models_gtr_ivt_60k --motifs all -w 10 \
    --neg-feature-folder "$IVT/features" --neg-event-folder "$IVT/events" \
    --neg-max-reads 60000 --scale-pos-weight balanced

# GTR + IVT, 8% of the IVT library (80% of the 10% subsample)
python "$REPO/train/train_single_read_rna004.py" \
    -s "$GTR/features" -e "$GTR/events" -o models_gtr_ivt_8pct --motifs all -w 10 \
    --neg-feature-folder "$IVT/features" --neg-event-folder "$IVT/events" \
    --neg-frac 0.8
