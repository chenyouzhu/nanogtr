# NanoGTR

m6A prediction for nanopore direct RNA sequencing (DRS) with models trained on
ground-truth RNAs (GTRs).

GTRs are synthetic 325-nt RNAs that carry either A or m6A at defined positions and
together cover all 256 NN(A/m6A)NN sequence contexts. Because the modification status
of every read is known, they can be used to train m6A classifiers for each 5-mer, both
at the site level (features averaged over reads) and for single reads. This repository
contains the code for feature extraction, model training and prediction on RNA002 and
RNA004 flow cells.

## Layout

| Path | Content |
|---|---|
| `preprocess/` | per-motif base-calling features from aligned reads |
| `train/` | model training: bulk models, single-read models for RNA002 and RNA004 |
| `predict/` | single-read m6A calling and per-site m6A fractions (RNA004) |
| `examples/` | the full command sequences for RNA004 training and prediction |
| `nanogtr/` | shared helpers |

## Installation

```bash
git clone https://github.com/chenyouzhu/nanogtr.git
cd nanogtr
pip install -e .
```

Python packages: numpy, pandas, scikit-learn, xgboost (we used xgboost 2.1.3 and
scikit-learn 1.8.0).

External tools: [Dorado](https://github.com/nanoporetech/dorado) (Guppy for RNA002),
[NanoSPA](https://github.com/sihaohuanguc/NanoSPA), minimap2, samtools,
[sam2tsv](https://github.com/lindenb/jvarkit) and, for RNA004,
[blue-crab](https://github.com/Psy-Fer/blue-crab), slow5tools and
[f5c](https://github.com/hasindu2008/f5c).

The feature scripts read the nine-column sam2tsv table (`READ_NAME`, `FLAG`, `CHROM`,
`READ_POS`, `BASE`, `QUAL`, `REF_POS`, `REF`, `OP`); sam2tsv builds that write additional
columns are not supported.

## Trained RNA004 models

The model files are attached to the
[releases](https://github.com/chenyouzhu/nanogtr/releases) of this repository. Every
archive unpacks to a folder of `<motif>_single_read.json` files that is passed to the
prediction script with `-m`.

| Archive | Motifs | Training reads |
|---|---|---|
| `nanogtr_rna004_drach18.tar.gz` | 18 DRACH | GTR |
| `nanogtr_rna004_gtr_only.tar.gz` | 256 NNANN | GTR |
| `nanogtr_rna004_gtr_ivt_8pct.tar.gz` | 256 NNANN | GTR + IVT mRNA (8% of the IVT library) |
| `nanogtr_rna004_gtr_ivt_60k.tar.gz` | 256 NNANN | GTR + IVT mRNA (up to 60,000 reads per motif) |

The DRACH set was used for the transcriptome-wide analyses and takes 58 features per
read: 11 base-calling features for each of the five positions of the motif, plus the mean,
standard deviation and dwell time of the f5c event. The 256-motif sets were used for the
benchmarks and take four more signal features (log dwell time, the difference of the event
mean to the k-mer model mean, and this difference in units of the model standard
deviation, signed and absolute). The prediction script reads the feature list from the
model file, so both kinds are used in the same way.

The analyses in the paper use GTR-only models. For transcriptome-wide site discovery we
recommend the models trained with IVT reads, which call far fewer false positives on
unmodified RNA.

## m6A calling on RNA004 data

`examples/rna004_predict.sh` runs all steps for one sequencing run:

```bash
bash examples/rna004_predict.sh pod5/ GRCh38.fa rna004_drach18/ sample_out/
```

1. Basecalling with Dorado (`fast@v5.1.0`, `--min-qscore 7`).
2. Splice-aware alignment with NanoSPA, which separates plus- and minus-strand reads.
3. `blue-crab` converts POD5 to BLOW5 and `f5c eventalign` assigns the signal to 5-mers
   (`--rna --signal-index --scale-events --print-read-name`). The event table is split
   into one file per motif, `eventalign_with_<motif>.txt`.
4. `sam2tsv` lists the base-calling result of every aligned position and
   `preprocess/split_genome_features.py` collects the five positions of each motif
   occurrence into `<motif>_single_read.txt`.
5. Prediction:

```bash
python predict/predict_single_read_rna004.py \
    -f features/ -e events/ -m rna004_drach18/ -o m6A_plus \
    --strand plus --chrom-sizes GRCh38.fa.fai
```

Output:

- `reads.tsv`: `chr`, `pos`, `strand`, `motif`, `read_name`, `prob`, `m6A` for every read
  covering a motif occurrence. `pos` is the 1-based position of the central A.
- `sites.tsv`: `depth`, `modified` and `fraction` for sites covered by at least
  `--min-depth` reads (default 20).

A read is called modified when its probability exceeds `--threshold` (default 0.5). NanoSPA
aligns minus-strand reads to the reverse complement of the reference; with `--chrom-sizes`
their positions are reported in forward-strand coordinates.

Use `--motifs all` with the 256-motif models; the event table and the base-calling features
then have to be split for all NNANN 5-mers (`all` instead of the DRACH list in steps 3
and 4).

## Training RNA004 single-read models

`examples/rna004_train.sh` lists the commands. The inputs are the same two kinds of
per-motif tables as for prediction, made from a GTR run aligned to the 32 GTR sequences
(`A_1`..`A_16`, `m6A_1`..`m6A_16`). Only the 16 designed positions of each GTR are used.
Reads on `m6A_*` are labelled 1 and reads on `A_*` are labelled 0.

```bash
python train/train_single_read_rna004.py \
    -s gtr/features -e gtr/events -o models_gtr_only --motifs all -w 10
```

For each motif the reads are split 80/20 by read, an XGBoost classifier is tuned on the
training reads (randomised search, 40 candidates, 5-fold cross-validation grouped by read)
and evaluated on the held-out reads. The output folder contains the model
(`<motif>_single_read.json`), its feature list and metrics (`<motif>_single_read.meta.json`)
and `training_summary.csv`.

### Adding IVT reads to the training set

Reads from in vitro transcribed mRNA carry no modification and can be added to the
negatives. They are processed like a sample (steps 1-4 above; we re-aligned the
plus-strand reads to the transcriptome) and passed with two extra options:

```bash
python train/train_single_read_rna004.py \
    -s gtr/features -e gtr/events -o models_gtr_ivt_8pct --motifs all -w 10 \
    --neg-feature-folder ivt/features --neg-event-folder ivt/events --neg-frac 0.8
```

- `--neg-frac` uses this fraction of the IVT reads of each motif (sampled by read).
- `--neg-max-reads` caps the number of IVT reads per motif.

IVT reads take part in the 80/20 split like GTR reads.

The two released configurations are:

| Model set | IVT input | Options |
|---|---|---|
| GTR + IVT 8% | features from a 10% read subsample of the IVT library | `--neg-frac 0.8` |
| GTR + IVT 60k | same | `--neg-max-reads 60000 --scale-pos-weight balanced` |

## RNA002 and bulk models

These models use base-calling features only. Reads are basecalled with Guppy and aligned
with NanoSPA.

Features of the GTR run (one BAM), or of both strands of a sample:

```bash
python preprocess/extract_feature.py \
    -b alignment/plus_strand/collect.sorted.bam -r alignment/plus_strand/reference.fa \
    -p picard.jar -o features --motifs drach

python preprocess/extract_single_read_transcriptome.py -p picard.jar --motifs drach
```

Both run samtools, sam2tsv and the motif split in one go;
`preprocess/split_genome_features.py` does the last step on an existing sam2tsv table.

Single-read models (one classifier per motif, equal numbers of A and m6A reads, 80/20
split, grid search with 5-fold cross-validation):

```bash
python train/train_single_read_rna002.py --motif drach \
    --split_folder features --reference_file alignment/plus_strand/reference.fa \
    --output single_read_model_rna002
```

Bulk models are trained on features averaged over randomly drawn sets of reads: 20,000 sets
of 20 A reads as negatives; 10,000 sets of 20 m6A reads and 10,000 sets of 180 m6A plus 20
A reads (90% m6A) as positives:

```bash
python train/train_average_model.py -r alignment/plus_strand/reference.fa -d features
```

## Data

Raw sequencing data are available from the European Nucleotide Archive under
[PRJEB98639](https://www.ebi.ac.uk/ena/browser/view/PRJEB98639). The IVT mRNA library is
from [PRJEB80229](https://www.ebi.ac.uk/ena/browser/view/PRJEB80229).

## License

MIT
