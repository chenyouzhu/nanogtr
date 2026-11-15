#!/usr/bin/env python3
"""Call m6A on single reads with per-motif RNA004 models and summarise the calls per site.

Run once per strand folder of a NanoSPA alignment. Writes reads.tsv (one row per read and
site) and sites.tsv (modified fraction of sites with enough reads) to the output folder.
"""
import argparse
import os

import pandas as pd
import xgboost as xgb

from nanogtr.features import build_table, strip_strand
from nanogtr.motifs import parse_motifs


def call_reads(motif, args):
    window_file = os.path.join(args.features, f'{motif}_single_read.txt')
    event_file = os.path.join(args.events, f'eventalign_with_{motif}.txt')
    model_file = os.path.join(args.models, f'{motif}_single_read.json')
    if not all(os.path.exists(f) and os.path.getsize(f) for f in (window_file, event_file, model_file)):
        return None

    model = xgb.Booster()
    model.load_model(model_file)
    names = model.feature_names
    # the 256-context models were trained on one event per read and site
    df = build_table(window_file, event_file, longest=any(n.startswith('sig_') for n in names))
    if df.empty:
        return None

    rows = pd.DataFrame({
        'chr': strip_strand(df['chr_2']),
        'pos': df['pos_2'].astype(int),
        'read_name': df['read_name'],
        'prob': model.predict(xgb.DMatrix(df[names].astype(float))),
    })
    rows['m6A'] = rows['prob'] > args.threshold
    # a read with several events on the 5-mer is modified if at least half of them are
    reads = rows.groupby(['chr', 'pos', 'read_name'], sort=False)[['prob', 'm6A']].mean().reset_index()
    reads['m6A'] = (reads['m6A'] >= 0.5).astype(int)
    return reads


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-f', '--features', required=True, help='<motif>_single_read.txt files')
    p.add_argument('-e', '--events', required=True, help='eventalign_with_<motif>.txt files')
    p.add_argument('-m', '--models', required=True, help='<motif>_single_read.json models')
    p.add_argument('-o', '--out', required=True)
    p.add_argument('--motifs', default='drach', help="'drach', 'all' (256 NNANN) or a comma-separated list")
    p.add_argument('--strand', choices=['plus', 'minus'], default='plus')
    p.add_argument('--chrom-sizes', help='.fai of the genome; minus-strand positions are '
                                         'converted to forward coordinates when given')
    p.add_argument('--threshold', type=float, default=0.5, help='probability above which a read is called m6A')
    p.add_argument('--min-depth', type=int, default=20, help='reads required to report a site')
    args = p.parse_args()

    sizes = None
    if args.chrom_sizes and args.strand == 'minus':
        fai = pd.read_csv(args.chrom_sizes, sep='\t', header=None, usecols=[0, 1], dtype={0: str})
        sizes = dict(zip(fai[0], fai[1]))

    os.makedirs(args.out, exist_ok=True)
    reads_file = os.path.join(args.out, 'reads.tsv')
    sites_file = os.path.join(args.out, 'sites.tsv')
    first = True
    for motif in parse_motifs(args.motifs):
        reads = call_reads(motif, args)
        if reads is None:
            print(f'{motif}: no input or model', flush=True)
            continue
        if sizes is not None:
            # minus-strand reads are aligned to the reverse complement of each chromosome
            reads['pos'] = reads['chr'].map(sizes) - reads['pos'] + 1
        reads.insert(2, 'strand', '+' if args.strand == 'plus' else '-')
        reads.insert(3, 'motif', motif)

        sites = (reads.groupby(['chr', 'pos', 'strand', 'motif'])['m6A']
                 .agg(depth='size', modified='sum').reset_index())
        sites = sites[sites['depth'] >= args.min_depth]
        sites['fraction'] = sites['modified'] / sites['depth']

        mode = 'w' if first else 'a'
        reads.to_csv(reads_file, sep='\t', index=False, header=first, mode=mode, float_format='%.4f')
        sites.to_csv(sites_file, sep='\t', index=False, header=first, mode=mode, float_format='%.4f')
        first = False
        print(f'{motif}: {len(reads)} read calls, {len(sites)} sites', flush=True)


if __name__ == '__main__':
    main()
