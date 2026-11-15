"""Extract single-read 5-mer features from a BAM and split them by motif.

Needs samtools, java (picard), jvarkit sam2tsv and awk on PATH.
"""
import argparse
import sys

from nanogtr.motifs import parse_motifs
from nanogtr.sam2tsv import extract_features

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-b', '--bam', default='alignment/plus_strand/collect.sorted.bam', help='input BAM')
    p.add_argument('-r', '--reference', default='alignment/plus_strand/reference.fa', help='reference FASTA')
    p.add_argument('-p', '--picard', default='picard.jar', help='picard.jar')
    p.add_argument('-o', '--output', default='./output', help='output directory')
    p.add_argument('--motifs', default='drach', help="'drach', 'all_A_center' (256 NNANN) or a comma-separated list")
    p.add_argument('-s', '--batch-size', type=int, default=5000, help='reads held in memory per write')
    args = p.parse_args()

    done = extract_features(args.bam, args.reference, args.picard, args.output,
                            parse_motifs(args.motifs), args.batch_size)
    sys.exit(0 if done else 1)
