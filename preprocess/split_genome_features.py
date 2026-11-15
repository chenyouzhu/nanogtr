"""Split a sam2tsv single-read table into one file of 5-mer windows per motif."""
import argparse

from nanogtr.motifs import parse_motifs
from nanogtr.sam2tsv import split_single_read

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('in_file', help='sam2tsv table without N rows (single_read_without_skip.txt)')
    p.add_argument('motif_option', help="'drach', 'all_A_center' (256 NNANN) or a comma-separated list")
    p.add_argument('-o', '--output', default='.', help='output directory')
    p.add_argument('-b', '--batch-size', type=int, default=5000, help='reads held in memory per write')
    args = p.parse_args()
    split_single_read(args.in_file, parse_motifs(args.motif_option), args.output, args.batch_size)
