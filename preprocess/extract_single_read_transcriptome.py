"""Extract single-read 5-mer features from the plus- and minus-strand BAMs and split them by motif.

Needs samtools, java (picard), jvarkit sam2tsv and awk on PATH.
"""
import argparse
import sys

from nanogtr.motifs import parse_motifs
from nanogtr.sam2tsv import extract_features

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--bam-forward', default='alignment/plus_strand/collect.sorted.bam', help='plus-strand BAM')
    p.add_argument('--bam-reverse', default='alignment/minus_strand/collect.sorted.bam', help='minus-strand BAM')
    p.add_argument('--output-forward', default='output_forward', help='plus-strand output directory')
    p.add_argument('--output-reverse', default='output_reverse', help='minus-strand output directory')
    p.add_argument('--reference-forward', default='alignment/plus_strand/reference.fa',
                   help='plus-strand reference FASTA')
    p.add_argument('--reference-reverse', default='alignment/minus_strand/reference.fa',
                   help='minus-strand reference FASTA')
    p.add_argument('-p', '--picard', default='picard.jar', help='picard.jar')
    p.add_argument('--motifs', default='drach', help="'drach', 'all_A_center' (256 NNANN) or a comma-separated list")
    p.add_argument('-s', '--batch-size', type=int, default=5000, help='reads held in memory per write')
    args = p.parse_args()

    motifs = parse_motifs(args.motifs)
    failed = []
    for strand, bam, reference, output in (
            ('forward', args.bam_forward, args.reference_forward, args.output_forward),
            ('reverse', args.bam_reverse, args.reference_reverse, args.output_reverse)):
        print(f'{strand} strand')
        if not extract_features(bam, reference, args.picard, output, motifs, args.batch_size):
            failed.append(strand)
    if failed:
        sys.exit(f"failed: {', '.join(failed)} strand")
