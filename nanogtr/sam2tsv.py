"""BAM -> jvarkit sam2tsv table -> one table of 5-mer windows per motif."""
import os
import shlex
import shutil
import subprocess
import sys
import time
from itertools import groupby
from pathlib import Path

from .motifs import DRACH

# sam2tsv columns used: 0 read name, 2 chrom, 4 read base, 5 quality, 6 ref pos, 7 ref base, 8 CIGAR op
KEEP_ALIGNED = r"""awk 'BEGIN {OFS="\t"} /^@/ || ($10 != "*" && length($10) > 0 && $6 != "*" && $6 != "")'"""


def get_five_mer_from_list(rows, motifs):
    """Reference positions of every 5-mer window of one read whose reference sequence is a motif."""
    aligned = [r for r in rows if r[6] != '.']
    hits = {m: [] for m in motifs}
    for i in range(len(aligned) - 4):
        window = aligned[i:i + 5]
        kmer = ''.join(r[7] for r in window).upper()
        if kmer in hits:
            hits[kmer].append([r[6] for r in window])
    return hits


def indel_after(rows, i):
    op = rows[i + 1][8] if i + 1 < len(rows) else ''
    if op not in ('D', 'I'):
        return '.'
    j = i + 1
    while j < len(rows) and rows[j][8] == op:
        j += 1
    return f"{'-' if op == 'D' else '+'}{j - i - 1}"


def process_df_mismatch(rows, positions):
    """Rows of one window, each with a match token: '.', mismatched base, -n, +n, '*' (deleted) or 'U'."""
    out = []
    for i, row in enumerate(rows):
        if row[6] not in positions:
            continue
        if row[8] == 'D':
            token = '*'
        elif row[8] != 'M':
            token = 'U'
        elif row[7] != row[4]:
            token = row[4]
        else:
            token = indel_after(rows, i)
        # stored on the row itself; write_dict_to_txt reads it back as column 9
        row.append(token)
        out.append(row)
    return out


def process_dict(features):
    for motif, windows in features.items():
        features[motif] = [w[i:i + 5] for w in windows for i in range(0, len(w), 5)]
    return features


def add_header(motifs, output_dir='.'):
    header = [f'{name}_{i}' for i in range(5) for name in ('chr', 'pos', 'match', 'qua')]
    os.makedirs(output_dir, exist_ok=True)
    for motif in motifs:
        with open(os.path.join(output_dir, f'{motif}_single_read.txt'), 'w') as f:
            f.write('\t'.join(header + ['read_name']) + '\n')


def write_dict_to_txt(features, output_dir='.'):
    for motif, windows in features.items():
        with open(os.path.join(output_dir, f'{motif}_single_read.txt'), 'a') as f:
            for w in windows:
                if len(w) < 5:
                    continue
                fields = [row[k] for row in w for k in (2, 6, 9, 5)]
                f.write('\t'.join(fields + [w[0][0]]) + '\n')


def split_single_read(in_file, motifs, output_dir='.', batch_size=5000):
    """Write <motif>_single_read.txt for each motif from a sam2tsv table sorted by read."""
    start = time.time()
    print(f'splitting {in_file} into {len(motifs)} motif files in {output_dir}')
    add_header(motifs, output_dir)
    features = {m: [] for m in motifs}
    n_reads = 0
    with open(in_file) as f:
        f.readline()
        rows = (line.strip().split('\t') for line in f)
        rows = (r for r in rows if len(r) >= 9)
        for _, read in groupby(rows, key=lambda r: r[0]):
            if n_reads and n_reads % batch_size == 0:
                write_dict_to_txt(process_dict(features), output_dir)
                features = {m: [] for m in motifs}
                print(f'{n_reads} reads, {time.time() - start:.0f}s')
            read = list(read)
            for motif, windows in get_five_mer_from_list(read, motifs).items():
                for positions in windows:
                    features[motif].append(process_df_mismatch(read, positions))
            n_reads += 1
    write_dict_to_txt(process_dict(features), output_dir)
    print(f'done: {n_reads} reads, {time.time() - start:.0f}s')


def q(path):
    return shlex.quote(str(path))


def run(cmd, what):
    print(f'{what}: {cmd}')
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f'{what} failed:\n{e.stderr}', file=sys.stderr)
        return False
    return True


def extract_features(bam, reference, picard='picard.jar', output_dir='./output', motifs=None, batch_size=5000):
    """Run samtools, sam2tsv and the motif split on one BAM. Returns the sam2tsv table path, None on failure."""
    bam, reference, picard, out = Path(bam), Path(reference), Path(picard), Path(output_dir)
    if motifs is None:
        motifs = DRACH
    out.mkdir(parents=True, exist_ok=True)

    missing = [tool for tool in ('samtools', 'java', 'sam2tsv', 'awk') if not shutil.which(tool)]
    if missing:
        print(f"missing tools: {', '.join(missing)}", file=sys.stderr)
        return None
    for path in (picard, bam, reference):
        if not path.exists():
            print(f'{path} not found', file=sys.stderr)
            return None

    filtered = out / 'filtered.bam'
    table = out / 'single_read.txt'
    final = out / 'single_read_without_skip.txt'
    ref_dict = reference.with_suffix('.dict')

    if not run(f'samtools view -h {q(bam)} | {KEEP_ALIGNED} | samtools view -b -o {q(filtered)}', 'filter BAM'):
        return None
    if not ref_dict.exists():
        make_dict = f'java -jar {q(picard)} CreateSequenceDictionary R={q(reference)} O={q(ref_dict)}'
        if not os.access(ref_dict.parent, os.W_OK):
            print(f'cannot write {ref_dict}; create it with:\n  {make_dict}', file=sys.stderr)
            return None
        if not run(make_dict, 'sequence dictionary'):
            return None
    if not run(f'samtools view -h {q(filtered)} | sam2tsv -R {q(reference)} > {q(table)}', 'sam2tsv'):
        return None
    if not run(f"cat {q(table)} | awk '$9 != \"N\"' > {q(final)}", 'drop skipped (N) positions'):
        return None
    try:
        split_single_read(str(final), motifs, str(out), batch_size)
    except Exception as e:
        print(f'splitting failed: {e}', file=sys.stderr)
        return None
    return str(final)
