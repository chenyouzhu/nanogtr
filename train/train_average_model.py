"""Train the per-motif average (site-level) m6A models on mixtures of GTR A and m6A reads."""
import argparse
import os
import pickle
import random
import statistics
import sys

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from nanogtr.encoding import encode_match, phred
from nanogtr.motifs import DRACH
from nanogtr.reference import get_motif_dict, site_of

MODEL_DIR = 'Ave_m6A_model'
# the models are trained on these column names
HEADER = [f'{name}_{i}' for i in range(5)
          for name in ('del', 'del_len', 'ins', 'ins_len', 'del_site', 'mis',
                       'mis_a', 'mis_t', 'mis_c', 'mis_g', 'qua', 'qua_std', 'qua_0')]
PARAM_GRID = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1, 5],
}


def split_a_m6a(folder, motif_dict):
    """Write <motif>_a_single_read.txt and <motif>_m6a_single_read.txt: the reads covering the GTR site."""
    header = '\t'.join(f'{name}_{i}' for i in range(5) for name in ('match', 'qua'))
    for motif in DRACH:
        motif_file = os.path.join(folder, f'{motif}_single_read.txt')
        if not os.path.exists(motif_file):
            print(f'{motif_file} not found, skipped')
            continue
        if motif not in motif_dict:
            print(f'{motif} is not a site of the reference, skipped')
            continue
        seq_id, pos = site_of(motif_dict[motif])
        with open(motif_file) as f, \
                open(os.path.join(folder, f'{motif}_a_single_read.txt'), 'w') as out_a, \
                open(os.path.join(folder, f'{motif}_m6a_single_read.txt'), 'w') as out_m6a:
            next(f)
            out_a.write(header + '\n')
            out_m6a.write(header + '\n')
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 21:
                    continue
                contig, center = parts[8], parts[9]
                try:
                    center = int(center)
                except ValueError:
                    continue
                if center != pos:
                    continue
                row = '\t'.join(parts[4 * i + k] for i in range(5) for k in (2, 3)) + '\n'
                if contig == f'A_{seq_id}_F':
                    out_a.write(row)
                elif contig == f'm6A_{seq_id}_F':
                    out_m6a.write(row)


def read_windows(path):
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            fields = line.strip().split('\t')
            for i in range(5):
                # deleted bases have no quality of their own
                if fields[2 * i] == '*':
                    fields[2 * i + 1] = '!'
            rows.append(fields)
    return rows


def mean_match(vectors):
    if not vectors:
        return [0] * 10
    return [sum(v[k] for v in vectors) / len(vectors) for k in range(10)]


def quality_stats(quals):
    if not quals:
        return [0, 0, 0]
    return [sum(quals) / len(quals), statistics.stdev(quals), quals.count(0) / len(quals)]


def mix_reads(m6a_file, a_file, motif, ratio, n_samples=10000):
    """Write n_samples rows, each the mean features of 20 A reads plus enough m6A reads to reach ratio."""
    os.makedirs(f'{MODEL_DIR}/{motif}', exist_ok=True)
    if ratio == 1:
        n_m6a, n_a = 20, 0
    else:
        n_m6a, n_a = int(round(20 / (1 - ratio) * ratio)), 20
    m6a_reads, a_reads = read_windows(m6a_file), read_windows(a_file)

    with open(f'{MODEL_DIR}/{motif}/{ratio}.txt', 'w') as out:
        out.write('\t'.join(HEADER) + '\n')
        for _ in range(n_samples):
            try:
                reads = random.sample(m6a_reads, n_m6a) + random.sample(a_reads, n_a)
            except ValueError:
                print(f'{motif}: fewer than {n_m6a} m6A or {n_a} A reads, {ratio}.txt left empty')
                return
            reads = [r for r in reads if r[0] != 'U']
            row = []
            for i in range(5):
                row += mean_match([encode_match(r[2 * i]) for r in reads])
                row += quality_stats([phred(r[2 * i + 1]) for r in reads])
            out.write('\t'.join(map(str, row)) + '\n')


def train_motif(motif, m6a_file, a_file):
    mix_reads(m6a_file, a_file, motif, 0, 20000)
    mix_reads(m6a_file, a_file, motif, 0.9, 10000)
    mix_reads(m6a_file, a_file, motif, 1, 10000)

    out_dir = f'./{MODEL_DIR}/{motif}'
    df_1 = pd.read_csv(f'{out_dir}/1.txt', sep='\t')
    df_09 = pd.read_csv(f'{out_dir}/0.9.txt', sep='\t')
    df_0 = pd.read_csv(f'{out_dir}/0.txt', sep='\t')
    df_m6a = pd.concat([df_1, df_09])
    df_m6a['m6A'] = 1
    df_0['m6A'] = 0
    df = pd.concat([df_0, df_m6a])
    X, y = df.drop('m6A', axis=1), df['m6A']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    search = GridSearchCV(xgb.XGBClassifier(objective='binary:logistic', random_state=42), PARAM_GRID,
                          scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{motif}: best parameters {search.best_params_}')
    print(classification_report(y_test, y_pred))

    model_path = os.path.join(out_dir, f'{motif}_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'{motif}: test accuracy {accuracy:.2%}, model saved to {model_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-r', '--reference', default='alignment/plus_strand/reference.fa', help='GTR reference FASTA')
    p.add_argument('-d', '--data_dir', default='./output', help='folder with <motif>_single_read.txt files')
    args = p.parse_args()
    for path in (args.reference, args.data_dir):
        if not os.path.exists(path):
            sys.exit(f'{path} not found')

    motif_dict = get_motif_dict(args.reference)
    print(f'{len(motif_dict)} sites in {args.reference}')
    split_a_m6a(args.data_dir, motif_dict)

    trained = 0
    for motif in DRACH:
        m6a_file = os.path.join(args.data_dir, f'{motif}_m6a_single_read.txt')
        a_file = os.path.join(args.data_dir, f'{motif}_a_single_read.txt')
        if not (os.path.exists(m6a_file) and os.path.exists(a_file)):
            print(f'{motif}: no A/m6A read files, skipped')
            continue
        try:
            train_motif(motif, m6a_file, a_file)
            trained += 1
        except Exception as e:
            print(f'{motif}: training failed, {e}')
    print(f'{trained}/{len(DRACH)} models written to ./{MODEL_DIR}')


if __name__ == '__main__':
    main()
