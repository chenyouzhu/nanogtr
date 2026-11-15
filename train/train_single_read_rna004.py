#!/usr/bin/env python3
"""Train per-motif single-read m6A models for RNA004 from GTR reads.

Reads aligned to m6A_* contigs are positives and reads on A_* contigs are negatives.
IVT mRNA reads can be added as further negatives (--neg-feature-folder/--neg-event-folder).
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV

from nanogtr.features import FEATURES, build_table
from nanogtr.motifs import parse_motifs

SEED = 42

PARAM_GRID = {
    'n_estimators': [200, 300, 400, 600],
    'max_depth': [3, 4, 5, 6, 8],
    'learning_rate': [0.02, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5, 10],
    'gamma': [0, 0.5, 1, 2],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2, 5],
}

# xgboost 2.1 still reads this attribute, which scikit-learn >= 1.6 no longer sets
xgb.XGBClassifier._estimator_type = 'classifier'


def auc_scorer(model, X, y):
    return roc_auc_score(y, model.predict_proba(X)[:, 1])


def load_gtr(motif, args):
    df = build_table(os.path.join(args.split_folder, f'{motif}_single_read.txt'),
                     os.path.join(args.event_folder, f'eventalign_with_{motif}.txt'),
                     longest=True)
    prefix = df['chr_2'].str.replace(r'_[FR]$', '', regex=True).str.split('_').str[0]
    df = df[prefix.isin(['m6A', 'A'])].copy()
    df['m6A'] = (prefix[df.index] == 'm6A').astype(int)
    return df


def load_ivt(motif, args):
    df = build_table(os.path.join(args.neg_feature_folder, f'{motif}_single_read.txt'),
                     os.path.join(args.neg_event_folder, f'eventalign_with_{motif}.txt'),
                     longest=True)
    df['m6A'] = 0
    df['read_name'] = 'IVT:' + df['read_name']
    reads = df['read_name'].unique()
    if not len(reads):
        return df
    if args.neg_frac < 1:
        n = max(1, int(round(len(reads) * args.neg_frac)))
        reads = np.random.default_rng(SEED).choice(reads, size=n, replace=False)
    if args.neg_max_reads and len(reads) > args.neg_max_reads:
        reads = np.random.default_rng(SEED).choice(reads, size=args.neg_max_reads, replace=False)
    return df[df['read_name'].isin(reads)]


def fit(df, motif, args, threads):
    X, y, reads = df[FEATURES].astype(float), df['m6A'], df['read_name']

    # hold out 20% of the reads; a read never contributes to both sides
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train, test = next(split.split(X, y, reads))
    X_train, y_train, reads_train = X.iloc[train], y.iloc[train], reads.iloc[train]
    X_test, y_test = X.iloc[test], y.iloc[test]

    n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    if args.scale_pos_weight == 'balanced' or (args.scale_pos_weight is None and not args.neg_feature_folder):
        weight = n_neg / n_pos if n_pos else 1.0
    else:
        weight = float(args.scale_pos_weight or 1.0)

    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', tree_method='hist',
                              scale_pos_weight=weight, random_state=SEED, n_jobs=threads)
    cv_auc = params = None
    folds = min(args.cv_folds, reads_train.nunique())
    if folds < 2:
        model.fit(X_train, y_train)
    else:
        search = RandomizedSearchCV(model, PARAM_GRID, n_iter=args.n_iter, scoring=auc_scorer,
                                    cv=GroupKFold(n_splits=folds), random_state=SEED, n_jobs=1)
        search.fit(X_train, y_train, groups=reads_train)
        model, cv_auc, params = search.best_estimator_, float(search.best_score_), search.best_params_

    prob = model.predict_proba(X_test)[:, 1]
    stats = {
        'motif': motif,
        'n_train': len(train),
        'n_test': len(test),
        'n_reads_train': int(reads_train.nunique()),
        'pos_train': n_pos,
        'neg_train': n_neg,
        'scale_pos_weight': round(weight, 4),
        'cv_best_auc': round(cv_auc, 4) if cv_auc is not None else None,
        'test_auc': round(float(roc_auc_score(y_test, prob)), 4) if y_test.nunique() > 1 else None,
        'test_accuracy': round(float(accuracy_score(y_test, prob >= 0.5)), 4),
        'best_params': params,
    }

    return model, stats


def train_motif(motif, args, threads):
    try:
        df = load_gtr(motif, args)
        if args.neg_feature_folder:
            df = pd.concat([df, load_ivt(motif, args)], ignore_index=True)
        if df['m6A'].nunique() < 2 or len(df) < args.min_rows:
            print(f'{motif}: skipped, {len(df)} labelled rows', flush=True)
            return None
        model, stats = fit(df, motif, args, threads)
    except Exception as err:
        print(f'{motif}: failed, {err}', flush=True)
        return None

    model.save_model(os.path.join(args.model_dir, f'{motif}_single_read.json'))
    with open(os.path.join(args.model_dir, f'{motif}_single_read.meta.json'), 'w') as out:
        json.dump({'feature_columns': FEATURES, **stats}, out, indent=2)
    print(f"{motif}: rows={stats['n_train']} cv_auc={stats['cv_best_auc']} "
          f"test_auc={stats['test_auc']}", flush=True)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-s', '--split-folder', required=True, help='GTR <motif>_single_read.txt files')
    p.add_argument('-e', '--event-folder', required=True, help='GTR eventalign_with_<motif>.txt files')
    p.add_argument('-o', '--model-dir', required=True)
    p.add_argument('--motifs', default='drach', help="'drach', 'all' (256 NNANN) or a comma-separated list")
    p.add_argument('-w', '--workers', type=int, default=4, help='motifs trained in parallel')
    p.add_argument('--n-iter', type=int, default=40, help='hyper-parameter candidates per motif')
    p.add_argument('--cv-folds', type=int, default=5)
    p.add_argument('--min-rows', type=int, default=50)
    p.add_argument('--neg-feature-folder', help='IVT <motif>_single_read.txt files, used as negatives')
    p.add_argument('--neg-event-folder', help='IVT eventalign_with_<motif>.txt files')
    p.add_argument('--neg-frac', type=float, default=1.0, help='fraction of IVT reads to use per motif')
    p.add_argument('--neg-max-reads', type=int, default=0, help='cap on IVT reads per motif (0 = no cap)')
    p.add_argument('--scale-pos-weight', default=None,
                   help="'balanced' (negatives/positives) or a number; "
                        "default: balanced without IVT reads, 1 with IVT reads")
    args = p.parse_args()
    if bool(args.neg_feature_folder) != bool(args.neg_event_folder):
        p.error('--neg-feature-folder and --neg-event-folder go together')

    motifs = parse_motifs(args.motifs)
    os.makedirs(args.model_dir, exist_ok=True)
    cores = int(os.environ.get('SLURM_CPUS_PER_TASK') or os.cpu_count() or 1)
    workers = max(1, min(args.workers, len(motifs)))
    threads = max(1, cores // workers)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(train_motif, m, args, threads) for m in motifs]
        summary = [job.result() for job in as_completed(jobs)]
    summary = [s for s in summary if s]
    if summary:
        pd.DataFrame(summary).sort_values('motif').to_csv(
            os.path.join(args.model_dir, 'training_summary.csv'), index=False)
    print(f'{len(summary)}/{len(motifs)} models written to {args.model_dir}')


if __name__ == '__main__':
    main()
