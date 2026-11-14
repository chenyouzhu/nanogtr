#!/usr/bin/env python3
"""Train per-motif single-read m6A models for RNA002 from GTR reads (base-calling errors only)."""
import argparse
import csv
import os
import pickle
import sys

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

from nanogtr.encoding import expand_window
from nanogtr.motifs import parse_motifs
from nanogtr.reference import get_motif_dict, site_of

PARAM_GRID = {
    'max_depth': [4, 5, 6],
    'min_child_weight': [1],
    'gamma': [0],
    'subsample': [1.0],
    'colsample_bytree': [1.0],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1],
}


# scoring='roc_auc' gives NaN for XGBClassifier with scikit-learn >= 1.6 and xgboost 2.1
def auc_scorer(model, X, y):
    return roc_auc_score(y, model.predict_proba(X)[:, 1])


def fit_xgboost(X, y, motif):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc',
                              use_label_encoder=False, random_state=42)
    search = GridSearchCV(model, PARAM_GRID, scoring=auc_scorer, cv=5, verbose=1, n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f'{motif}: best parameters {search.best_params_}, CV AUC {search.best_score_:.4f}')
    print(f'{motif}: test accuracy {accuracy_score(y_test, model.predict(X_test)):.4f}, '
          f'test AUC {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}')
    return model


def train_motif(motif, split_folder, motif_dict, model_dir):
    feature_file = f'{split_folder}/{motif}_single_read.txt'
    if not os.path.exists(feature_file):
        print(f'{feature_file} not found, skipped')
        return False
    df = expand_window(pd.read_csv(feature_file, sep='\t', quoting=csv.QUOTE_NONE))
    if motif not in motif_dict:
        print(f'{motif} is not a site of the reference, skipped')
        return False

    seq_id, pos = site_of(motif_dict[motif])
    at_site = df['pos_2'] == pos
    df_a = df[(df['chr_2'] == f'A_{seq_id}_F') & at_site].copy()
    df_m6a = df[(df['chr_2'] == f'm6A_{seq_id}_F') & at_site].copy()
    df_a['m6A'] = 0
    df_m6a['m6A'] = 1
    n = min(len(df_a), len(df_m6a))
    if n == 0:
        print(f'{motif}: no A or no m6A reads at the site, skipped')
        return False

    # same number of reads per class
    df = pd.concat([df_a.sample(n=n, random_state=42), df_m6a.sample(n=n, random_state=42)], ignore_index=True)
    df = df.drop(columns=[f'{name}_{i}' for i in range(5) for name in ('chr', 'pos')] + ['read_name'])
    print(f'{motif}: {n} reads per class')
    model = fit_xgboost(df.drop(columns='m6A'), df['m6A'], motif)

    os.makedirs(model_dir, exist_ok=True)
    with open(f'{model_dir}/{motif}_single_read.pkl', 'wb') as f:
        pickle.dump(model, f)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--motif', default='drach', help="'drach', 'all_with_A' (256 NNANN) or custom:AAACT,GGACT")
    p.add_argument('--split_folder', default='./output', help='folder with <motif>_single_read.txt files')
    p.add_argument('--reference_file', default='alignment/plus_strand/reference.fa', help='GTR reference FASTA')
    p.add_argument('--output', default='./single_read_model_rna002', help='model directory')
    args = p.parse_args()
    for path in (args.split_folder, args.reference_file):
        if not os.path.exists(path):
            sys.exit(f'{path} not found')

    motifs = parse_motifs(args.motif)
    motif_dict = get_motif_dict(args.reference_file)
    print(f'{len(motifs)} motifs, {len(motif_dict)} sites in {args.reference_file}')

    trained = 0
    for motif in motifs:
        try:
            if train_motif(motif, args.split_folder, motif_dict, args.output):
                trained += 1
        except Exception as e:
            print(f'{motif}: training failed, {e}')
    print(f'{trained}/{len(motifs)} models written to {args.output}')


if __name__ == '__main__':
    main()
