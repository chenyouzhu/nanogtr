"""Single-read feature tables for RNA004: base-calling errors of a 5-mer plus its f5c event."""
import csv

import numpy as np
import pandas as pd

from .encoding import MATCH_FIELDS, WINDOW, expand_window

WINDOW_COLUMNS = [f'{name}_{i}' for i in range(WINDOW)
                  for name in ('chr', 'pos', 'match', 'qua')] + ['read_name']
EVENT_COLUMNS = ['contig', 'position', 'reference_kmer', 'read_name', 'strand',
                 'event_index', 'event_level_mean', 'event_stdv', 'event_length',
                 'model_kmer', 'model_mean', 'model_stdv', 'standardized_level',
                 'start_idx', 'end_idx']
EVENT_NUMERIC = ['position', 'event_level_mean', 'event_stdv', 'event_length',
                 'model_mean', 'model_stdv']

BASE_FEATURES = [f'{name}_{i}' for i in range(WINDOW) for name in MATCH_FIELDS + ['qua']]
SIGNAL_FEATURES = ['sig_level_mean', 'sig_level_stdv', 'sig_dwell', 'sig_log_dwell',
                   'sig_level_diff', 'sig_norm_level', 'sig_abs_norm_level']
FEATURES = BASE_FEATURES + SIGNAL_FEATURES

KEYS = ['read_name', 'contig', 'position']


def strip_strand(names):
    return names.astype(str).str.replace(r'_[FR]$', '', regex=True)


def read_windows(path):
    # skiprows: some versions of split_genome_features glue the header to the first record
    df = pd.read_csv(path, sep='\t', header=None, skiprows=1, names=WINDOW_COLUMNS,
                     usecols=range(len(WINDOW_COLUMNS)), dtype=str,
                     quoting=csv.QUOTE_NONE, keep_default_na=False)
    df = df[df['match_0'] != 'U'].reset_index(drop=True)
    df['contig'] = strip_strand(df['chr_0'])
    # f5c reports the 0-based start of the 5-mer
    df['position'] = df['pos_0'].astype(int) - 1
    return df


def read_events(path, longest=False):
    df = pd.read_csv(path, sep='\t', header=None, names=EVENT_COLUMNS,
                     usecols=KEYS + EVENT_NUMERIC[1:], on_bad_lines='skip', low_memory=False)
    df[EVENT_NUMERIC] = df[EVENT_NUMERIC].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=EVENT_NUMERIC)
    df['read_name'] = df['read_name'].astype(str)
    df['contig'] = strip_strand(df['contig'])
    df['position'] = df['position'].astype(int)

    level, model = df['event_level_mean'], df['model_mean']
    df['sig_level_mean'] = level
    df['sig_level_stdv'] = df['event_stdv']
    df['sig_dwell'] = df['event_length']
    df['sig_log_dwell'] = np.log1p(df['event_length'].clip(lower=0))
    df['sig_level_diff'] = level - model
    df['sig_norm_level'] = ((level - model) / df['model_stdv'].replace(0, np.nan)).fillna(0.0)
    df['sig_abs_norm_level'] = df['sig_norm_level'].abs()
    df = df.drop(columns=['model_mean', 'model_stdv'])

    if longest:
        df = df.sort_values('sig_dwell', ascending=False).drop_duplicates(KEYS)
    return df


def build_table(window_file, event_file, longest=False):
    """One row per event of the motif on a read; longest=True keeps one event per read and site."""
    df = read_windows(window_file).merge(read_events(event_file, longest), on=KEYS)
    return expand_window(df)
