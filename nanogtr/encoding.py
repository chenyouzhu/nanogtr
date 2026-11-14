import re

import pandas as pd

WINDOW = 5
MATCH_FIELDS = ['del', 'del_len', 'ins', 'ins_len', 'ins_site',
                'mis', 'mis_A', 'mis_T', 'mis_C', 'mis_G']


def encode_match(token):
    vec = [0] * len(MATCH_FIELDS)
    if token == '.':
        return vec
    if token == '*':
        vec[4] = 1
    elif token in ('A', 'T', 'C', 'G'):
        vec[5] = 1
        vec[6 + 'ATCG'.index(token)] = 1
    elif re.match(r'^-\d+$', token):
        vec[0], vec[1] = 1, int(token[1:])
    elif re.match(r'^\+\d+$', token):
        vec[2], vec[3] = 1, int(token[1:])
    return vec


def phred(char):
    return ord(char) - 33


def expand_window(df):
    """Replace match_i/qua_i of the 5-mer window by numeric feature columns."""
    df = df.copy()
    for i in range(WINDOW):
        # deleted bases have no quality of their own
        df.loc[df[f'match_{i}'] == '*', f'qua_{i}'] = '!'
    for i in range(WINDOW):
        df[f'qua_{i}'] = df[f'qua_{i}'].map(phred)
        cols = [f'{name}_{i}' for name in MATCH_FIELDS]
        expanded = pd.DataFrame(df[f'match_{i}'].map(encode_match).tolist(),
                                columns=cols, index=df.index)
        df = df.drop(columns=f'match_{i}').join(expanded)
    return df
