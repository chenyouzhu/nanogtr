BASES = 'ATCG'

DRACH = [d + r + 'AC' + h for d in 'AGT' for r in 'AG' for h in 'ACT']
NNANN = [a + b + 'A' + c + d for a in BASES for b in BASES for c in BASES for d in BASES]


def parse_motifs(spec):
    """'drach', 'all' or a comma-separated list of 5-mers (optionally prefixed with 'custom:')."""
    key = spec.lower()
    if key == 'drach':
        return list(DRACH)
    if key in ('all', 'all_a', 'all_with_a', 'all_a_center', 'nnan', 'nnann'):
        return list(NNANN)
    if key.startswith('custom:'):
        spec = spec[len('custom:'):]
    return [m.strip().upper() for m in spec.split(',') if m.strip()]
