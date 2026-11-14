"""Motif sites of the ground-truth RNA (GTR) reference."""


def get_motif_dict(fasta):
    """Map the 5-mer at each site of every A_<id> record to 'A_<id>_<site>'."""
    records, name = {}, None
    with open(fasta) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                name = line[1:].split()[0]
                records[name] = []
            elif name is not None:
                records[name].append(line)

    motif_dict = {}
    for name, chunks in records.items():
        parts = name.split('_')
        base, seq_id = parts[0], parts[1]
        if base != 'A':
            continue
        seq = ''.join(chunks)
        # 16 sites per GTR, 15 nt apart, the first one centred on base 28
        for site in range(16):
            center = 27 + 15 * site
            motif_dict[seq[center - 2:center + 3]] = f'A_{seq_id}_{site}'
    return motif_dict


def site_of(name):
    """GTR id and 1-based position of a get_motif_dict value."""
    _, seq_id, site = name.split('_')
    return seq_id, 28 + 15 * int(site)
