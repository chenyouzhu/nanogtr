"""
Motif Utilities Module

This module provides functions to generate common motif patterns for RNA modification analysis.
"""

from typing import List


def generate_all_5mer_with_center_A() -> List[str]:
    """
    Generate all possible 5-mer motifs with A at center position.
    Format: NNANN where A is at the center position (4^4 = 256 motifs).
    
    Returns:
        List of 256 5-mer sequences with A at center
    """
    nucleotides = ['A', 'T', 'C', 'G']
    motifs = []
    
    for n1 in nucleotides:
        for n2 in nucleotides:
            for n4 in nucleotides:
                for n5 in nucleotides:
                    motifs.append(f"{n1}{n2}A{n4}{n5}")
    
    return motifs


def generate_drach_motifs() -> List[str]:
    """
    Generate DRACH motifs for m6A modification sites.
    
    DRACH pattern:
    - D = A, G, or T (not C)
    - R = A or G (purines)
    - A = A
    - C = C
    - H = A, C, or T (not G)
    
    Total: 3 × 2 × 1 × 1 × 3 = 18 motifs
    
    Returns:
        List of 18 DRACH motif sequences
    """
    D = ['A', 'G', 'T']  # Not C
    R = ['A', 'G']       # Purines
    A = ['A']            # Adenine
    C = ['C']            # Cytosine
    H = ['A', 'C', 'T']  # Not G
    
    motifs = []
    for d in D:
        for r in R:
            for a in A:
                for c in C:
                    for h in H:
                        motifs.append(f"{d}{r}{a}{c}{h}")
    
    return motifs


def get_motif_list(motif_type: str) -> List[str]:
    """
    Get a predefined motif list based on type.
    
    Args:
        motif_type: Type of motifs to generate
            - 'all_A_center' or 'all_with_A': All 256 5-mers with A at center
            - 'drach': 18 DRACH motifs for m6A
            - 'custom:MOTIF1,MOTIF2,...': Custom comma-separated list
            
    Returns:
        List of motif sequences
        
    Raises:
        ValueError: If motif_type is invalid
    """
    if motif_type == 'all_with_A' or motif_type == 'all_A_center':
        return generate_all_5mer_with_center_A()
    elif motif_type == 'drach' or motif_type == 'DRACH':
        return generate_drach_motifs()
    elif motif_type.startswith('custom:'):
        return motif_type.replace('custom:', '').split(',')
    else:
        raise ValueError(
            f"Invalid motif_type: {motif_type}. "
            f"Options: 'all_with_A', 'drach', or 'custom:MOTIF1,MOTIF2,...'"
        )


def print_motif_info(motif_list: List[str]) -> None:
    """
    Print information about a motif list.
    
    Args:
        motif_list: List of motifs
    """
    print(f"Total motifs: {len(motif_list)}")
    print(f"First 10 motifs: {', '.join(motif_list[:10])}")
    if len(motif_list) > 10:
        print(f"Last 10 motifs: {', '.join(motif_list[-10:])}")


if __name__ == "__main__":
    print("=" * 60)
    print("Motif Generation Examples")
    print("=" * 60)
    
    print("\n1. DRACH Motifs (m6A sites):")
    drach = generate_drach_motifs()
    print_motif_info(drach)
    print(f"All DRACH motifs: {', '.join(drach)}")
    
    print("\n2. All 5-mers with A at center (256 motifs):")
    all_a = generate_all_5mer_with_center_A()
    print_motif_info(all_a)
    
    print("\n" + "=" * 60)