"""
Split Genome Features Module

This module splits single-read genome features by 5-mer motifs.
Processes mismatches, insertions, deletions, and quality scores.
"""

import numpy as np
import argparse
import time
import gc
import os
from typing import List, Dict, Optional


def get_five_mer_from_list(data_list: List, motif_list: List[str]) -> Dict[str, List]:
    """
    Extract 5-mer motifs from a list of genomic data.
    
    Args:
        data_list: List of genomic data rows
        motif_list: List of 5-mer motifs to search for
        
    Returns:
        Dictionary with motifs as keys and lists of positions as values
    """
    list_length = len(data_list)
    
    # Find positions to skip (where column 6 is '.')
    pos_skip_list = [idx for idx in range(list_length) if data_list[idx][6] == '.']
    pos_without_skip_list = [idx for idx in range(list_length) if idx not in pos_skip_list]
    real_pos_length = len(pos_without_skip_list)
    
    temp_dict = {item: [] for item in motif_list}

    # Extract 5-mers centered at each position
    for idx in range(2, real_pos_length):
        if idx + 2 >= real_pos_length:
            break
        
        # Build 5-mer from positions -2 to +2 relative to current position
        temp_five_mer = "".join([
            data_list[pos_without_skip_list[idx + j]][7] 
            for j in range(-2, 3)
        ]).upper()
        
        if temp_five_mer in motif_list:
            temp_dict[temp_five_mer].append([
                data_list[pos_without_skip_list[idx + j]][6] 
                for j in range(-2, 3)
            ])

    return temp_dict


def process_df_mismatch(data_list: List, pos_list: List) -> List:
    """
    Process mismatches, insertions, and deletions for specific positions.
    
    Args:
        data_list: Input data list with multiple rows
        pos_list: Reference position list for filtering
        
    Returns:
        Processed data list with added 'match' column
    """
    list_length = len(data_list)
    output_list = []
    
    for i in range(list_length):
        # Check if column 6 is in pos_list
        if data_list[i][6] not in pos_list:
            continue
        
        # Process 'M' (Match) operations
        if data_list[i][8] == 'M':
            if data_list[i][7] != data_list[i][4]:  # Mismatch
                data_list[i].append(data_list[i][4])
            elif i + 1 == list_length:  # Last row
                data_list[i].append('.')
            elif data_list[i + 1][8] == 'D':  # Next is deletion
                j = i + 1
                while j < list_length and data_list[j][8] == 'D':
                    j += 1
                data_list[i].append(f'-{j - i - 1}')
            elif data_list[i + 1][8] == 'I':  # Next is insertion
                j = i + 1
                while j < list_length and data_list[j][8] == 'I':
                    j += 1
                data_list[i].append(f'+{j - i - 1}')
            else:
                data_list[i].append('.')
        elif data_list[i][8] == 'D':  # Deletion
            data_list[i].append('*')
        else:  # Unknown type
            data_list[i].append('U')
        
        output_list.append(data_list[i])
    
    return output_list


def process_dict(feature_dict: Dict[str, List]) -> Dict[str, List]:
    """
    Reorganize feature dictionary by splitting sublists into 5-element chunks.
    
    Args:
        feature_dict: Dictionary of features by motif
        
    Returns:
        Reorganized feature dictionary
    """
    for motif in feature_dict:
        result_list = []
        
        for sublist in feature_dict[motif]:
            if not sublist:
                continue
            
            # Split into 5-element chunks
            for i in range(0, len(sublist), 5):
                result_list.append(sublist[i:i + 5])
        
        feature_dict[motif] = result_list
    
    return feature_dict


def add_header(motif_list: List[str], output_dir: str = ".") -> None:
    """
    Add headers to output files for each motif.
    
    Args:
        motif_list: List of motifs
        output_dir: Directory for output files
    """
    headers = []
    for i in range(5):
        headers.extend([f'chr_{i}', f'pos_{i}', f'match_{i}', f'qua_{i}'])
    headers.append('read_name')
    
    os.makedirs(output_dir, exist_ok=True)
    
    for item in motif_list:
        output_file = os.path.join(output_dir, f"{item}_single_read.txt")
        with open(output_file, 'w') as f:
            f.write('\t'.join(headers) + '\n')  # Fixed typo: was 'n', now '\n'


def write_dict_to_txt(feature_dict: Dict[str, List], output_dir: str = ".") -> None:
    """
    Write feature dictionary to text files.
    
    Args:
        feature_dict: Dictionary of features by motif
        output_dir: Directory for output files
    """
    for item, data_list in feature_dict.items():
        output_file = os.path.join(output_dir, f"{item}_single_read.txt")
        
        with open(output_file, 'a') as f:
            for sublist in data_list:
                if len(sublist) < 5:
                    continue
                
                row_data = []
                for j in range(5):
                    match_val = sublist[j][9]
                    qua_val = sublist[j][5]
                    chr_val = sublist[j][2]
                    pos_val = sublist[j][6]
                    row_data.extend([chr_val, pos_val, match_val, qua_val])
                
                row_data.append(sublist[0][0])
                f.write('\t'.join(map(str, row_data)) + '\n')
        
        print(f"{item} features have been written to {output_file}")


def split_single_read(in_file: str, motif_list: List[str], 
                     output_dir: str = ".",
                     batch_size: int = 5000) -> None:
    """
    Split single-read file into motif-specific files.
    
    Args:
        in_file: Input single-read text file
        motif_list: List of 5-mer motifs to extract
        output_dir: Directory for output files
        batch_size: Number of reads to process before writing to disk
    """
    start_time = time.time()
    
    print("=" * 60)
    print("Split Genome Features Pipeline")
    print("=" * 60)
    print(f"Input file: {in_file}")
    print(f"Motifs: {', '.join(motif_list)}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)
    
    # Initialize output files with headers
    add_header(motif_list, output_dir)
    
    with open(in_file, 'r') as file:
        file.readline()  # Skip header
        
        tem_list = []
        feature_dict = {item: [] for item in motif_list}
        current_read_name = None
        read_count = 0
        read_lane = 0
        
        for line in file:
            read_lane += 1
            fields = line.strip().split('\t')
            
            if len(fields) < 9:
                continue
            
            read_name = fields[0]
            
            if current_read_name is None:
                current_read_name = read_name
            
            if read_name == current_read_name:
                tem_list.append(fields)
            else:
                # Progress reporting
                if read_count % 1000 == 0:
                    print(f"Processed: {read_count} reads and {read_lane} lines")
                    print(f"Elapsed time: {time.time() - start_time:.2f}s")
                
                read_count += 1
                
                # Process current read
                temp_dict = get_five_mer_from_list(tem_list, motif_list)
                for motif in temp_dict:
                    for i in range(len(temp_dict[motif])):
                        feature_dict[motif].append(
                            process_df_mismatch(tem_list, temp_dict[motif][i])
                        )
                
                # Write to disk periodically
                if read_count % batch_size == 0:
                    feature_dict = process_dict(feature_dict)
                    write_dict_to_txt(feature_dict, output_dir)
                    feature_dict = {item: [] for item in motif_list}
                    gc.collect()
                
                tem_list = [fields]
                current_read_name = read_name
        
        # Process last read
        temp_dict = get_five_mer_from_list(tem_list, motif_list)
        for motif in temp_dict:
            for i in range(len(temp_dict[motif])):
                feature_dict[motif].append(
                    process_df_mismatch(tem_list, temp_dict[motif][i])
                )
        
        feature_dict = process_dict(feature_dict)
        write_dict_to_txt(feature_dict, output_dir)
    
    end_time = time.time()
    print("=" * 60)
    print(f"✓ Processing complete!")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Total reads processed: {read_count}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process genome single reads into motif-specific files",
        epilog="""
Motif Options:
  drach          - 18 DRACH motifs for m6A sites (DRACH pattern)
  all_A_center   - All 256 5-mers with A at center position
  custom         - Comma-separated custom motif list (e.g., AAACA,GAACT,TAACA)
        """
    )
    parser.add_argument(
        "in_file",
        help="Input single-read txt file"
    )
    parser.add_argument(
        "motif_option",
        help="Motif option: 'drach', 'all_A_center', or comma-separated custom list"
    )
    parser.add_argument(
        "-o", "--output",
        default=".",
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=5000,
        help="Batch size for writing to disk (default: 5000)"
    )
    
    args = parser.parse_args()
    
    # Parse motif option
    try:
        from motif_utils import get_motif_list
        
        if args.motif_option.lower() == 'drach':
            motif_list = get_motif_list('drach')
            print(f"Using DRACH motifs ({len(motif_list)} motifs)")
        elif args.motif_option.lower() in ['all_a_center', 'all_with_a']:
            motif_list = get_motif_list('all_with_A')
            print(f"Using all 5-mers with A at center ({len(motif_list)} motifs)")
        else:
            # Custom motif list
            motif_list = args.motif_option.split(',')
            print(f"Using custom motif list ({len(motif_list)} motifs)")
    except ImportError:
        # Fallback if motif_utils not available
        print("Warning: motif_utils not found, treating as custom motif list")
        motif_list = args.motif_option.split(',')
    
    split_single_read(args.in_file, motif_list, args.output, args.batch_size)