"""
GTR Feature Extraction and Splitting Module

This module extracts GTR features from BAM files and splits them by motifs.
Combines feature extraction and motif-based splitting in one pipeline.
"""

import os
import subprocess
import sys
import shutil
import numpy as np
import argparse
import time
import gc
from pathlib import Path
from typing import List, Dict, Optional


# ============================================================================
# MOTIF SPLITTING FUNCTIONS (from split_genome_features.py)
# ============================================================================

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
            f.write('\t'.join(headers) + '\n')


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
    print(f"Motifs: {', '.join(motif_list[:10])}" + 
          (f" ... ({len(motif_list)} total)" if len(motif_list) > 10 else ""))
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


# ============================================================================
# GTR EXTRACTION CLASSES AND FUNCTIONS
# ============================================================================

class GTRFeatureExtractor:
    """Extract GTR features from BAM alignment files."""
    
    def __init__(self, bam_file: str, reference_fasta: str, 
                 picard_jar: str = "picard.jar",
                 output_dir: str = "./output",
                 motif_list: Optional[List[str]] = None,
                 batch_size: int = 5000):
        """
        Initialize the GTR feature extractor.
        
        Args:
            bam_file: Path to input BAM file
            reference_fasta: Path to reference FASTA file
            picard_jar: Path to picard.jar file
            output_dir: Directory for output files
            motif_list: List of motifs for splitting (if None, uses DRACH)
            batch_size: Batch size for splitting (default: 5000)
        """
        self.bam_file = Path(bam_file)
        self.reference_fasta = Path(reference_fasta)
        self.picard_jar = Path(picard_jar)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        
        # Set default motif list if not provided
        if motif_list is None:
            from motif_utils import get_motif_list
            self.motif_list = get_motif_list('drach')
        else:
            self.motif_list = motif_list
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output files
        self.filtered_bam = self.output_dir / "filtered.bam"
        self.reference_dict = self.reference_fasta.with_suffix('.dict')
        self.single_read_txt = self.output_dir / "single_read.txt"
        self.final_output = self.output_dir / "single_read_without_skip.txt"
        
        # Handle reference file location
        self.working_reference = self.reference_fasta
    
    def _run_command(self, command: str, description: str) -> bool:
        """
        Run a shell command and handle errors.
        
        Args:
            command: Shell command to execute
            description: Description of the command for logging
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Running: {description}")
        print(f"Command: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"✓ {description} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Error in {description}:", file=sys.stderr)
            print(f"  {e.stderr}", file=sys.stderr)
            return False
    
    def check_dependencies(self) -> bool:
        """
        Check if required tools are installed.
        
        Returns:
            True if all dependencies are available
        """
        dependencies = ['samtools', 'java', 'sam2tsv', 'awk']
        missing = []
        
        for dep in dependencies:
            try:
                subprocess.run(
                    ['which', dep],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError:
                missing.append(dep)
        
        if missing:
            print(f"Error: Missing dependencies: {', '.join(missing)}", file=sys.stderr)
            return False
        
        # Check if picard jar exists
        if not self.picard_jar.exists():
            print(f"Error: Picard JAR not found at {self.picard_jar}", file=sys.stderr)
            return False
        
        return True
    
    def filter_bam(self) -> bool:
        """
        Filter BAM file to remove reads with empty sequences or CIGAR strings.
        
        Returns:
            True if successful
        """
        command = (
            f"samtools view -h {self.bam_file} | "
            f"awk 'BEGIN {{OFS=\"\\t\"}} /^@/ || ($10 != \"*\" && length($10) > 0 && $6 != \"*\" && $6 != \"\")' | "
            f"samtools view -b -o {self.filtered_bam}"
        )
        return self._run_command(command, "Filter BAM file")
    
    def create_sequence_dictionary(self) -> bool:
        """
        Create sequence dictionary for reference FASTA.
        The .dict file MUST be in the same directory as the reference FASTA.
        
        Returns:
            True if successful
        """
        # Check if dict already exists
        if self.reference_dict.exists():
            print(f"✓ Reference dictionary already exists: {self.reference_dict}")
            return True
        
        # Check if we can write to reference directory
        ref_dir_writable = os.access(self.reference_dict.parent, os.W_OK)
        
        if not ref_dir_writable:
            print(f"\n{'='*60}")
            print("ERROR: Cannot write to reference directory!")
            print(f"{'='*60}")
            print(f"Reference: {self.reference_fasta}")
            print(f"Directory: {self.reference_dict.parent}")
            print(f"Required file: {self.reference_dict}")
            print()
            print("SOLUTION:")
            print("Create the dictionary file manually with write permissions:")
            print(f"  sudo java -jar {self.picard_jar} CreateSequenceDictionary \\")
            print(f"    R={self.reference_fasta} \\")
            print(f"    O={self.reference_dict}")
            print()
            print(f"{'='*60}\n")
            return False
        
        # We have write permission, create dict
        command = (
            f"java -jar {self.picard_jar} CreateSequenceDictionary "
            f"R={self.reference_fasta} O={self.reference_dict}"
        )
        return self._run_command(command, "Create sequence dictionary")
    
    def convert_bam_to_tsv(self) -> bool:
        """
        Convert filtered BAM to TSV format using sam2tsv.
        
        Returns:
            True if successful
        """
        command = (
            f"samtools view -h {self.filtered_bam} | "
            f"sam2tsv -R {self.reference_fasta} > {self.single_read_txt}"
        )
        return self._run_command(command, "Convert BAM to TSV")
    
    def filter_skip_regions(self) -> bool:
        """
        Filter out rows with 'N' in column 9 (skip regions).
        
        Returns:
            True if successful
        """
        command = (
            f"cat {self.single_read_txt} | "
            f"awk '$9 != \"N\"' > {self.final_output}"
        )
        return self._run_command(command, "Filter skip regions")
    
    def split_by_motifs(self) -> bool:
        """
        Split features by motifs.
        
        Returns:
            True if successful
        """
        print("\n" + "=" * 60)
        print("Starting motif-based splitting...")
        print("=" * 60 + "\n")
        
        try:
            split_single_read(
                str(self.final_output), 
                self.motif_list, 
                str(self.output_dir),
                self.batch_size
            )
            return True
        except Exception as e:
            print(f"Error during splitting: {e}", file=sys.stderr)
            return False
    
    def extract_features(self) -> Optional[str]:
        """
        Run the complete feature extraction pipeline.
        
        Returns:
            Path to final output file if successful, None otherwise
        """
        print("=" * 60)
        print("GTR Feature Extraction Pipeline")
        print("=" * 60)
        print(f"Input BAM: {self.bam_file}")
        print(f"Reference: {self.reference_fasta}")
        print(f"Output directory: {self.output_dir}")
        print(f"Motifs: {len(self.motif_list)}")
        print("=" * 60)
        
        # Check dependencies
        if not self.check_dependencies():
            return None
        
        # Check if input files exist
        if not self.bam_file.exists():
            print(f"Error: BAM file not found: {self.bam_file}", file=sys.stderr)
            return None
        
        if not self.reference_fasta.exists():
            print(f"Error: Reference FASTA not found: {self.reference_fasta}", file=sys.stderr)
            return None
        
        # Run pipeline steps
        steps = [
            self.filter_bam,
            self.create_sequence_dictionary,
            self.convert_bam_to_tsv,
            self.filter_skip_regions,
            self.split_by_motifs
        ]
        
        for step in steps:
            if not step():
                print(f"\n✗ Pipeline failed at step: {step.__name__}", file=sys.stderr)
                return None
            print()
        
        print("=" * 60)
        print("✓ Pipeline completed successfully!")
        print(f"Single-read output: {self.final_output}")
        print(f"Motif-split files: {self.output_dir}/*_single_read.txt")
        print("=" * 60)
        
        return str(self.final_output)


def extract_gtr_features(bam_file: str, reference_fasta: str,
                         picard_jar: str = "picard.jar",
                         output_dir: str = "./output",
                         motif_list: Optional[List[str]] = None,
                         batch_size: int = 5000) -> Optional[str]:
    """
    Extract GTR features from a BAM file and split by motifs.
    
    Args:
        bam_file: Path to input BAM file
        reference_fasta: Path to reference FASTA file
        picard_jar: Path to picard.jar file (default: "picard.jar")
        output_dir: Directory for output files (default: "./output")
        motif_list: List of motifs for splitting (default: None, uses DRACH)
        batch_size: Batch size for splitting (default: 5000)
    
    Returns:
        Path to final output file if successful, None otherwise
    
    Example:
        >>> output = extract_gtr_features(
        ...     bam_file="/path/to/sample.bam",
        ...     reference_fasta="/path/to/reference.fa",
        ...     picard_jar="/path/to/picard.jar",
        ...     output_dir="./gtr_output"
        ... )
        >>> if output:
        ...     print(f"Features extracted to: {output}")
    """
    extractor = GTRFeatureExtractor(
        bam_file, reference_fasta, picard_jar, 
        output_dir, motif_list, batch_size
    )
    return extractor.extract_features()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract GTR features from BAM files and split by motifs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Motif Options:
  drach          - 18 DRACH motifs for m6A sites (default)
  all_A_center   - All 256 5-mers with A at center position
  custom         - Comma-separated custom list (e.g., AAACA,GAACT,TAACA)
  
Examples:
  # Extract and split by DRACH motifs (default):
  python extract_gtr_feature.py -b sample.bam -r reference.fa
  
  # Extract and split by all A-centered motifs:
  python extract_gtr_feature.py -b sample.bam -r reference.fa --motifs all_A_center
  
  # Extract and split by custom motifs:
  python extract_gtr_feature.py -b sample.bam -r reference.fa --motifs AAACA,GAACT,TAACA
  
  # With custom output directory:
  python extract_gtr_feature.py -b sample.bam -r reference.fa -o ./my_output
        """
    )
    parser.add_argument(
        "-b", "--bam",
        default = 'alignment/plus_strand/collect.sorted.bam',
        help="Path to input BAM file, (default path is alignment/plus_strand/collect.sorted.bam)"
    )
    parser.add_argument(
        "-r", "--reference",
        default = 'alignment/plus_strand/reference.fa',
        help="Path to reference FASTA file, (default path is alignment/plus_strand/reference.fa)"
    )
    parser.add_argument(
        "-p", "--picard",
        default="picard.jar",
        help="Path to picard.jar file (default: picard.jar)"
    )
    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--motifs",
        default="drach",
        help="Motif option: 'drach' (default), 'all_A_center', or comma-separated custom list"
    )
    parser.add_argument(
        "-s", "--batch-size",
        type=int,
        default=5000,
        help="Batch size for splitting (default: 5000)"
    )
    
    args = parser.parse_args()
    
    # Parse motif option
    try:
        from motif_utils import get_motif_list
        
        if args.motifs.lower() == 'drach':
            motif_list = get_motif_list('drach')
            print(f"Using DRACH motifs ({len(motif_list)} motifs)")
        elif args.motifs.lower() in ['all_a_center', 'all_with_a']:
            motif_list = get_motif_list('all_with_A')
            print(f"Using all 5-mers with A at center ({len(motif_list)} motifs)")
        else:
            # Custom motif list
            motif_list = args.motifs.split(',')
            print(f"Using custom motif list ({len(motif_list)} motifs)")
    except ImportError:
        print("Warning: motif_utils not found, treating as custom motif list")
        motif_list = args.motifs.split(',')
    
    result = extract_gtr_features(
        bam_file=args.bam,
        reference_fasta=args.reference,
        picard_jar=args.picard,
        output_dir=args.output,
        motif_list=motif_list,
        batch_size=args.batch_size
    )
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)