"""
nanoGTR Train Average Model Module

This module trains average m6A modification detection models for DRACH motifs
using XGBoost classification on nanopore sequencing data.

Usage:
    nanogtr train_ave_model -r reference.fa -d /path/to/split_folder/
"""

import os
import re
import random
import statistics
import argparse
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle


# DRACH motif definitions
D = ['A', 'G', 'T']  # Not C
R = ['A', 'G']       # Purines
A = ['A']            # Adenine
C = ['C']            # Cytosine
H = ['A', 'C', 'T']  # Not G

# Generate all DRACH motif combinations
drach_list = []
for d in D:
    for r in R:
        for a in A:
            for c in C:
                for h in H:
                    drach_list.append(f"{d}{r}{a}{c}{h}")

# Feature header for output files
header = []
for i in range(5):
    header += [f"del_{i}", f"del_len_{i}", f"ins_{i}", f"ins_len_{i}", f"del_site_{i}",
               f"mis_{i}", f"mis_a_{i}", f"mis_t_{i}", f"mis_c_{i}", f"mis_g_{i}",
               f"qua_{i}", f"qua_std_{i}", f"qua_0_{i}"]


def get_motif_dict(file_path):
    """
    Read FASTA file and create motif dictionary.
    
    Args:
        file_path: Path to FASTA reference file
    
    Returns:
        dict: Mapping of 5-base motif sequences to position names
    """
    fasta_dict = {}
    current_chr = None
    current_seq = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('>'):
                if current_chr is not None:
                    fasta_dict[current_chr] = ''.join(current_seq)
                
                current_chr = line[1:].split()[0]
                current_seq = []
            
            elif line:
                current_seq.append(line)
        
        if current_chr is not None:
            fasta_dict[current_chr] = ''.join(current_seq)
    
    motif_dict = {}
    for key in fasta_dict:
        base = key.split('_')[0]
        seq = key.split('_')[1]
        
        if base == 'A':
            for i in range(16):
                n_cen = 27 + 15 * i
                name = f'{base}_{seq}_{i}'
                five_motif = fasta_dict[key][n_cen-2:n_cen+3]
                motif_dict[five_motif] = name
    
    return motif_dict


def get_a_m6a_file_from_single_read(feature_folder, motif_dict):
    """
    Split single read files into A and m6A files based on motif positions.
    
    Args:
        feature_folder: Directory containing motif single read files
        motif_dict: Dictionary mapping motifs to position names
    """
    output_header = 'match_0\tqua_0\tmatch_1\tqua_1\tmatch_2\tqua_2\tmatch_3\tqua_3\tmatch_4\tqua_4'

    for motif in drach_list:
        motif_file = os.path.join(feature_folder, f'{motif}_single_read.txt')
        a_file = os.path.join(feature_folder, f'{motif}_a_single_read.txt')
        m6a_file = os.path.join(feature_folder, f'{motif}_m6a_single_read.txt')
        
        if not os.path.exists(motif_file):
            print(f"Warning: {motif_file} not found, skipping...")
            continue
        
        name = motif_dict.get(motif)
        if name is None:
            print(f"Warning: motif {motif} not found in motif_dict, skipping...")
            continue
            
        seq = name.split('_')[1]
        pos_num = int(name.split('_')[2])
        pos = pos_num * 15 + 28
        
        a_name = f'A_{seq}_F'
        m6a_name = f'm6A_{seq}_F'
        
        with open(motif_file, 'r') as f, open(a_file, 'w') as fa, open(m6a_file, 'w') as fm6:
            next(f)  # Skip header
            
            fa.write(output_header + '\n')
            fm6.write(output_header + '\n')
            
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 21:
                    (chr_0, pos_0, match_0, qua_0,
                     chr_1, pos_1, match_1, qua_1,
                     chr_2, pos_2, match_2, qua_2,
                     chr_3, pos_3, match_3, qua_3,
                     chr_4, pos_4, match_4, qua_4,
                     read_name) = parts
                    
                    try:
                        pos_2 = int(pos_2)
                    except ValueError:
                        continue
                    
                    output_line = '\t'.join([match_0, qua_0, match_1, qua_1, 
                                            match_2, qua_2, match_3, qua_3, 
                                            match_4, qua_4]) + '\n'
                    
                    if chr_2 == a_name and pos_2 == pos:
                        fa.write(output_line)
                    elif chr_2 == m6a_name and pos_2 == pos:
                        fm6.write(output_line)
        
        print(f'Successfully processed {motif} file')


def process_match(item):
    """Process match information into feature vector."""
    if item == '.':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif item == '*':
        return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif item in ['A', 'T', 'C', 'G']:
        return [0, 0, 0, 0, 0, 1] + [1 if base == item else 0 for base in 'ATCG']
    if re.match(r'^-\d+$', item):
        return [1, int(item[1:]), 0, 0, 0, 0, 0, 0, 0, 0]
    if re.match(r'^\+\d+$', item):
        return [0, 0, 1, int(item[1:]), 0, 0, 0, 0, 0, 0]
    return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def process_quality_score(item):
    """Convert ASCII quality score to numeric value."""
    return ord(item) - 33


def process_match_list(input_list):
    """Calculate average metrics from match feature vectors."""
    list_length = len(input_list)
    if list_length == 0:
        return [0] * 10
    
    metrics = [sum(x[i] for x in input_list) / list_length for i in range(10)]
    return metrics


def process_quali_list(input_list):
    """Calculate quality score statistics."""
    if not input_list:
        return [0, 0, 0]
    
    list_length = len(input_list)
    qua_scor = sum(input_list)
    qua_zero = input_list.count(0)
    
    return [qua_scor / list_length, statistics.stdev(input_list), qua_zero / list_length]


def write_processed_data(output, mat_lists, qua_lists):
    """Write processed match and quality data to output file."""
    for j in range(5):
        processed_match = process_match_list(mat_lists[j])
        processed_quali = process_quali_list(qua_lists[j])
        match_str = "\t".join(map(str, processed_match))
        qua_str = "\t".join(map(str, processed_quali))
        
        if j < 4:
            output.write(f"{match_str}\t{qua_str}\t")
        else:
            output.write(f"{match_str}\t{qua_str}\n")


def quality_score(in_file, out_file=None):
    """
    Correct quality scores by setting '*' match quality to '!'.
    
    Args:
        in_file: Input file path
        out_file: Output file path (optional)
    
    Returns:
        str: Path to corrected output file
    """
    if out_file is None:
        out_file = os.path.splitext(in_file)[0] + '_quacorrect.txt'
    
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"Input file {in_file} does not exist.")
    
    with open(in_file, 'r') as f, open(out_file, 'w') as out:
        lines = f.readlines()
        out.write(lines[0])
        
        for line in lines[1:]:
            fields = line.strip().split('\t')
            for j in range(5):
                if fields[2 * j] == '*':
                    fields[2 * j + 1] = '!'
            out.write("\t".join(fields) + '\n')
    
    return out_file


def mix_m6a_a_file(m6a_file, a_file, motif, ratio, sample_time=10000):
    """
    Mix m6A and A reads at specified ratio to create training data.
    
    Args:
        m6a_file: Path to m6A single read file
        a_file: Path to A single read file
        motif: DRACH motif string
        ratio: Modification ratio (0 to 1)
        sample_time: Number of samples to generate
    """
    if not os.path.exists('Ave_m6A_model'):
        os.mkdir('Ave_m6A_model')
    if not os.path.exists(f'Ave_m6A_model/{motif}'):
        os.mkdir(f'Ave_m6A_model/{motif}')
    
    output_dir = f'Ave_m6A_model/{motif}'
    output_file = os.path.join(output_dir, f'{ratio}.txt')
    
    if ratio != 1:
        selected_num = int(round(20 / (1 - ratio) * ratio))
        selected_unmod_num = 20
    else:
        selected_num = 20
        selected_unmod_num = 0
    
    m6a_qual_file = quality_score(m6a_file)
    a_qual_file = quality_score(a_file)
    
    with open(m6a_qual_file, 'r') as f1, open(a_qual_file, 'r') as f2, open(output_file, 'w') as output:
        next(f1)
        next(f2)
        lines_mod = f1.readlines()
        lines_unmod = f2.readlines()
        
        output.write("\t".join(header) + "\n")
        
        for _ in range(sample_time):
            try:
                selected_lines_mod = random.sample(lines_mod, selected_num)
                selected_lines_unmod = random.sample(lines_unmod, selected_unmod_num)
                selected_lines = selected_lines_mod + selected_lines_unmod
            except ValueError:
                print("Error: Sample size exceeds available data")
                return
            
            mat_lists, qua_lists = [[] for _ in range(5)], [[] for _ in range(5)]
            
            for sublines in selected_lines:
                fields = sublines.strip("\n").split('\t')
                if fields[0] == "U":
                    continue
                
                for j in range(5):
                    mat_lists[j].append(process_match(fields[j * 2]))
                    qua_lists[j].append(process_quality_score(fields[j * 2 + 1]))
            
            write_processed_data(output, mat_lists, qua_lists)
    
    os.remove(m6a_qual_file)
    os.remove(a_qual_file)


def train_motif_ave_model(ave_dir, m6a_file, a_file, motif):
    """
    Train XGBoost model for a specific motif with hyperparameter tuning.
    
    Args:
        ave_dir: Directory containing mixed ratio files
        m6a_file: Path to m6A single read file
        a_file: Path to A single read file
        motif: DRACH motif string
    """
    # Generate training data at different modification ratios
    mix_m6a_a_file(m6a_file, a_file, motif, ratio=0, sample_time=20000)
    mix_m6a_a_file(m6a_file, a_file, motif, ratio=0.9, sample_time=10000)
    mix_m6a_a_file(m6a_file, a_file, motif, ratio=1, sample_time=10000)
    
    # Load and prepare data
    df_1 = pd.read_csv(f'{ave_dir}/1.txt', sep='\t')
    df_09 = pd.read_csv(f'{ave_dir}/0.9.txt', sep='\t')
    df_0 = pd.read_csv(f'{ave_dir}/0.txt', sep='\t')
    
    df_m6a = pd.concat([df_1, df_09])
    df_m6a['m6A'] = 1
    
    df_a = df_0
    df_a['m6A'] = 0
    
    df_target = pd.concat([df_a, df_m6a])
    X = df_target.drop('m6A', axis=1)
    y = df_target['m6A']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1, 5]
    }
    
    # XGBoost model with grid search
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                              scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    
    print("Best parameters found:", grid_search.best_params_)
    
    # Evaluate model
    best_xgb_model = grid_search.best_estimator_
    y_pred = best_xgb_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(f'Optimized XGBoost Test Accuracy: {accuracy:.2%}')
    
    # Save model
    model_path = os.path.join(ave_dir, f'{motif}_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_xgb_model, f)
    print(f'Model saved to {model_path}')
    
    return accuracy


def train_average_model(reference_fasta, split_folder):
    """
    Main function to train average m6A models for all DRACH motifs.
    
    Args:
        reference_fasta: Path to reference FASTA file containing motif sequences
        split_folder: Path to folder containing single read feature files
    """
    print("=" * 80)
    print("nanoGTR Train Average Model")
    print("=" * 80)
    print(f"Reference FASTA: {reference_fasta}")
    print(f"Split folder: {split_folder}")
    print("=" * 80)
    
    # Validate inputs
    if not os.path.exists(reference_fasta):
        raise FileNotFoundError(f"Reference FASTA file not found: {reference_fasta}")
    
    if not os.path.exists(split_folder):
        raise FileNotFoundError(f"Split folder not found: {split_folder}")
    
    print(f"\nLoading motif dictionary from {reference_fasta}...")
    motif_dict = get_motif_dict(reference_fasta)
    print(f"Loaded {len(motif_dict)} motifs from reference")
    
    print(f"\nSplitting single read files in {split_folder}...")
    get_a_m6a_file_from_single_read(split_folder, motif_dict)
    
    print("\n" + "=" * 80)
    print("Training models for DRACH motifs")
    print("=" * 80)
    
    trained_count = 0
    skipped_count = 0
    
    for motif in drach_list:
        m6a_file = os.path.join(split_folder, f'{motif}_m6a_single_read.txt')
        a_file = os.path.join(split_folder, f'{motif}_a_single_read.txt')
        
        if not os.path.exists(m6a_file) or not os.path.exists(a_file):
            print(f"\nSkipping {motif}: required files not found")
            skipped_count += 1
            continue
        
        ave_dir = f'./Ave_m6A_model/{motif}'
        
        print(f"\n{'*' * 80}")
        print(f"Training model for motif: {motif}")
        print(f"{'*' * 80}")
        
        try:
            accuracy = train_motif_ave_model(ave_dir, m6a_file, a_file, motif)
            print(f'✓ Successfully trained model for motif {motif}, accuracy: {accuracy:.2%}')
            trained_count += 1
        except Exception as e:
            print(f'✗ Error training model for motif {motif}: {str(e)}')
            skipped_count += 1
    
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Total motifs: {len(drach_list)}")
    print(f"Successfully trained: {trained_count}")
    print(f"Skipped: {skipped_count}")
    print("=" * 80)
    print("\nTraining complete!")


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Train average m6A modification detection models for DRACH motifs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  nanogtr train_ave_model -r reference.fa -d /path/to/split_folder/
  nanogtr train_ave_model --reference reference.fa --data_dir /path/to/split_folder/
        """
    )
    
    parser.add_argument(
        '-r', '--reference',
        default='alignment/plus_strand/reference.fa',
        help='Path to reference FASTA file containing motif sequences, (default alignment/plus_strand/reference.fa)'
    )
    
    parser.add_argument(
        '-d', '--data_dir',
        default = './output',
        help='Path to folder containing single read feature files, (default ./output)'
    )
    
    args = parser.parse_args()
    
    try:
        train_average_model(args.reference, args.data_dir)
    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()