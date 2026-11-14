#!/usr/bin/env python3
"""
RNA m6A Modification Detection - Model Training Script

This script trains XGBoost models for detecting m6A modifications in RNA sequences
using nanopore sequencing features and motif-based analysis.

Usage:
    python train_m6a_model.py --motif drach --split_folder ./features --reference_file ./reference.fasta
    python train_m6a_model.py --motif all_with_T --split_folder ./features --reference_file ./reference.fasta
    python train_m6a_model.py --motif custom:AAACT,GGACT --split_folder ./features --reference_file ./reference.fasta
"""

import argparse
import pandas as pd
import csv
import pickle
import os
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import List


# ============================================================================
# MOTIF UTILITIES
# ============================================================================

def generate_all_5mer_with_center_A() -> List[str]:
    """
    Generate all possible 5-mer motifs with A at center position.
    Format: NNANN where A is the center nucleotide (4^4 = 256 motifs).
    
    Returns:
        List of 256 5-mer sequences with A at center
    """
    nucleotides = ['A', 'T', 'C', 'G']
    motifs = []
    center = 'A'
    
    for n1 in nucleotides:
        for n2 in nucleotides:
            for n4 in nucleotides:
                for n5 in nucleotides:
                    motifs.append(f"{n1}{n2}{center}{n4}{n5}")
    
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
            - 'all_with_A' or 'all_A': All 256 5-mers with A at center
            - 'drach' or 'DRACH': 18 DRACH motifs for m6A
            
    Returns:
        List of motif sequences
        
    Raises:
        ValueError: If motif_type is invalid
    """
    if motif_type == 'all_with_A' or motif_type == 'all_A':
        return generate_all_5mer_with_center_A()
    elif motif_type == 'drach' or motif_type == 'DRACH':
        return generate_drach_motifs()
    else:
        raise ValueError(
            f"Invalid motif_type: {motif_type}. "
            f"Options: 'all_with_A' (or 'all_A') or 'drach' (or 'DRACH')"
        )


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def get_motif_dict(file_path: str) -> dict:
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


def correct_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct quality scores for matches marked as '*'.
    
    Args:
        df: DataFrame with match and quality columns
        
    Returns:
        DataFrame with corrected quality scores
    """
    for j in range(5):
        mask = df[f'match_{j}'] == '*'
        df.loc[mask, f'qua_{j}'] = '!'
    return df


def process_quality_score(item: str) -> int:
    """
    Convert ASCII quality character to numeric score.
    
    Args:
        item: Quality character
        
    Returns:
        Numeric quality score
    """
    return ord(item) - 33


def process_match(item: str) -> List[int]:
    """
    Process match string into feature vector.
    
    Args:
        item: Match string from alignment
        
    Returns:
        Feature vector [del, del_len, ins, ins_len, ins_site, mis, mis_A, mis_T, mis_C, mis_G]
    """
    if item == '.':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif item == '*':
        return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif item == 'A':
        return [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    elif item == 'T':
        return [0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
    elif item == 'C':
        return [0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
    elif item == 'G':
        return [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    elif re.match(r'^-\d+$', item):
        num = int(item[1:])
        return [1, num, 0, 0, 0, 0, 0, 0, 0, 0]
    elif re.match(r'^\+\d+$', item):
        num = int(item[1:])
        return [0, 0, 1, num, 0, 0, 0, 0, 0, 0]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def process_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process match and quality columns in DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Processed DataFrame
    """
    for i in range(5):
        df[f'match_{i}'] = df[f'match_{i}'].apply(process_match)
        df[f'qua_{i}'] = df[f'qua_{i}'].apply(process_quality_score)
    return df


def merge_match_list_of_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand match feature lists into separate columns.
    
    Args:
        df: DataFrame with match lists
        
    Returns:
        DataFrame with expanded match features
    """
    for i in range(5):
        df_expanded = pd.DataFrame(
            df[f'match_{i}'].tolist(),
            columns=[
                f'del_{i}', f'del_len_{i}', f'ins_{i}', f'ins_len_{i}', f'ins_site_{i}',
                f'mis_{i}', f'mis_A_{i}', f'mis_T_{i}', f'mis_C_{i}', f'mis_G_{i}'
            ]
        )
        df = df.drop(columns=[f'match_{i}']).join(df_expanded)
    return df


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def fitting_xgboost(X: pd.DataFrame, y: pd.Series, motif: str):
    """
    Train XGBoost model with hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target labels
        motif: Motif identifier for logging
        
    Returns:
        Trained XGBoost model
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define base classifier
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42
    )
    
    # Define parameter grid
    param_grid = {
        'max_depth': [4, 5, 6],
        'min_child_weight': [1],
        'gamma': [0],
        'subsample': [1.0],
        'colsample_bytree': [1.0],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1]
    }
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Print results
    print(f"Best Parameters for motif {motif}:", grid_search.best_params_)
    print(f"Best ROC AUC Score (CV) for motif {motif}:", grid_search.best_score_)
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    print(f"Test Accuracy for motif {motif}:", accuracy_score(y_test, y_pred))
    print(f"Test ROC AUC for motif {motif}:", roc_auc_score(y_test, y_pred_proba))
    
    return best_model


def train_single_read_model_rna002(motif: str, split_folder: str, motif_dict: dict, 
                                   model_save_path: str = './single_read_model'):
    """
    Train a single-read XGBoost model for m6A detection.
    
    Parameters:
    -----------
    motif : str
        Motif identifier for training
    split_folder : str
        Path to folder containing feature files
    motif_dict : dict
        Dictionary mapping motifs to position names
    model_save_path : str
        Path to save trained model (default: './single_read_model')
    """
    # Construct feature file path
    feature_file = f'{split_folder}/{motif}_single_read.txt'
    
    if not os.path.exists(feature_file):
        print(f"Warning: Feature file not found: {feature_file}")
        print(f"Skipping motif {motif}")
        return None
    
    # Load and preprocess feature data
    print(f"\nProcessing motif: {motif}")
    print(f"Loading features from: {feature_file}")
    
    df_feature = pd.read_csv(feature_file, sep='\t', quoting=csv.QUOTE_NONE)
    df_feature = correct_quality(df_feature)
    df_feature = process_input_df(df_feature)
    df_feature = merge_match_list_of_df(df_feature)
    
    # Parse motif information
    if motif not in motif_dict:
        print(f"Warning: Motif {motif} not found in motif_dict. Skipping.")
        return None
    
    A_name = motif_dict[motif]
    num = int(A_name.split("_")[2])
    strand = A_name.split("_")[1]
    m6A_name = f'm6A_{strand}_F'
    A_name = f'A_{strand}_F'
    pos_num = 28 + 15 * num
    
    # Separate modified and unmodified samples
    df_A = df_feature[(df_feature['chr_2'] == A_name) & 
                      (df_feature['pos_2'] == pos_num)].copy()
    df_m6A = df_feature[(df_feature['chr_2'] == m6A_name) & 
                        (df_feature['pos_2'] == pos_num)].copy()
    
    # Add labels BEFORE sampling and concatenation
    df_A['m6A'] = 0
    df_m6A['m6A'] = 1
    
    # Balance classes by downsampling to minority class size
    sample_size = min(len(df_A), len(df_m6A))
    
    if sample_size == 0:
        print(f"Warning: No samples found for motif {motif}. Check your filtering criteria.")
        return None
    
    df_A = df_A.sample(n=sample_size, random_state=42)
    df_m6A = df_m6A.sample(n=sample_size, random_state=42)
    drop_cols = ['chr_0','pos_0','chr_1','pos_1','chr_2','pos_2','chr_3','pos_3','chr_4','pos_4','read_name']
    # Combine datasets and prepare features
    df_target = pd.concat([df_A, df_m6A], ignore_index=True)
    df_target = df_target.drop(columns= drop_cols)
    
    # Split features and target
    X = df_target.drop(columns='m6A')
    y = df_target['m6A']
    
    print(f"Training on {len(X)} samples ({sample_size} per class)")
    
    # Train model
    model = fitting_xgboost(X, y, motif)
    
    # Save model
    os.makedirs(model_save_path, exist_ok=True)
    model_path = f"{model_save_path}/{motif}_single_read.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f'Model for motif {motif} saved to {model_path}')
    
    return model


def train_drach_model_rna002(motif_list: List[str], split_folder: str, 
                             reference_file: str, model_save_path: str = './single_read_model'):
    """
    Train models for a list of motifs.
    
    Args:
        motif_list: List of motifs to train
        split_folder: Path to folder containing feature files
        reference_file: Path to FASTA reference file
        model_save_path: Path to save trained models
    """
    print(f"\n{'='*80}")
    print(f"Training m6A Detection Models")
    print(f"{'='*80}")
    print(f"Number of motifs: {len(motif_list)}")
    print(f"Split folder: {split_folder}")
    print(f"Reference file: {reference_file}")
    print(f"Model save path: {model_save_path}")
    print(f"{'='*80}\n")
    
    # Load motif dictionary
    print("Loading motif dictionary from reference file...")
    motif_dict = get_motif_dict(reference_file)
    print(f"Loaded {len(motif_dict)} motifs from reference\n")
    
    # Train models for each motif
    successful = 0
    failed = 0
    
    for idx, motif in enumerate(motif_list, 1):
        print(f"\n[{idx}/{len(motif_list)}] Training model for motif: {motif}")
        print("-" * 80)
        
        try:
            model = train_single_read_model_rna002(motif, split_folder, motif_dict, model_save_path)
            if model is not None:
                successful += 1
                print(f"✓ Successfully trained model for motif {motif}")
            else:
                failed += 1
                print(f"✗ Failed to train model for motif {motif}")
        except Exception as e:
            failed += 1
            print(f"✗ Error training model for motif {motif}: {str(e)}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Training Summary")
    print(f"{'='*80}")
    print(f"Total motifs: {len(motif_list)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Models saved to: {model_save_path}")
    print(f"{'='*80}\n")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to handle command-line arguments and execute training."""
    parser = argparse.ArgumentParser(
        description='Train XGBoost models for m6A modification detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on DRACH motifs (18 motifs)
  python train_m6a_model.py --motif drach --split_folder ./features --reference_file ./ref.fasta
  
  # Train on all 5-mers with A at center (256 motifs)
  python train_m6a_model.py --motif all_with_A --split_folder ./features --reference_file ./ref.fasta
  
  # Specify output directory
  python train_m6a_model.py --motif drach --split_folder ./features --reference_file ./ref.fasta --output ./models
        """
    )
    
    parser.add_argument(
        '--motif',
        type=str,
        default='drach',
        help="Motif type: 'drach' (18 DRACH motifs) or 'all_with_A' (256 motifs with A at center), default drach"
    )
    
    parser.add_argument(
        '--split_folder',
        type=str,
        default='./output',
        help='Path to folder containing feature files (e.g., ./features), default: ./output'
    )
    
    parser.add_argument(
        '--reference_file',
        type=str,
        default='alignment/plus_strand/reference.fa',
        help='Path to FASTA reference file (e.g., ./reference.fasta)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./single_read_model_rna002',
        help='Output directory for trained models (default: ./single_read_model)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.split_folder):
        print(f"Error: Split folder not found: {args.split_folder}")
        return 1
    
    if not os.path.exists(args.reference_file):
        print(f"Error: Reference file not found: {args.reference_file}")
        return 1
    
    # Get motif list
    try:
        motif_list = get_motif_list(args.motif)
        print(f"\nGenerated {len(motif_list)} motifs from type: {args.motif}")
        if len(motif_list) <= 20:
            print(f"Motifs: {', '.join(motif_list)}")
        else:
            print(f"First 10 motifs: {', '.join(motif_list[:10])}")
            print(f"Last 10 motifs: {', '.join(motif_list[-10:])}")
    except ValueError as e:
        print(f"Error: {str(e)}")
        return 1
    
    # Train models
    train_drach_model_rna002(
        motif_list=motif_list,
        split_folder=args.split_folder,
        reference_file=args.reference_file,
        model_save_path=args.output
    )
    
    return 0


if __name__ == "__main__":
    exit(main())