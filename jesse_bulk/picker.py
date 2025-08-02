import pathlib

import pandas as pd
from .optuna_reader import read_optuna_study, find_optuna_databases


def filter_and_sort_dna_df(db_path: str, cfg: dict):
    """
    Filter and sort DNA data from Optuna database.
    
    Args:
        db_path: Path to Optuna SQLite database file
        cfg: Configuration dictionary
        
    Returns:
        Filtered and sorted DataFrame
    """
    # Load data from Optuna database
    if not db_path.endswith('.db'):
        # Try to auto-detect Optuna database in project
        db_paths = find_optuna_databases()
        if db_paths:
            print(f"Found Optuna database: {db_paths[0]}")
            db_path = db_paths[0]
        else:
            raise ValueError("No Optuna database found. Please provide path to .db file")
    
    study_name = cfg.get('optuna_study_name', None)
    dna_df = read_optuna_study(db_path, study_name)
    
    # Remove duplicates based on DNA
    dna_df.drop_duplicates(subset=['dna'], inplace=True)

    for metric in cfg['filter_dna']['training'].items():
        key = metric[0]
        min_value = metric[1]['min']
        if min_value and min_value != 'None':
            dna_df.drop(dna_df[dna_df[f'training_log.{key}'] < min_value].index, inplace=True)
        max_value = metric[1]['max']
        if max_value and max_value != 'None':
            dna_df.drop(dna_df[dna_df[f'training_log.{key}'] > max_value].index, inplace=True)

    for metric in cfg['filter_dna']['testing'].items():
        key = metric[0]
        min_value = metric[1]['min']
        if min_value and min_value != 'None':
            dna_df.drop(dna_df[dna_df[f'testing_log.{key}'] < min_value].index, inplace=True)
        max_value = metric[1]['max']
        if max_value and max_value != 'None':
            dna_df.drop(dna_df[dna_df[f'testing_log.{key}'] > max_value].index, inplace=True)

    dna_df.sort_values(by=[cfg['sort_by']], ascending=False, inplace=True)
    
    # Generate output filename
    db_path_obj = pathlib.Path(db_path)
    new_path = db_path_obj.with_suffix('.csv').with_stem(f'{db_path_obj.stem}-picked')
    
    # Save with tab separator (Jesse's format)
    dna_df.to_csv(new_path, header=True, index=False, encoding='utf-8', sep='\t')
    print(f"Saved filtered results to: {new_path}")

    return dna_df
