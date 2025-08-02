import json
import base64
import sqlite3
import pandas as pd
import optuna
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_optuna_study(db_path: str, study_name: Optional[str] = None) -> pd.DataFrame:
    """
    Read optimization results from an Optuna study database.
    
    Args:
        db_path: Path to the SQLite database file
        study_name: Name of the study to load. If None, uses the first study found.
        
    Returns:
        DataFrame with trial results
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    # Create storage URL
    storage_url = f"sqlite:///{db_path}"
    
    # If no study name provided, get the first available study
    if study_name is None:
        studies = optuna.study.get_all_study_names(storage_url)
        if not studies:
            raise ValueError("No studies found in the database")
        if len(studies) == 1:
            study_name = studies[0]
            print(f"Found one study: {study_name}")
        else:
            # Use the most recent study (last in the list)
            study_name = studies[-1]
            print(f"Multiple studies found. Using most recent: {study_name}")
            print(f"Available studies: {studies}")
    
    # Load the study
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    # Extract trial data
    trials_data = []
    for trial in study.get_trials():
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
            
        trial_data = {
            'trial_number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'dna': _params_to_dna(trial.params),
            'state': trial.state.name,
        }
        
        # Add user attributes (training and testing metrics)
        if trial.user_attrs:
            if 'training_metrics' in trial.user_attrs:
                for key, val in trial.user_attrs['training_metrics'].items():
                    trial_data[f'training_log.{key}'] = val
            if 'testing_metrics' in trial.user_attrs:
                for key, val in trial.user_attrs['testing_metrics'].items():
                    trial_data[f'testing_log.{key}'] = val
        
        trials_data.append(trial_data)
    
    return pd.DataFrame(trials_data)


def _params_to_dna(params: Dict) -> str:
    """Convert hyperparameters dict to DNA string (base64 encoded)."""
    params_str = json.dumps(params, sort_keys=True)
    return base64.b64encode(params_str.encode()).decode()


def _dna_to_params(dna: str) -> Dict:
    """Convert DNA string (base64 encoded) to hyperparameters dict."""
    params_str = base64.b64decode(dna.encode()).decode()
    return json.loads(params_str)


def find_optuna_databases(project_path: str = '.') -> List[str]:
    """
    Find all Optuna database files in the Jesse project.
    
    Args:
        project_path: Path to Jesse project root
        
    Returns:
        List of database file paths
    """
    db_paths = []
    
    # Check common locations
    possible_paths = [
        Path(project_path) / 'storage' / 'temp' / 'optuna' / 'optuna_study.db',
        Path(project_path) / 'storage' / 'optuna_study.db',
        Path(project_path) / 'optuna_study.db'
    ]
    
    for path in possible_paths:
        if path.exists():
            db_paths.append(str(path))
    
    # Also search recursively
    for db_file in Path(project_path).rglob('*.db'):
        if 'optuna' in db_file.name.lower() and str(db_file) not in db_paths:
            db_paths.append(str(db_file))
    
    return db_paths