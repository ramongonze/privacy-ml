"""
Calculate the accuracy of trained attack models against their corresponding utility models.

This script reads the results from the unified training pipeline, identifies
all trained attack models, finds their corresponding utility models, and runs
the model inversion evaluation. 

It ensures consistent preprocessing by either loading a saved DataManager object 
from the training run or by recreating it on the fly using the exact same code,
imported directly from the training script.
"""
import os
import re
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import traceback

# --- KEY CHANGE: Import classes directly from the training script ---
# This assumes 'unified_train.py' is in the same directory or accessible in the Python path.
try:
    from unified_train import Config, DataManager
except ImportError:
    print("FATAL: Could not import 'Config' and 'DataManager' from 'unified_train.py'.")
    print("Please ensure 'unified_train.py' is in the same directory as this script.")
    exit()

# Import the optimized evaluation function
from privacy_evaluation import model_inversion_acc


def parse_attack_model_path(path_str: str) -> dict:
    """Extracts metadata from an attack model's file path."""
    # Example: on_random_forest_adult_data_eps_1.0_with_naive_bayes.pkl
    pattern = re.compile(
        r"on_(?P<utility_model>.*?)_"
        r"(?P<dataset>adult|hospitals)_"
        r"(?P<dp_type>none|data|algorithm|output)_eps_"
        r"(?P<epsilon>[\d\.]+|None)_with_"
        r"(?P<attack_model>.*)\.pkl"
    )
    match = pattern.search(os.path.basename(path_str))
    if not match:
        raise ValueError(f"Could not parse attack model path: {path_str}")
    
    data = match.groupdict()
    if data['epsilon'] == 'None':
        data['epsilon'] = None
    else:
        data['epsilon'] = float(data['epsilon'])
        
    return data

def main():
    config = Config()
    print("--- STARTING ATTACK EVALUATION PIPELINE ---")
    
    # --- 1. Load or Recreate DataManager ---
    data_manager = None
    dm_path = Path(config.EXPERIMENT_DIR) / "data_manager.pkl"
    if dm_path.exists():
        print(f"Loading pre-fitted DataManager from: {dm_path}")
        with open(dm_path, 'rb') as f:
            data_manager = pickle.load(f)
    else:
        print(f"WARNING: DataManager not found at '{dm_path}'.")
        print("Recreating it from scratch to ensure consistency with the training run.")
        try:
            data_manager = DataManager(config)
            data_manager.fit_preprocessors() # This re-runs the fitting logic
            print("✅ DataManager successfully recreated.")
        except Exception as e:
            print(f"FATAL: Failed to recreate DataManager. Error: {e}")
            traceback.print_exc()
            return

    # --- 2. Load Training and Original Data ---
    if not config.RESULTS_CSV_PATH.exists():
        print(f"ERROR: Training results not found at '{config.RESULTS_CSV_PATH}'. Please run the training script first.")
        return

    print(f"Loading training results from: {config.RESULTS_CSV_PATH}")
    results_df = pd.read_csv(config.RESULTS_CSV_PATH)

    attack_models_df = results_df[results_df['study_id'].str.startswith('attack_', na=False)].copy()
    attack_models_df.dropna(subset=['model_path'], inplace=True)
    print(f"Found {len(attack_models_df)} trained attack models to evaluate.")
    
    print("Loading original datasets for evaluation...")
    original_data = {}
    qids_map = {
        "adult": ["age", "workclass", "occupation", "sex", "education", "native-country", "marital-status"],
        "hospitals": ["TYPE_OF_ADMISSION", "PAT_ZIP", "PAT_COUNTY", "PAT_STATUS", "SEX_CODE", "RACE", "ADMIT_WEEKDAY", "PAT_AGE"]
    }
    for ds_name in config.UTILITY_TARGETS.keys():
        path = Path(config.DATA_DIR) / ds_name / f"{ds_name}_pp.csv"
        df = pd.read_csv(path)
        if ds_name == "hospitals":
            print("Subsampling hospitals dataset for evaluation (50k samples).")
            df = df.sample(n=50000, random_state=42)
        original_data[ds_name] = df

    # --- 3. Run Evaluation Loop ---
    evaluation_records = []
    output_path = Path(config.EXPERIMENT_DIR) / "attack_evaluation_results.csv"
    
    for _, row in tqdm(attack_models_df.iterrows(), total=len(attack_models_df), desc="Evaluating Attack Models"):
        try:
            with open(row['model_path'], 'rb') as f:
                attack_model = pickle.load(f)
            
            attack_info = parse_attack_model_path(row['model_path'])
            dataset = attack_info['dataset']
            utility_model_type = attack_info['utility_model']
            
            dp_type_for_M = attack_info['dp_type'] if attack_info['dp_type'] != 'output' else 'none'
            epsilon_for_M = attack_info['epsilon'] if attack_info['dp_type'] != 'output' else None
            eps_str_for_M = "None" if epsilon_for_M is None else str(epsilon_for_M)
            
            utility_model_path = Path(config.EXPERIMENT_DIR) / "models" / utility_model_type / f"{dataset}_{dp_type_for_M}_eps_{eps_str_for_M}.pkl"
            
            if not utility_model_path.exists():
                print(f"WARNING: Skipping attack. Could not find corresponding utility model at: {utility_model_path}")
                continue

            with open(utility_model_path, 'rb') as f:
                utility_model = pickle.load(f)

            accuracy = model_inversion_acc(
                data_ori=original_data[dataset],
                qids=qids_map[dataset],
                target_col=config.UTILITY_TARGETS[dataset],
                sensitive_col=config.SENSITIVE_ATTRIBUTES[dataset],
                utility_model=utility_model,
                attack_model=attack_model,
                data_manager=data_manager,
                dataset_name=dataset,
                dp_output=(attack_info['dp_type'] == 'output'),
                epsilon=attack_info['epsilon']
            )
            
            evaluation_records.append({
                "dataset": dataset,
                "dp_type": attack_info['dp_type'],
                "epsilon": attack_info['epsilon'],
                "utility_model": utility_model_type,
                "attack_model": attack_info['attack_model'],
                "attack_accuracy": accuracy
            })

        except Exception as e:
            print(f"\nERROR processing model {row['model_path']}: {e}")
            traceback.print_exc()

    # --- 4. Save Results ---
    if not evaluation_records:
        print("No evaluations were successfully completed.")
        return
        
    eval_df = pd.DataFrame(evaluation_records)
    eval_df.sort_values(by=["dataset", "utility_model", "attack_model", "dp_type", "epsilon"], inplace=True)
    eval_df.to_csv(output_path, index=False)
    
    print("\n--- ATTACK EVALUATION COMPLETE ---")
    print(f"Results saved to: {output_path}")
    print(eval_df.head())

if __name__ == "__main__":
    main()