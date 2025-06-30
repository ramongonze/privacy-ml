#!/usr/bin/env python3
"""
Unified and Resumable Differential Privacy Experiment Pipeline

This single script orchestrates a comprehensive machine learning experiment 
in a linear, fault-tolerant, and resumable manner, adhering to a specific
project file structure.

Execution Flow:
1.  For each dataset ('adult', then 'hospitals'):
    a. Generate or load a single, shared synthetic dataset for this domain.
2.    For each utility model type ('random_forest', 'logistic_regression', 'naive_bayes'):
3.      a. Train Utility Models:
           - No DP (baseline)
           - Data DP (for each epsilon)
           - Algorithm DP (for each epsilon)
           - Models and results are saved incrementally to 'incremental_results.csv'.
      b. Generate Attack Datasets:
           - The shared synthetic dataset from step (1a) is used.
           - All utility models from step (3a) are queried.
           - Output DP (k-RR) is applied to the baseline model's predictions.
           - All resulting query datasets (features + predictions) are saved.
      c. Train Attack Models:
           - For each attack dataset from step (3b):
           - Train all three attack model types using a dedicated attack preprocessor.
           - Models and results are saved incrementally.

Features:
- Single-File Orchestration: All logic is contained herein.
- Domain-Based Preprocessing: Separate, dedicated preprocessors for utility and attack tasks.
- Centralized Synthetic Data: A single synthetic dataset is used per domain for fair comparison.
- Incremental & Resumable: Reads 'incremental_results.csv' to skip completed work.
- Strict File Structure: All outputs (models, datasets, results) are saved to the specified paths.
- Linear Execution: Experiments run sequentially, with individual models using all available cores.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm

# Scikit-learn and related imports
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Differential Privacy libraries
from diffprivlib.models import RandomForestClassifier as DPRandomForestClassifier
from diffprivlib.models import GaussianNB as DPGaussianNB
from diffprivlib.models import LogisticRegression as DPLogisticRegression

# Hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# Setup
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- UTILITY AND HELPER FUNCTIONS ---

def krr(original_value: Any, domain: List[Any], epsilon: float) -> Any:
    """k-Randomized Response mechanism."""
    k = len(domain)
    if k <= 1: return original_value
    p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    
    if np.random.random() < p:
        return original_value
    else:
        other_options = [v for v in domain if v != original_value]
        return np.random.choice(other_options)

@dataclass
class Config:
    """Configuration for the entire experimental pipeline."""
    # --- Directories ---
    DATA_DIR: str = "data"
    DP_DATA_DIR: str = "results"
    EXPERIMENT_DIR: str = "experiment_results"
    
    # --- Experiment Settings ---
    DATASETS: List[str] = field(default_factory=lambda: ["adult", "hospitals"])
    UTILITY_MODELS: List[str] = field(default_factory=lambda: ['naive_bayes', 'logistic_regression', 'random_forest'])
    ATTACK_MODELS: List[str] = field(default_factory=lambda: ['naive_bayes', 'logistic_regression', 'random_forest'])
    EPSILON_VALUES: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 10.0, 50.0, 100.0])
    
    # --- Optuna Settings ---
    N_TRIALS: int = 20
    CV_FOLDS: int = 5
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # --- Dataset Specifics ---
    UTILITY_TARGETS: Dict[str, str] = field(default_factory=lambda: {"adult": "income", "hospitals": "TOTAL_CHARGES"})
    SENSITIVE_ATTRIBUTES: Dict[str, str] = field(default_factory=lambda: {"adult": "race", "hospitals": "PRINC_DIAG_CODE"})
    
    # --- Attack Dataset Generation ---
    ADULT_SAMPLES: int = 20000
    HOSPITALS_SAMPLES: int = 100000

    @property
    def RESULTS_CSV_PATH(self) -> Path:
        return Path(self.EXPERIMENT_DIR) / "incremental_results.csv"

class DataManager:
    """Centralized data handler with dedicated preprocessors for utility and attack tasks."""
    def __init__(self, config: Config):
        self.config = config
        self.domains: Dict[str, Dict[str, Any]] = self._load_domains()
        self.utility_preprocessors: Dict[str, ColumnTransformer] = {}
        self.attack_preprocessors: Dict[str, ColumnTransformer] = {}
        self.utility_label_encoders: Dict[str, LabelEncoder] = {}
        self.sensitive_label_encoders: Dict[str, LabelEncoder] = {}

    def _load_domains(self) -> Dict[str, Dict[str, Any]]:
        """Loads all feature and target domains from hardcoded values and files."""
        domains = {
            'adult': {
                'age': (17, 90), 'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 'Self-emp-inc', 'Without-pay'],
                'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Craft-repair', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'],
                'race': ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], 'sex': ['Male', 'Female'],
                'education': ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school', '5th-6th', '10th', 'Preschool', '12th', '1st-4th'],
                'native-country': ['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands'],
                'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'],
                'income': ['<=50K', '>50K']
            },
            'hospitals': {
                'ADMIT_WEEKDAY': (1, 7), 'TYPE_OF_ADMISSION': [1, 2, 3, 4, 5, 9], 'SEX_CODE': ['M', 'F', 'U'], 'PAT_AGE': (0, 26),
                'PAT_ZIP': [(700, 800), (75000, 80000)], 'RACE': (1, 5),
                'TOTAL_CHARGES': ['[0,1000)', '[1000,2000)', '[2000,3000)', '[3000,4000)', '[4000,5000)', '[5000,6000)', '[6000,7000)', '[7000,8000)', '[8000,9000)', '[9000,10000)', '[10000,15000)', '[15000,20000)', '[20000,25000)', '[25000,30000)', '[30000,40000)', '[40000,50000)', '[50000,100000)', '[100000,200000)', '[200000,500000)', '500000+']
            }
        }
        for col in ['PRINC_DIAG_CODE', 'PAT_STATUS', 'PAT_COUNTY']:
            path = Path(self.config.DATA_DIR) / "hospitals" / f"{col}.csv"
            if path.exists():
                domains['hospitals'][col] = pd.read_csv(path)[col].unique().tolist()
        return domains

    def fit_preprocessors(self):
        """Fits dedicated preprocessors for utility and attack tasks based on full domains."""
        logger.info("Fitting dedicated preprocessors for utility and attack tasks...")
        for ds_name in self.config.DATASETS:
            utility_target = self.config.UTILITY_TARGETS[ds_name]
            sensitive_attribute = self.config.SENSITIVE_ATTRIBUTES[ds_name]
            original_path = Path(self.config.DATA_DIR) / ds_name / f"{ds_name}_pp.csv"
            df_sample = pd.read_csv(original_path)

            # --- 1. Fit UTILITY Preprocessor (predicts utility_target) ---
            X_utility = df_sample.drop(columns=[utility_target])
            cat_feat_util = X_utility.select_dtypes(include=['object']).columns.tolist()
            num_feat_util = X_utility.select_dtypes(include=['int64', 'float64']).columns.tolist()
            ohe_cat_util = [self.domains[ds_name][col] for col in cat_feat_util]
            
            utility_preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_feat_util),
                    ('cat', OneHotEncoder(categories=ohe_cat_util, handle_unknown='ignore', sparse_output=False), cat_feat_util)
                ], remainder='passthrough'
            ).fit(X_utility)
            self.utility_preprocessors[ds_name] = utility_preprocessor
            logger.info(f"✅ Fitted utility preprocessor for {ds_name}.")

            # --- 2. Fit ATTACK Preprocessor (predicts sensitive_attribute) ---
            X_attack = df_sample.drop(columns=[sensitive_attribute])
            # The utility target is now a feature for the attacker
            cat_feat_attack = X_attack.select_dtypes(include=['object']).columns.tolist()
            num_feat_attack = X_attack.select_dtypes(include=['int64', 'float64']).columns.tolist()
            ohe_cat_attack = [self.domains[ds_name][col] for col in cat_feat_attack]

            attack_preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_feat_attack),
                    ('cat', OneHotEncoder(categories=ohe_cat_attack, handle_unknown='ignore', sparse_output=False), cat_feat_attack)
                ], remainder='passthrough'
            ).fit(X_attack)
            self.attack_preprocessors[ds_name] = attack_preprocessor
            logger.info(f"✅ Fitted attack preprocessor for {ds_name}.")

            # --- 3. Fit LabelEncoders on their full domains ---
            self.utility_label_encoders[ds_name] = LabelEncoder().fit(self.domains[ds_name][utility_target])
            self.sensitive_label_encoders[ds_name] = LabelEncoder().fit(self.domains[ds_name][sensitive_attribute])
        
        logger.info("✅ All preprocessors and encoders fitted.")

    def get_synthetic_data(self, dataset_name: str) -> pd.DataFrame:
        """Generates a synthetic dataset by sampling from known domains."""
        logger.info(f"Generating synthetic data for {dataset_name} by sampling from domains...")
        n_samples = self.config.ADULT_SAMPLES if dataset_name == 'adult' else self.config.HOSPITALS_SAMPLES
        
        synth_dict = {}
        domain = self.domains[dataset_name]
        
        # We sample based on the features the utility model expects
        feature_cols = list(self.utility_preprocessors[dataset_name].feature_names_in_)
        
        for col in feature_cols:
            col_domain = domain[col]

            if dataset_name == 'adult' and col == 'native-country':
                countries = col_domain
                try:
                    us_index = countries.index('United-States')
                    num_countries = len(countries)
                    # 6/7 for US, 1/7 for the rest, distributed equally
                    p = [(1/7) / (num_countries - 1)] * num_countries
                    p[us_index] = 6/7
                    p = np.array(p) / np.sum(p) # Normalize
                    synth_dict[col] = np.random.choice(countries, size=n_samples, p=p)
                except (ValueError, IndexError): # Fallback if 'United-States' not in domain
                    synth_dict[col] = np.random.choice(countries, size=n_samples)
                continue

            if isinstance(col_domain, list): # Categorical
                synth_dict[col] = np.random.choice(col_domain, size=n_samples)
            elif isinstance(col_domain, tuple) and len(col_domain) == 2: # Integer range
                if isinstance(col_domain[0], int): # Simple range
                     synth_dict[col] = np.random.randint(col_domain[0], col_domain[1] + 1, size=n_samples)
                else: # Special case for PAT_ZIP with two ranges
                    synth_dict[col] = np.random.randint(col_domain[1][0], col_domain[1][1] + 1, size=n_samples)
        
        return pd.DataFrame(synth_dict)

    def load_for_utility(self, dataset_name: str, dp_type: str, epsilon: Optional[float]) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads data for utility model training."""
        if dp_type == 'data':
            eps_str = str(int(epsilon)) if epsilon == int(epsilon) else str(epsilon)
            path = Path(self.config.DP_DATA_DIR) / f"{dataset_name}_san_{eps_str}.csv"
        else:
            path = Path(self.config.DATA_DIR) / dataset_name / f"{dataset_name}_pp.csv"
        df = pd.read_csv(path)
        y = df[self.config.UTILITY_TARGETS[dataset_name]]
        X = df.drop(columns=[self.config.UTILITY_TARGETS[dataset_name]])
        return X, y

    def load_for_attack(self, attack_dataset_path: Path) -> Tuple[str, pd.DataFrame, pd.Series]:
        """Loads data for attack model training."""
        df = pd.read_csv(attack_dataset_path)
        dataset_name = "adult" if "income" in df.columns else "hospitals"
        sensitive_col = self.config.SENSITIVE_ATTRIBUTES[dataset_name]
        y = df[sensitive_col]
        # X for the attack model includes the utility target as a feature
        X = df.drop(columns=[sensitive_col])
        return dataset_name, X, y

    def preprocess_data(self, dataset_name: str, X: pd.DataFrame, y: pd.Series, task: str) -> Tuple[np.ndarray, np.ndarray]:
        """Applies the appropriate already-fitted preprocessor based on the task."""
        if task == 'utility':
            preprocessor = self.utility_preprocessors[dataset_name]
            y_processed = self.utility_label_encoders[dataset_name].transform(y)
        elif task == 'attack':
            preprocessor = self.attack_preprocessors[dataset_name]
            y_processed = self.sensitive_label_encoders[dataset_name].transform(y)
        else:
            raise ValueError(f"Unknown preprocessing task: {task}. Must be 'utility' or 'attack'.")
        
        X_processed = preprocessor.transform(X)
        return X_processed, y_processed

class IncrementalSaver:
    """Handles saving results and checking for completed experiments based on the specified file structure."""
    def __init__(self, config: Config):
        self.config = config
        self.results_path = config.RESULTS_CSV_PATH
        self._initialize_file()
        self.completed_model_paths = self._load_completed()

    def _initialize_file(self):
        if not self.results_path.exists():
            headers = [
                'study_id', 'model_type', 'dataset', 'dp_type', 'epsilon', 'worker_id',
                'experiment_idx', 'best_cv_score', 'final_test_score', 'best_params',
                'model_path', 'timestamp', 'status'
            ]
            pd.DataFrame(columns=headers).to_csv(self.results_path, index=False)

    def _load_completed(self) -> set:
        """Loads completed experiments by checking for non-null model_path in the results file."""
        if self.results_path.exists():
            try:
                df = pd.read_csv(self.results_path)
                return set(df[df['model_path'].notna()]['model_path'])
            except (pd.errors.EmptyDataError, KeyError):
                return set()
        return set()

    def is_completed(self, model_path: Path) -> bool:
        return str(model_path) in self.completed_model_paths

    def save_result(self, result_dict: Dict):
        if self.is_completed(Path(result_dict.get("model_path", ""))):
            return
        all_cols = [
            'study_id', 'model_type', 'dataset', 'dp_type', 'epsilon', 'worker_id',
            'experiment_idx', 'best_cv_score', 'final_test_score', 'best_params',
            'model_path', 'timestamp', 'status'
        ]
        for col in all_cols:
            result_dict.setdefault(col, None)
        pd.DataFrame([result_dict])[all_cols].to_csv(self.results_path, mode='a', header=False, index=False)
        if result_dict.get("model_path"):
            self.completed_model_paths.add(result_dict["model_path"])
        logger.info(f"💾 Saved result for model: {result_dict.get('model_path')}")

class ModelFactory:
    """Creates model instances for both utility and attack tasks."""
    def __init__(self, config: Config):
        self.config = config
    def create_model(self, model_type: str, hyperparams: Dict, dp_type: str = 'none', epsilon: float = 1.0) -> Any:
        if dp_type == 'algorithm':
            if model_type == 'random_forest': return DPRandomForestClassifier(epsilon=epsilon, random_state=self.config.RANDOM_STATE, **hyperparams, n_jobs=-1)
            elif model_type == 'naive_bayes': return DPGaussianNB(epsilon=epsilon, **hyperparams)
            elif model_type == 'logistic_regression': return DPLogisticRegression(epsilon=epsilon, random_state=self.config.RANDOM_STATE, **hyperparams)
        else:
            if model_type == 'random_forest': return RandomForestClassifier(random_state=self.config.RANDOM_STATE, **hyperparams, n_jobs=-1)
            elif model_type == 'naive_bayes': return GaussianNB(**hyperparams)
            elif model_type == 'logistic_regression': return LogisticRegression(random_state=self.config.RANDOM_STATE, **hyperparams, n_jobs=-1)
        raise ValueError(f"Unknown model/dp_type combination: {model_type}/{dp_type}")
    def sample_hyperparams(self, trial: optuna.Trial, model_type: str, dp_type: str) -> Dict:
        """
        Suggests a comprehensive set of hyperparameters for a given model.
        Handles dependent hyperparameters correctly by pruning invalid trials.
        """
        if model_type == 'random_forest':
            # DP-RandomForest has a more restricted set of tunable parameters
            if dp_type == 'algorithm':
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 25)
                }
            # Standard RandomForest
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
            }

        elif model_type == 'logistic_regression':
            # DP-LogisticRegression from diffprivlib has no standard tunable hyperparameters
            if dp_type == 'algorithm':
                return {}

            # --- CORRECTED LOGIC FOR DEPENDENT HYPERPARAMETERS ---
            
            # 1. Define the full search space for all parameters
            solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
            
            # 2. Check for invalid combinations and prune the trial if found
            if solver == 'liblinear' and penalty == 'elasticnet':
                # This combination is not supported by sklearn
                raise optuna.exceptions.TrialPruned()
            
            if penalty == 'elasticnet' and solver != 'saga':
                # Elasticnet is only supported by the 'saga' solver
                raise optuna.exceptions.TrialPruned()

            # 3. Build the parameter dictionary
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e4, log=True),
                'solver': solver,
                'penalty': penalty,
                'max_iter': trial.suggest_int('max_iter', 200, 2000)
            }
            
            # 4. Conditionally add 'l1_ratio' only if needed
            if penalty == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
            
            return params

        elif model_type == 'naive_bayes':
            # DP-GaussianNB from diffprivlib has no standard tunable hyperparameters
            if dp_type == 'algorithm':
                return {}
            
            # Standard GaussianNB - only var_smoothing is typically tuned
            return {
                'var_smoothing': trial.suggest_float('var_smoothing', 1e-10, 1e-1, log=True)
            }
            
        return {}


def run_training_and_evaluation(
    config: Config,
    model_factory: ModelFactory,
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dp_type: str = 'none',
    epsilon: float = None
) -> Tuple[Any, float, float, Dict]:
    """Runs Optuna, trains final model, and evaluates. Returns model, CV score, final score, and params."""
    def objective(trial: optuna.Trial) -> float:
        hyperparams = model_factory.sample_hyperparams(trial, model_type, dp_type)
        model = model_factory.create_model(model_type, hyperparams, dp_type, epsilon)
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_val_fold)
            scores.append(accuracy_score(y_val_fold, preds))
        return np.mean(scores)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=config.RANDOM_STATE))
    study.optimize(objective, n_trials=config.N_TRIALS)
    best_params = study.best_params
    best_cv_score = study.best_value
    final_model = model_factory.create_model(model_type, best_params, dp_type, epsilon)
    final_model.fit(X_train, y_train)
    final_preds = final_model.predict(X_test)
    final_score = accuracy_score(y_test, final_preds)
    return final_model, best_cv_score, final_score, best_params

# --- MAIN ORCHESTRATION SCRIPT ---

def main():
    config = Config()
    saver = IncrementalSaver(config)
    data_manager = DataManager(config)
    model_factory = ModelFactory(config)
    
    # This crucial step fits dedicated preprocessors for utility AND attack tasks
    data_manager.fit_preprocessors()
    
    logger.info("="*80)
    logger.info("STARTING UNIFIED DIFFERENTIAL PRIVACY EXPERIMENT PIPELINE (V3)")
    logger.info(f"Found {len(saver.completed_model_paths)} previously completed models. They will be skipped.")
    logger.info("="*80)

    for dataset_name in config.DATASETS:
        logger.info(f"\n{'='*30} PROCESSING DATASET: {dataset_name.upper()} {'='*30}")
        
        # --- CENTRALIZED SYNTHETIC DATASET GENERATION ---
        synth_base_path = Path(config.EXPERIMENT_DIR) / "datasets" / f"synthetic_base_{dataset_name}.csv"
        synth_base_path.parent.mkdir(parents=True, exist_ok=True)
        if synth_base_path.exists():
            logger.info(f"Loading existing shared synthetic dataset: {synth_base_path.name}")
            synth_data_base = pd.read_csv(synth_base_path)
        else:
            logger.info(f"Generating new shared synthetic dataset for {dataset_name}...")
            synth_data_base = data_manager.get_synthetic_data(dataset_name)
            synth_data_base.to_csv(synth_base_path, index=False)
            logger.info(f"📝 Saved shared synthetic dataset to: {synth_base_path.name}")

        for utility_model_type in config.UTILITY_MODELS:
            logger.info(f"\n{'-'*25} UTILITY MODEL: {utility_model_type.upper()} {'-'*25}")
            
            # --- 1. TRAIN UTILITY MODELS ---
            logger.info(f"PHASE 1: Training Utility Models for {utility_model_type} on {dataset_name}")
            utility_experiments = [('none', None)] + [('data', e) for e in config.EPSILON_VALUES] + [('algorithm', e) for e in config.EPSILON_VALUES]

            for dp_type, epsilon in tqdm(utility_experiments, desc=f"Training {utility_model_type}"):
                eps_str = "None" if epsilon is None else str(epsilon)
                model_path = Path(config.EXPERIMENT_DIR) / "models" / utility_model_type / f"{dataset_name}_{dp_type}_eps_{eps_str}.pkl"
                
                if saver.is_completed(model_path):
                    logger.info(f"⏭️ Skipping completed utility model: {model_path.name}")
                    continue

                X, y = data_manager.load_for_utility(dataset_name, dp_type, epsilon)
                X_proc, y_proc = data_manager.preprocess_data(dataset_name, X, y, task='utility')
                X_train, X_test, y_train, y_test = train_test_split(X_proc, y_proc, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y_proc)
                
                model, cv_score, score, params = run_training_and_evaluation(config, model_factory, utility_model_type, X_train, y_train, X_test, y_test, dp_type, epsilon)
                
                model_path.parent.mkdir(parents=True, exist_ok=True)
                with open(model_path, 'wb') as f: pickle.dump(model, f)
                
                saver.save_result({
                    'study_id': f"{utility_model_type}_{dataset_name}_{dp_type}", 'model_type': utility_model_type, 'dataset': dataset_name,
                    'dp_type': dp_type, 'epsilon': epsilon, 'worker_id': 0, 'experiment_idx': 0,
                    'best_cv_score': cv_score, 'final_test_score': score, 'best_params': json.dumps(params),
                    'model_path': str(model_path), 'timestamp': pd.Timestamp.now(), 'status': 'completed'
                })

            # --- 2. GENERATE ATTACK DATASETS ---
            logger.info(f"\nPHASE 2: Generating Attack Datasets using trained {utility_model_type} models on the shared synthetic data")
            X_synth_raw = synth_data_base.copy()
            # Use the dedicated UTILITY preprocessor to transform synthetic data before prediction
            X_synth_proc = data_manager.utility_preprocessors[dataset_name].transform(X_synth_raw)
            le_utility = data_manager.utility_label_encoders[dataset_name]
            
            attack_dataset_configs = [('none', None)] + [('data', e) for e in config.EPSILON_VALUES] + \
                                     [('algorithm', e) for e in config.EPSILON_VALUES] + [('output', e) for e in config.EPSILON_VALUES]

            for dp_type, epsilon in tqdm(attack_dataset_configs, desc=f"Generating attack data"):
                eps_str = "None" if epsilon is None else str(epsilon)
                attack_ds_path = Path(config.EXPERIMENT_DIR) / "datasets" / utility_model_type / f"{utility_model_type}_{dataset_name}_{dp_type}_eps_{eps_str}.csv"
                
                if attack_ds_path.exists():
                    logger.info(f"⏭️ Skipping existing attack dataset: {attack_ds_path.name}")
                    continue
                
                model_src_dp, model_src_eps = (dp_type, epsilon) if dp_type != 'output' else ('none', None)
                model_src_eps_str = "None" if model_src_eps is None else str(model_src_eps)
                utility_model_path = Path(config.EXPERIMENT_DIR) / "models" / utility_model_type / f"{dataset_name}_{model_src_dp}_eps_{model_src_eps_str}.pkl"
                
                if not utility_model_path.exists():
                    logger.warning(f"⚠️ Utility model not found: {utility_model_path}. Cannot generate attack data.")
                    continue
                
                with open(utility_model_path, 'rb') as f: utility_model = pickle.load(f)
                preds_enc = utility_model.predict(X_synth_proc)
                preds_dec = le_utility.inverse_transform(preds_enc)
                
                final_preds = [krr(p, le_utility.classes_, epsilon) for p in preds_dec] if dp_type == 'output' else preds_dec
                
                attack_df = synth_data_base.copy()
                attack_df[config.UTILITY_TARGETS[dataset_name]] = final_preds
                attack_ds_path.parent.mkdir(parents=True, exist_ok=True)
                attack_df.to_csv(attack_ds_path, index=False)
                logger.info(f"📝 Generated attack dataset: {attack_ds_path.name}")

            # --- 3. TRAIN ATTACK MODELS ---
            logger.info(f"\nPHASE 3: Training Attack Models on generated datasets")
            attack_datasets_dir = Path(config.EXPERIMENT_DIR) / "datasets" / utility_model_type
            if not attack_datasets_dir.exists(): continue
            
            for attack_dataset_path in tqdm(list(attack_datasets_dir.glob(f"*{dataset_name}*.csv")), desc="Training on attack datasets"):
                attack_target_name = attack_dataset_path.stem
                
                for attack_model_type in config.ATTACK_MODELS:
                    model_path = Path(config.EXPERIMENT_DIR) / "models" / "attack" / utility_model_type / f"on_{attack_target_name}_with_{attack_model_type}.pkl"
                    
                    if saver.is_completed(model_path):
                        logger.info(f"⏭️ Skipping completed attack model: {model_path.name}")
                        continue
                    
                    ds_name, X, y = data_manager.load_for_attack(attack_dataset_path)
                    # Use the dedicated ATTACK preprocessor
                    X_proc, y_proc = data_manager.preprocess_data(ds_name, X, y, task='attack')
                    if len(np.unique(y_proc)) < 2: continue
                    
                    X_train, X_test, y_train, y_test = train_test_split(X_proc, y_proc, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y_proc)
                    
                    model, cv_score, score, params = run_training_and_evaluation(config, model_factory, attack_model_type, X_train, y_train, X_test, y_test)

                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(model_path, 'wb') as f: pickle.dump(model, f)
                    
                    dp_info = attack_target_name.split('_eps_')
                    saver.save_result({
                        'study_id': f"attack_{attack_target_name}_with_{attack_model_type}", 'model_type': attack_model_type, 'dataset': dataset_name,
                        'dp_type': dp_info[0].split('_')[-1], 'epsilon': dp_info[1] if len(dp_info)>1 else None,
                        'best_cv_score': cv_score, 'final_test_score': score, 'best_params': json.dumps(params),
                        'model_path': str(model_path), 'timestamp': pd.Timestamp.now(), 'status': 'completed'
                    })

    logger.info("\n" + "="*80)
    logger.info("✅✅✅ EXPERIMENT PIPELINE COMPLETE ✅✅✅")
    logger.info(f"All results saved in: {config.EXPERIMENT_DIR}")
    logger.info(f"Summary CSV: {config.RESULTS_CSV_PATH}")
    logger.info("="*80)

if __name__ == "__main__":
    main()