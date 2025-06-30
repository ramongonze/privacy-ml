
"""
Evaluate the performance of the attack model using vectorized operations.
This script relies on a pre-saved DataManager object from the training pipeline
to ensure consistent data preprocessing.
"""

import numpy as np
import pandas as pd

# This mechanism is also used in unified_train.py
def krr(original_value, domain, epsilon: float):
    """k-Randomized Response mechanism."""
    k = len(domain)
    if k <= 1: return original_value
    p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    
    if np.random.random() < p:
        return original_value
    else:
        other_options = [v for v in domain if v != original_value]
        return np.random.choice(other_options)

def model_inversion_acc(
        data_ori: pd.DataFrame,
        qids: list[str],
        target_col: str,
        sensitive_col: str,
        utility_model,
        attack_model,
        data_manager,
        dataset_name: str,
        dp_output: bool = False,
        epsilon: float = None
    ):
    """
    Calculates model inversion attack accuracy using efficient, vectorized operations.

    The adversary's goal is to guess the sensitive attribute of a target individual,
    knowing their QIDs and having black-box access to the target model M.

    The process is:
    1. For each individual, create `k` hypothetical records, where `k` is the
       number of possible values for the sensitive attribute. Each record has the
       individual's true QIDs but a different hypothetical sensitive value.
    2. Feed all `n * k` hypothetical records into the utility model (M) to get
       `n * k` predictions for the target attribute.
    3. Feed these `n * k` records (QIDs + M's prediction) into the attack model (A)
       to get `n * k` confidence scores for each possible sensitive value.
    4. For each individual, find which of the `k` initial hypotheses yields the
       highest confidence score from the attack model. This becomes the guess.
    5. Compare the guesses against the true sensitive values to get the accuracy.

    Parameters:
        data_ori (pd.DataFrame): Original dataset of individuals to attack.
        qids (list[str]): List of quasi-identifier column names.
        target_col (str): The name of the target attribute column for M.
        sensitive_col (str): The name of the sensitive attribute column.
        utility_model: The trained target model (M).
        attack_model: The trained attack model (A).
        data_manager: The loaded DataManager object with fitted preprocessors.
        dataset_name (str): The name of the dataset ('adult' or 'hospitals').
        dp_output (bool): If True, apply k-RR to M's output.
        epsilon (float): Epsilon for k-RR if dp_output is True.

    Returns:
        float: The adversary's accuracy.
    """
    # --- 1. SETUP ---
    n = len(data_ori)
    if n == 0:
        return 0.0

    # Get objects from the DataManager for consistent processing
    utility_preprocessor = data_manager.utility_preprocessors[dataset_name]
    attack_preprocessor = data_manager.attack_preprocessors[dataset_name]
    le_utility = data_manager.utility_label_encoders[dataset_name]
    le_sensitive = data_manager.sensitive_label_encoders[dataset_name]
    domain_sensitive = le_sensitive.classes_
    k = len(domain_sensitive)

    # --- 2. CREATE HYPOTHETICAL RECORDS for UTILITY MODEL (M) ---
    # Create a dataframe with n * k rows for all hypothetical scenarios
    # This is more memory-efficient than repeating a pandas DataFrame
    qids_repeated = pd.DataFrame(np.repeat(data_ori[qids].values, k, axis=0), columns=qids)
    sensitive_tiled = pd.Series(np.tile(domain_sensitive, n), name=sensitive_col)
    
    # The utility model's input features include QIDs and the sensitive attribute
    X_M_raw = pd.concat([qids_repeated, sensitive_tiled], axis=1)

    # --- 3. PREDICT WITH UTILITY MODEL (M) ---
    X_M_processed = utility_preprocessor.transform(X_M_raw)
    y_M_pred_encoded = utility_model.predict(X_M_processed)

    y_M_pred_decoded = le_utility.inverse_transform(y_M_pred_encoded)

    if dp_output:
        if isinstance(epsilon, str):
            epsilon = float(epsilon)
        
        # Apply k-Randomized Response to the decoded predictions
        y_M_pred_decoded = np.array([krr(p, le_utility.classes_, epsilon) for p in y_M_pred_decoded])

    # Decode predictions back to original labels (e.g., '<=50K')
    y_M_pred_decoded = le_utility.inverse_transform(y_M_pred_encoded)



    # --- 4. CREATE INPUTS for ATTACK MODEL (A) ---
    # The attack model's input features are QIDs and the utility model's prediction
    X_A_raw = qids_repeated.copy() # Use .copy() to avoid SettingWithCopyWarning
    X_A_raw[target_col] = y_M_pred_decoded
    
    # --- 5. PREDICT WITH ATTACK MODEL (A) ---
    X_A_processed = attack_preprocessor.transform(X_A_raw)
    # Get confidence scores for each possible sensitive value
    y_A_confidences = attack_model.predict_proba(X_A_processed) # Shape: (n * k, k)

    # --- 6. VECTORIZED GUESSING ---
    # Reshape confidences to (n, k, k): (individual, hypothesis, confidence_for_each_class)
    conf_matrix = y_A_confidences.reshape(n, k, k)

    # For each individual, we want the confidence score for the value we hypothesized.
    # e.g., for hypothesis 0 (e.g., 'White'), what's the confidence for 'White'?
    # This corresponds to the diagonal of the inner (k, k) matrices.
    # `conf_for_hypotheses` will have shape (n, k)
    conf_for_hypotheses = np.diagonal(conf_matrix, axis1=1, axis2=2)

    # For each individual, find which hypothesis (0 to k-1) yielded the max confidence
    best_guess_indices = np.argmax(conf_for_hypotheses, axis=1) # Shape: (n,)

    # Convert indices back to actual class labels (e.g., 'White', 'Black')
    guesses = le_sensitive.classes_[best_guess_indices]
    
    # --- 7. CALCULATE ACCURACY ---
    true_values = data_ori[sensitive_col].values
    accuracy = np.mean(guesses == true_values)
    
    return accuracy