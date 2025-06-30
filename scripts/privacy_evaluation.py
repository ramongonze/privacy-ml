"""Evaluate the performance of the attack model."""

import gc
import math
import functools
import numpy as np
import pandas as pd
from mechanisms import krr
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import sys
import os

REPOSITORY_PATH = "/home/ramongonze/phd/privacy-ml" # privacy-ml repository path

def max_isclose(iterable, *, rel_tol=1e-9, abs_tol=0.0):
    key = functools.cmp_to_key(
        lambda x, y: (
            0 if math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)
            else -1 if x < y else 1
        )
    )
    return max(iterable, key=key)

def make_guess(predictions:list, confidences:list[float]):
    """Adversary's guess when she has a list of predictions for the sensitive attribute and the model's confidences for each prediction.
    
    Takes the value with the maximum confidence. If there are more than 2 
    
    Parameters:
        predictions (list): List of predictions (attribute values) for the sensitive attribute.
        confidences (list[float]): Model's confidence for each prediction (list of floats).

    Returns:
        guess (any): Adversary's guess.
    """
    maximum_confidence = max_isclose(confidences)
    candidates = []
    for i in np.arange(len(predictions)):
        if math.isclose(confidences[i], maximum_confidence):
            candidates.append(predictions[i])
    
    # Choose randomly one value from the argmax set
    return np.random.choice(candidates)

def process_x(X:pd.DataFrame):
    # Create preprocessors
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ])
    preprocessor.fit(X)
    X_processed = preprocessor.transform(X)
    return X_processed

def process_y(y:pd.Series):
    # Fit preprocessors on original data
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    y_processed = label_encoder.transform(y)
    return label_encoder, y_processed

def model_inversion_acc(
        data_ori:pd.DataFrame,
        qids:list[str],
        target:str,
        sensitive:str,
        M,
        A,
        dp_output=False,
        epsilon=None
    ):
    """Adversary's accuracy when trying to guess the sensitive attribute value. The evaluation is done considering all individuals in the original dataset as possible targets. The accuracy is the number of individuals the adversary guessed correctly the sensitive attribute using the attack model.

    Parameters:
        dataset_path (str): Original dataset path (pre-processed).
        qids (list[str]): List of quasi-identifiers.
        target (str): Target attribute for the machine learning model.
        sensitive (str): Sensitive attribute.
        M: Target model. It must implement the method 'predict' that returns, for each instance, an array of confidences (same order as 'domain_sensitive').
        A: Attack model. It must implement the method 'predict' that returns, for each instance, an array of confidences (same order as 'domain_sensitive').
        dp_output (bool): Whether to apply noise in the prediction of M. The noise is going to be added using kRR mechanism and the privacy level is passed through parameter `epsilon`.Default is False.
        epsilon (float): Privacy parameter. Is is used only when `dp_output`=True.

    Returns:
        accuracy (float): Adversary's accuracy.
    """
    # Load original dataset (pre-processed)
    domain_sensitive = data_ori[sensitive].unique().tolist()
    n = len(data_ori) # Number of individuals

    X_M = data_ori[qids].copy()
    X_M = X_M.loc[X_M.index.repeat(len(domain_sensitive))] # Duplicate each individual k k = |domain(sensitive)| times
    X_M[sensitive] = domain_sensitive * n # Add all possible sensitive values to each duplicate of each individual
    
    # Process the fetaures to be in the same format as the model
    X_M = process_x(X_M)

    y_M_conf = M.predict_proba(X_M)
    y_M_pred = np.argmax(y_M_conf, axis=1)

    if dp_output:
        if isinstance(epsilon, str):
            epsilon = float(epsilon)
        y_M_pred = [krr(c, M.classes_, epsilon) for c in y_M_pred]

    # Get the predictions to the original domain
    label_encoder, _ = process_y(data_ori[target])
    y_M_pred = label_encoder.inverse_transform(y_M_pred)

    # STEP 4 #############
    # Get predictions and confidences for the sensitive attribute
    
    # Build the dataset to pass through A
    X_A = data_ori[qids].copy()
    X_A = X_A.loc[X_A.index.repeat(len(domain_sensitive))] # Duplicate each individual k = |domain(sensitive)| times
    X_A[target] = y_M_pred # Predictions from M

    X_A = process_x(X_A)
    y_A_conf = A.predict_proba(X_A)
    y_A_pred = np.argmax(y_A_conf, axis=1)

    # Get the predictions to the original domain
    label_encoder, _ = process_y(data_ori[sensitive])
    y_A_pred = label_encoder.inverse_transform(y_A_pred)

    correct_guesses = 0
    for i in np.arange(n):
        y_A_conf_ind = y_A_conf[i*len(domain_sensitive):(i+1)*len(domain_sensitive)]
        y_A_conf_ind = np.max(y_A_conf_ind, axis=1)
        y_A_pred_ind = y_A_pred[i*len(domain_sensitive):(i+1)*len(domain_sensitive)]
        y_original = data_ori.loc[i][sensitive]
        guess = make_guess(y_A_pred_ind, y_A_conf_ind)
        if guess == y_original:
            correct_guesses += 1

    accuracy = correct_guesses/n
    return accuracy
