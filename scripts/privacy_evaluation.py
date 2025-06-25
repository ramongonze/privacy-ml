"""Evaluate the performance of the attack model."""

import math
import functools
import numpy as np
import pandas as pd
from mechanisms import krr
from tqdm.auto import tqdm
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

def model_inversion_acc(
        dataset_path:str,
        qids:list[str],
        target:str,
        sensitive:str,
        domain_sensitive:list,
        M,
        A,
        dp=False,
        epsilon=math.log(2)
    ):
    """Adversary's accuracy when trying to guess the sensitive attribute value. The evaluation is done considering all individuals in the original dataset as possible targets. The accuracy is the number of individuals the adversary guessed correctly the sensitive attribute using the attack model.

    Parameters:
        dataset_path (str): Original dataset path (pre-processed).
        qids (list[str]): List of quasi-identifiers.
        target (str): Target attribute for the machine learning model.
        sensitive (str): Sensitive attribute.
        domain_sensitive (list): Domain of the sensitive attribute.
        M: Target model. It must implement the method 'predict' that returns, for each instance, an array of confidences (same order as 'domain_sensitive').
        A: Attack model. It must implement the method 'predict' that returns, for each instance, an array of confidences (same order as 'domain_sensitive').
        dp (bool): Whether to apply noise in the prediction of M. The noise is going to be added using kRR mechanism and the privacy level is passed through parameter `epsilon`.Default is False.
        epsilon (float): Privacy parameter. Is is used only when `dp`=True.

    Returns:
        accuracy (float): Adversary's accuracy.
    """

    # Load original dataset (pre-processed)
    data_ori = pd.read_csv(dataset_path)

    # Get the list of predictions for the target attribute
    correct_guesses = 0
    for i in np.arange(len(data_ori)):
        individual_qids = data_ori.loc[i][qids].tolist() # Individual's QID
        individual_sensitive = data_ori.loc[i][sensitive] # Individual's real sensitive value

        # STEP 3 #############
        # Build tuples of (qids,sensitive) to be passed through M to get the target predictions
        features_prediction = [individual_qids + [value] for value in domain_sensitive]
        features_prediction = pd.DataFrame(features_prediction, columns=qids + [sensitive])
        target_confidences = M.predict(features_prediction)
        target_predictions = [domain_sensitive[np.argmax(c)] for c in target_confidences]
        
        if dp:
            target_predictions = [krr(c, domain_sensitive, epsilon) for c in target_predictions]

        # Build the dataset to pass through A
        features_attack = features_prediction[qids] # QID values
        features_attack[target] = target_predictions # Predictions from M

        # STEP 4 #############
        # Get predictions and confidences for the sensitive attribute
        sensitive_confidences = A.predict(features_attack)
        sensitive_predictions = [domain_sensitive[np.argmax(c)] for c in sensitive_confidences]

        # Make the guess and check whether the adversary was correct
        guess = make_guess(sensitive_predictions, sensitive_confidences)
        if guess == individual_sensitive:
            correct_guesses += 1

    accuracy = correct_guesses/len(data_ori)

    return accuracy