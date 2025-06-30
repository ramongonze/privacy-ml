"""Calculate the accuracy of the attack model."""

import numpy as np
import pandas as pd
import itertools as it
from privacy_evaluation import model_inversion_acc
from tqdm.auto import tqdm
from datetime import datetime
import os
import pickle

REPOSITORY_PATH = "/Users/ramongonze/phd/courses/privacidade_ml/privacy-ml" # privacy-ml repository path
RESULTS_PATH = os.path.join(REPOSITORY_PATH, "results")
log = lambda msg : print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def attack(dataset, qids, target, sensitive):
    """Measure the accuracy of the attack model A.

    Parameters:
        dataset (str): Dataset, "adult" or "hospitals".
        qids (list): List of QIDs.
        target (str): Target attribute.
        sensitive (str): Sensitive attribute.        
    """
    # Original dataset path
    data_ori = pd.read_csv(os.path.join(REPOSITORY_PATH, f"data/{dataset}/{dataset}_pp.csv"))

    if dataset == "hospitals":
        np.random.seed(78923465)
        data_ori = data_ori.sample(50000)
    
    results_path = os.path.join(RESULTS_PATH, "privacy_evaluation.csv")

    # Privacy parameters
    epsilons = [0.1, 0.5, 1, 10, 20, 50, 100]

    # Write header
    write_header = not os.path.exists(results_path)
    with open(results_path, "a") as file:
        if write_header:
            file.write("dataset,noise,epsilon,target_model,attack_model,accuracy\n")


    noise_param = list(it.product(
        ["data", "algorithm", "output"], # where noise was added
        epsilons
    ))
    ml_models = ["random_forest", "naive_bayes", "logistic_regression"]

    for noise in tqdm(["none"] + noise_param, desc="Noise"):
        if noise == "none":
            ep = None
        else:
            noise, ep = noise
            ep = f"{ep:.1f}"

        for M_name in tqdm(["random_forest", "naive_bayes", "logistic_regression"], desc="Target Model", leave=False):
            # Load M
            try:
                # The model M for "output" is always the same
                if noise == "output":
                    noise_m = "none"
                    ep_m = "None"
                else:
                    noise_m = noise
                    ep_m = ep
                
                with open(
                    os.path.join(RESULTS_PATH, f"target_models/{M_name}/{dataset}_{noise_m}_eps_{ep_m}.pkl"), 'rb'
                ) as file:
                    M = pickle.load(file)

                for A_name in tqdm(ml_models, desc="Attack Model", leave=False):
                    try:
                        # Load A
                        with open(
                            os.path.join(RESULTS_PATH, f"attack_models/{M_name}_{dataset}_{noise}_eps_{ep}_{A_name}_attack.pkl"), 'rb'
                        ) as file:
                            A = pickle.load(file)

                        try:
                            acc = model_inversion_acc(
                                data_ori=data_ori,
                                qids=qids,
                                target=target,
                                sensitive=sensitive,
                                M=M,
                                A=A,
                                dp_output=(noise == "output"),
                                epsilon=ep
                            )

                            with open(results_path, "a") as file:
                                file.write(f"{dataset},{noise},{ep},{M_name},{A_name},{acc:.10f}\n")

                        except Exception as e:
                            print(f"Error at {dataset},{noise},{ep},{M_name},{A_name}\nError: {e}\n")  
                    except Exception as e:
                        print(f"Error loading A: {dataset},{noise},{ep},{M_name},{A_name}\nError: {e}\n")
            except Exception as e:
                print(f"Error loading M: {dataset},{noise},{ep},{M_name}\nError: {e}\n")

def main():
    # Adult dataset
    log("Evaluating privacy on adult dataset")
    attack(
        dataset="adult",
        qids=["age", "workclass", "occupation", "sex", "education", "native-country", "marital-status"],
        target="income",
        sensitive="race"
    )
    
    # Hospitals
    log("Evaluating privacy on hospitals dataset")
    attack(
        dataset="hospitals",
        qids=["TYPE_OF_ADMISSION", "PAT_ZIP", "PAT_COUNTY", "PAT_STATUS", "SEX_CODE", "RACE", "ADMIT_WEEKDAY", "PAT_AGE"],
        target="TOTAL_CHARGES",
        sensitive="PRINC_DIAG_CODE"
    )

    log("Finished experiment")

if __name__ == "__main__":
    main()