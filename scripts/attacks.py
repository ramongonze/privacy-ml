"""Calculate the accuracy of the attack model."""

import numpy as np
import pandas as pd
from privacy_evaluation import model_inversion_acc
from tqdm.auto import tqdm
import os

REPOSITORY_PATH = "/home/ramongonze/phd/privacy-ml" # privacy-ml repository path
RESULTS_PATH = os.path.join(REPOSITORY_PATH, "results")

def attack(results_path, dataset_path, qids, target, sensitive, domain_sensitive, all_M:dict, all_A:dict, dp):
    """Measure the accuracy of the attack model A.

        all_M (dict[str, object]): Dictionary with target model's name and the model object. The possible keys are "random_forest", "neural_network", "naive_bayes" and "logistic_regression".
        all_A (dict[str, object]): Dictionary with attack model's name and the model object. The possible keys are "random_forest", "neural_network", "naive_bayes" and "logistic_regression".
    """
    # Privacy parameters
    epsilons = [0.1, 0.5, 1, 10, 20, 50, 100]

    # Write header
    with open(results_path, "w") as f:
        f.write("target_model,attack_model,dp,epsilon,accuracy\n")
         
    for M_name, M in all_M.items(): # Target model
        for A_name, A in all_A.items(): # Attack model
            if dp:
                for ep in epsilons:
                    acc = model_inversion_acc(
                        dataset_path=dataset_path,
                        qids=qids,
                        target=target,
                        sensitive=sensitive,
                        domain_sensitive=domain_sensitive,
                        M=M,
                        A=A,
                        dp=dp,
                        epsilon=ep
                    )
                    with open(results_path, "a") as f:
                        f.write(f"{M_name},{A_name},{dp},{ep},{acc:.10f}\n")
            else:
                acc = model_inversion_acc(
                    dataset_path=dataset_path,
                    qids=qids,
                    target=target,
                    sensitive=sensitive,
                    domain_sensitive=domain_sensitive,
                    M=M,
                    A=A,
                    dp=dp
                )
                with open(results_path, "a") as f:
                    f.write(f"{M_name},{A_name},False,None,{acc:.10f}\n")

def main():
    # DATASET INFO #####################################
    # Adult
    adult_dataset_path = os.path.join(REPOSITORY_PATH, "data/adult/adult_pp.csv")
    adult_qids = ["age", "workclass", "occupation", "sex", "education", "native-country", "marital-status"]
    adult_target = "income"
    adult_sensitive = "race"
    adult_domain_sensitive = ["Asian-Pac-Islander", "Amer-Indian-Eskimo", "Black", "Other", "White"]
    
    # Hospitals
    hospitals_dataset_path = os.path.join(REPOSITORY_PATH, "data/hospitals/hospitals_pp.csv")
    hospitals_qids = ["TYPE_OF_ADMISSION", "PAT_ZIP", "PAT_COUNTY", "PAT_STATUS", "SEX_CODE", "RACE", "ADMIT_WEEKDAY", "PAT_AGE"]
    hospitals_target = "TOTAL_CHARGES"
    hospitals_sensitive = "PRINC_DIAG_CODE"
    hospitals_domain_sensitive = ['M1711', 'M1712', 'M1612', 'M1611', 'T814XXA', 'E6601', 'L03116', 'G7281', 'I69354', 'I4891', 'R5381', 'I69351', 'J441', 'J209', 'G459', 'J9621', 'A0472', 'A419', 'E1169', 'I132', 'K5720', 'J9600', 'L03115', 'E11621', 'I639', 'N390', 'J690', 'N10', 'E1152', 'J9622', 'J9601', 'M48061', 'I214', 'J189', 'T83511A', 'I110', 'J101', 'J440', 'I2699', 'N179', 'A4189', 'K529', 'I480', 'J1000', 'F319', 'F250', 'F332', 'F209', 'F333', 'F322', 'F3481', 'I130', 'I350', 'K922', 'K8590', 'J45901', 'R0789', 'E860', 'E871', 'K56609', 'S72142A', 'I2510', 'R55', 'K5732', 'J159', 'O6981X0', 'Z3801', 'O34219', 'O76', 'O700', 'O34211', 'Z3800', 'O134', 'O80', 'O99824', 'K56600', 'O365930', 'O701', 'J210', 'K8000', 'Z3831', 'K3580', 'I25110', 'E8770', 'E1110', 'E1010', 'O9902', 'O321XX0', 'I160', 'A4151', 'I120', 'O480', 'E875', 'O1414', 'R079', 'O4103X0', 'O6014X0', 'Z5111', 'O4202', 'O4292']
    ####################################################
    
    # TRAINING DATASET: ORIGINAL #######################
    # Load target models
    M_trained_original = []

    # Load attack models
    A_trained_original = []
    
    # Adult
    for dp in [False,True]:
        attack(
            results_path=os.path.join(RESULTS_PATH, "attack_adult_original.csv"),
            dataset_path=adult_dataset_path,
            qids=adult_qids,
            target=adult_target,
            sensitive=adult_sensitive,
            domain_sensitive=adult_domain_sensitive,
            all_M=M_trained_original,
            all_A=A_trained_original,
            dp=dp
        )

    # Hospitals
    for dp in [False,True]:
        attack(
            results_path=os.path.join(RESULTS_PATH, "attack_hospitals_original.csv"),
            dataset_path=hospitals_dataset_path,
            qids=hospitals_qids,
            target=hospitals_target,
            sensitive=hospitals_sensitive,
            domain_sensitive=hospitals_domain_sensitive,
            all_M=M_trained_original,
            all_A=A_trained_original,
            dp=dp
        )
    ####################################################

    # TRAINING DATASET: SANITIZED WITH LDP #############
    # Load target models
    M_trained_ldp = []

    # Load attack models
    A_trained_ldp = []

    # Adult
    attack(
        results_path=os.path.join(RESULTS_PATH, "attack_adult_ldp.csv"),
        dataset_path=adult_dataset_path,
        qids=adult_qids,
        target=adult_target,
        sensitive=adult_sensitive,
        domain_sensitive=adult_domain_sensitive,
        all_M=M_trained_ldp,
        all_A=A_trained_ldp,
        dp=False
    )

    # Hospitals
    attack(
        results_path=os.path.join(RESULTS_PATH, "attack_hospitals_ldp.csv"),
        dataset_path=hospitals_dataset_path,
        qids=hospitals_qids,
        target=hospitals_target,
        sensitive=hospitals_sensitive,
        domain_sensitive=hospitals_domain_sensitive,
        all_M=M_trained_ldp,
        all_A=A_trained_ldp,
        dp=False
    )
    ####################################################

if __name__ == "__main__":
    main()