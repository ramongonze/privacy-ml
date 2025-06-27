"""Sanitize hospitals dataset using local differential privacy."""

import numpy as np
import pandas as pd
from mechanisms import krr, geometric_truncated
from tqdm.auto import tqdm
import os

REPOSITORY_PATH = "/home/ramongonze/phd/privacy-ml" # privacy-ml repository path
RESULTS_PATH = os.path.join(REPOSITORY_PATH, "results")

def main():
    # Read hospitals dataset
    data = pd.read_csv(os.path.join(REPOSITORY_PATH, "data/hospitals/hospitals_pp.csv"))
    
    # kRR will be applied to all columns, since all are categorical
    cols = ["TYPE_OF_ADMISSION", "PAT_ZIP", "PAT_COUNTY", "PAT_STATUS", "SEX_CODE", "RACE", "ADMIT_WEEKDAY", "PAT_AGE", "TOTAL_CHARGES", "PRINC_DIAG_CODE"]

    # Get column domains
    domains = {col:list(data[col].unique()) for col in cols}

    # Global epsilon, splitted equally for all columns
    epsilons = [0.1, 0.5, 1, 10, 20, 50, 100]
    for epsilon in epsilons:
        # Apply noise individually in each column
        data_san = data.copy()
        for col in tqdm(cols, desc=f"Sanitizing columns for epsilon = {epsilon}"):
            data_san[col] = data_san[col].apply(lambda x: krr(x, domains[col], epsilon/len(cols)))

        # Save sanitized dataset
        data_san.to_csv(os.path.join(RESULTS_PATH, f"hospitals_san_{epsilon}.csv"), index=False)

if __name__ == "__main__":
    main()