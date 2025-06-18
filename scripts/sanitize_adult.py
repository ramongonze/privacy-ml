"""Sanitize adult dataset using local differential privacy."""

import numpy as np
import pandas as pd
from mechanisms import krr, geometric_truncated
from tqdm.auto import tqdm
import os

REPOSITORY_PATH = "/home/ramongonze/phd/privacy-ml" # privacy-ml repository path
RESULTS_PATH = os.path.join(REPOSITORY_PATH, "results")

def main():
    # Read adult dataset
    data = pd.read_csv(os.path.join(REPOSITORY_PATH, "data/adult/adult_pp.csv"))
    
    # kRR will be applied to all columns, since all are categorical
    cols = ["age", "workclass", "occupation", "race", "sex", "education", "native-country", "marital-status", "income"]

    # Get column domains
    domains = {col:list(data[col].unique()) for col in cols}

    # Global epsilon, splitted equally for all columns
    epsilons = [0.1, 0.5, 1, 10, 20, 50, 100]
    for epsilon in epsilons:
        # Apply noise individually in each column
        data_san = data.copy()
        for col in tqdm(cols, desc=f"Sanitizing columns for epsilon = {epsilon}"):
            
            # Geometric truncated for age and kRR for the other attributes
            if col == "age":
                data_san[col].apply(
                    lambda x: geometric_truncated(
                        x=x,
                        lower=min(domains[col]),
                        upper=max(domains[col]),
                        epsilon=epsilon/len(cols)
                    )
                )
            else:
                data_san[col].apply(lambda x: krr(x, domains[col], epsilon/len(cols)))

        # Save sanitized dataset
        data_san.to_csv(os.path.join(RESULTS_PATH, f"adult_san_{epsilon}.csv"), index=False)

if __name__ == "__main__":
    main()