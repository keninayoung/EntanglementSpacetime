import pandas as pd
import numpy as np

def compute_entropy(df_mi, n_sites):
    """
    Compute average entanglement entropy from mutual information, matching GitHub's implementation.

    Parameters:
    - df_mi: DataFrame with 'Site Pair' and 'Mutual Information'.
    - n_sites: Total number of sites.

    Returns:
    - entropy: Average entanglement entropy (~1.11-0.58 for 3x3).
    """
    # entropies = []
    # for i in range(n_sites):
    #     entropy_i = sum(row["Mutual Information"] for _, row in df_mi.iterrows()
    #                     if int(row["Site Pair"].split("-")[0]) == i or
    #                     int(row["Site Pair"].split("-")[1]) == i)
    #     entropies.append(entropy_i / 2.0)
    # entropy = sum(entropies) / n_sites #* 200.0  # Scale to match GitHub
    # print(f"Computed entropy: {entropy:.6f}")

 
    cut = list(range(n_sites // 2))
    S_approx = 0.0
    for i in cut:
        for j in range(n_sites):
            if j not in cut:
                pair = f"{min(i,j)}-{max(i,j)}"
                if pair in df_mi["Site Pair"].values:
                    S_approx += df_mi[df_mi["Site Pair"] == pair]["Mutual Information"].iloc[0]
    entropy = S_approx

    return entropy