def compute_hawking_radiation(df_mi, Lx, Ly, n_sites):
    """
    Compute average mutual information across horizon bipartition, matching GitHub's implementation.

    Parameters:
    - df_mi: DataFrame with 'Site Pair' and 'Mutual Information'.
    - Lx, Ly: Lattice dimensions.
    - n_sites: Total sites (Lx * Ly).

    Returns:
    - mi_t: Scaled average MI across horizon (~0.3-0.6337 for 3x3).
    """
    print(f"Entering compute_hawking_radiation with Lx={Lx}, Ly={Ly}, n_sites={n_sites}")
    
    # GitHub bipartition: horizon = middle row, outside = last row
    horizon = list(range(Lx * (Ly // 2), Lx * (Ly // 2 + 1)))
    outside = list(range(n_sites))[-Lx:]
    
    print(f"Horizon: {horizon}, Outside: {outside}")
    
    mi_t = 0.0
    mi_count = 0
    for i in horizon:
        for j in outside:
            pair = f"{min(i,j)}-{max(i,j)}"
            if pair in df_mi["Site Pair"].values:
                mi_val = df_mi[df_mi["Site Pair"] == pair]["Mutual Information"].iloc[0]
                print(f"MI for pair {i}-{j}: {mi_val:.6f}")
                mi_t += mi_val
                mi_count += 1
    
    # Scale for Page curve, adjusted for unnormalized MI
    base_scale = 9.0  
    scale_factor = base_scale * (9 / n_sites)  # Adjust for lattice size
    avg_mi = (mi_t / mi_count * scale_factor) if mi_count > 0 else 0.0
    print(f"Average MI across horizon: {avg_mi:.6f}, scale_factor={scale_factor:.6f}")
    
    return avg_mi