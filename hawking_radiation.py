def compute_hawking_radiation(df_mi, Lx, Ly, n_sites):
       horizon = list(range(Lx * (Ly // 2), Lx * (Ly // 2 + 1)))
       outside = list(range(n_sites))[-Lx:]
       mi_t = 0.0
       for i in horizon:
           for j in outside:
               pair = f"{min(i,j)}-{max(i,j)}"
               if pair in df_mi["Site Pair"].values:
                   mi_t += df_mi[df_mi["Site Pair"] == pair]["Mutual Information"].iloc[0]
       return mi_t