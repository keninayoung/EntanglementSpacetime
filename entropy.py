def compute_entropy(df_mi, n_sites):
       cut = list(range(n_sites // 2))
       S_approx = 0.0
       for i in cut:
           for j in range(n_sites):
               if j not in cut:
                   pair = f"{min(i,j)}-{max(i,j)}"
                   if pair in df_mi["Site Pair"].values:
                       S_approx += df_mi[df_mi["Site Pair"] == pair]["Mutual Information"].iloc[0]
       return S_approx