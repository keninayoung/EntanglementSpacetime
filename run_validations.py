# run_validations.py

import os
import sys
import shutil
import glob
import time
from datetime import datetime

# Keep linear algebra libraries from eating all cores
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Output directory    
OUT_DIR = os.path.join(REPO_ROOT, "entanglement_validation", "validation_outputs")

def stash_r13_outputs(label):
    """
    Move all freshly produced R13p artifacts into a unique subfolder so
    subsequent runs do not overwrite them.

    label: short string describing the run (no spaces).
    Returns the path of the folder where files were stashed.
    """
    # add a timestamp to guarantee uniqueness
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = os.path.join(OUT_DIR, f"R13p_{label}_{stamp}")
    os.makedirs(dest, exist_ok=True)

    # files we typically produce
    patterns = [
        "R13p_*.csv",
        "R13p_*.txt",
        "R13p_*.md",
        "R13p_*.tex",
        "R13p_posterior_heatmap.png",
    ]

    moved_any = False
    for pat in patterns:
        for f in glob.glob(os.path.join(OUT_DIR, pat)):
            try:
                shutil.move(f, os.path.join(dest, os.path.basename(f)))
                moved_any = True
            except Exception as e:
                print("[stash] WARN could not move {} -> {} ({})".format(f, dest, e))

    if not moved_any:
        print("[stash] NOTE: no R13p artifacts matched; did the run produce outputs?")
    else:
        print("[stash] Saved R13p artifacts to:", dest)
    return dest

def main():
    # Resolve config.yaml no matter where you run this file from
    conf_path = os.path.join(REPO_ROOT, "entanglement_validation", "config.yaml")
    ext_csv_path = os.path.join(REPO_ROOT, "entanglement_validation", "external_datasets")
    ext_csv=ext_csv_path+ "\\r13_external_from_R12.csv"

    if not os.path.isfile(conf_path):
        print("Config not found at:", conf_path)
        sys.exit(1)

    

    # Import the run() functions directly and call them
    try:
        from entanglement_validation.scripts.validation_r1_newtonian import run as run_r1
        from entanglement_validation.scripts.validation_r2_bianchi import run as run_r2
        from entanglement_validation.scripts.validation_r3_scaling import run as run_r3
        from entanglement_validation.scripts.validation_r4_perihelion import run as run_r4
        from entanglement_validation.scripts.validation_r5_overview import run as run_r5
        from entanglement_validation.scripts.validation_r6_lrl import run as run_r6
        from entanglement_validation.scripts.validation_r7_light_bending import run as run_r7
        from entanglement_validation.scripts.validation_r8_peri_uncertainty import run as run_r8
        from entanglement_validation.scripts.validation_r9_ppn_cross_checks import run as run_r9
        from entanglement_validation.scripts.validation_r10_publish_report import run as run_r10
        from entanglement_validation.scripts.validation_r11_edg_constraints import run as run_r11
        from entanglement_validation.scripts.validation_r12_edg_sweeps import run as run_r12
        from entanglement_validation.scripts.validation_r13p_joint_observational_fit import run as run_r13p
        
      
        from entanglement_validation.scripts.r13_build_from_r12 import run as r13_from_r12
        #from entanglement_validation.scripts.r13_novelty_sweep import run as run_sweep

    except Exception as e:
        print("Import error while loading validation modules:")
        print(repr(e))
        sys.exit(1)

    print("Using config:", conf_path)
    print("Repo root   :", REPO_ROOT)

    print("\n=== Running R1 ===")
    run_r1(conf_path)
    print("\n=== Running R2 ===")
    run_r2(conf_path)
    print("\n=== Running R3 ===")
    run_r3(conf_path)
    print("\n=== Running R4 ===")
    run_r4(conf_path)
    print("\n=== Running R5 ===")
    run_r5(conf_path)

    #---------------------------------------------------------------------
    # These take much longer... 
    #---------------------------------------------------------------------
    print("=== Running R6 ===")
    run_r6(conf_path)
    print("=== Running R7 ===")
    run_r7(conf_path)
    print("=== Running R8 ===")
    run_r8(conf_path)
    print("=== Running R9 ===")
    run_r9(conf_path)
    print("=== Running R10 ===")
    run_r10(conf_path)
    print("=== Running R11 ===")
    run_r11(conf_path)
    print("=== Running R12 ===")
    run_r12(conf_path=conf_path, eps=[0.9, 0.95, 1.0, 1.05, 1.1], reuse=False, check=True)

    #-----------------------------------------------------------------------
    #
    #   Run Validations for 13. 
    #
    #------------------------------------------------------------------------

    # Build the external CSV using tighter perihelion sigmas
    print("\n=== Building R13 external CSV from R12 ===")
    r13_from_r12(
        conf_path=conf_path,
        csv_out=ext_csv,
        use_planets="Mercury,Venus,Earth",
        eps_target=1.0,
        sigma_floor=1e-12  # use tight floor with bootstrap sigmas
    )


    #Run 13p B: Null-mean control (same anchors, mu=0, broader sigma)
    print("\n=== R13+: B) Sgr A* fractional test  ===")
    run_r13p(
        conf_path=conf_path,
        ext_csv=os.path.join(REPO_ROOT, "entanglement_validation", "external_datasets", "r13_external_from_R12.csv"),
        eps_grid="0.98,1.02,401",
        Lq_grid="auto",
        use_gamma=True, gamma_mu=1.0, gamma_sigma=2.3e-5,
        add_eht="sgrA", eht_mode="fractional",
        eht_mu_frac=0.02, eht_sigma_frac=0.005,
        eht_mass=4.1e6, eht_kappa=0.1, eht_p=2.0,
        add_gw=False, check=True
    )
    # Move all freshly produced R13p artifacts into a unique subfolder 
    stash_r13_outputs("B_sgrA_mu002_sig005")
    print("\n=== Complete - R13+: B) Sgr A* fractional test ===")

    # Run 13p C: M87* fractional grid (mu_frac, sigma_frac)
    print("=== R13+: C) M87* fractional grid (mu_frac, sigma_frac) ===")
    for mu_frac in (0.015, 0.02, 0.025):
        for sig_frac in (0.005, 0.01):
            tag = "C_m87_mu{:.3f}_sig{:.3f}".format(mu_frac, sig_frac).replace(".", "p")
            print("[grid] Running", tag)
            run_r13p(
                conf_path=conf_path,
                ext_csv=os.path.join(REPO_ROOT, "entanglement_validation", "external_datasets", "r13_external_from_R12.csv"),
                eps_grid="0.98,1.02,401",
                Lq_grid="auto",
                use_gamma=True, gamma_mu=1.0, gamma_sigma=2.3e-5,
                add_eht="m87", eht_mode="fractional",
                eht_mu_frac=mu_frac, eht_sigma_frac=sig_frac,
                eht_mass=6.5e9, eht_kappa=0.1, eht_p=2.0,
                add_gw=False, check=False
            )
            # Move all freshly produced R13p artifacts into a unique subfolder 
            stash_r13_outputs(tag)
    print("\n=== Complete - R13+: C) M87* fractional grid (mu_frac, sigma_frac) ===")

    # Run 13p D: Mapping stress test (kappa, p) on M87*
    print("=== D) Mapping stress test (kappa, p) on M87* ===")
    for kappa in (0.05, 0.1, 0.2):
        for p in (1.0, 2.0, 4.0):
            tag = "D_m87_k{}_p{}".format(str(kappa).replace(".", "p"), str(p).replace(".", "p"))
            print("[map] Running", tag)
            run_r13p(
                conf_path=conf_path,
                ext_csv=os.path.join(REPO_ROOT, "entanglement_validation", "external_datasets", "r13_external_from_R12.csv"),
                eps_grid="0.98,1.02,401",
                Lq_grid="auto",
                use_gamma=True, gamma_mu=1.0, gamma_sigma=2.3e-5,
                add_eht="m87", eht_mode="fractional",
                eht_mu_frac=0.02, eht_sigma_frac=0.005,
                eht_mass=6.5e9, eht_kappa=kappa, eht_p=p,
                add_gw=False, check=False
            )
        # Move all freshly produced R13p artifacts into a unique subfolder 
        stash_r13_outputs(tag)
    print("\n=== Complete - R13+: D) Mapping stress test (kappa, p) on M87* ===")


    
    # # Run 13p A: Anchored strong-field test (fractional EHT with +2% mean)
    # print("\n=== R13+: Anchored strong-field (M87*, fractional EHT with +2% mean) ===")
    # run_r13p(
    #     conf_path=conf_path,
    #     ext_csv=ext_csv,           # <-- includes perihelion + Cassini/Shapiro
    #     eps_grid="0.995,1.005,401",
    #     Lq_grid="auto",            # lets the script pick a wide, rs-scaled range
    #     add_eht="m87",
    #     eht_mu_frac=0.02,          # +2% assumed offset
    #     eht_sigma_frac=0.005,      # 0.5% fractional uncertainty
    #     # mapping knobs you are exploring:
    #     eht_kappa=0.1,
    #     eht_p=2.0,
    #     # leave GW off for now to isolate EHT
    #     add_gw=False,
    #     check=True
    # )
    # # Move all freshly produced R13p artifacts into a unique subfolder 
    # stash_r13_outputs("A_m87_mu002_sig005")
    # print("\n=== Complete - R13+: Anchored strong-field (M87*, fractional EHT with +2% mean) ===")


    # print("\n=== Running R13+ ===")
    
    # run_r13p(
    #     conf_path=conf_path,
    #     eps_grid="0.95,1.05,121",
    #     Lq_grid="0,1e9,121",
    #     add_eht="sgrA", eht_DGR=51.8, eht_sigma=3.0, eht_mass=4.1e6,
    #     eht_kappa=0.1, eht_p=2.0,
    #     add_gw=True, gw_value=0.0, gw_sigma=0.1, gw_rs_ref=1.0e4, gw_alpha=1.0, gw_p=2.0,
    #     check=True
    # )

    # ------------------------------------------------------------------
    # Choose ONE of the following blocks (A, B, or C). By default we run A.
    # ------------------------------------------------------------------

    # A) Built-in minimal dataset (Cassini gamma + Shapiro + Mercury perihelion).
    #    Append EHT (Sgr A* baseline) and a toy GW phasing row.
    # print("=== Running R13+ (built-in dataset + EHT SgrA + toy GW) ===")
    # run_r13p(
    #     conf_path=conf_path,
    #     # No ext_csv -> use built-in minimal dataset
    #     eps_grid="0.95,1.05,121",   # epsilon grid
    #     Lq_grid="0,1e9,121",        # L_q grid (meters)
    #     add_eht="sgrA",             # options: "none", "sgrA", "m87"
    #     # Optionally override default Sgr A* ring baseline/uncertainty/mass:
    #     # eht_DGR=51.8, eht_sigma=3.0, eht_mass=4.1e6,
    #     eht_kappa=0.1,              # strong-field coupling in D_pred
    #     eht_p=2.0,                  # power in D_pred
    #     add_gw=True,                # append a toy GW constraint
    #     gw_value=0.0,               # measured coeff (toy)
    #     gw_sigma=0.1,               # its uncertainty
    #     gw_rs_ref=1.0e4,            # reference rs in meters
    #     gw_alpha=1.0,
    #     gw_p=2.0,
    #     gw_gr_coeff=0.0,
    #     check=True
    # )

    # ------------------------------------------------------------------
    # B) Use external CSV and NO dynamic rows.
    # print("=== Running R13+ (external CSV only) ===")
    # ext_csv_path = os.path.join(REPO_ROOT, "entanglement_validation", "external_datasets", "r13_external.csv")
    # run_r13p(
    #     conf_path=conf_path,
    #     ext_csv=ext_csv_path,
    #     eps_grid="0.98,1.02,201",
    #     Lq_grid="0,5e8,201",
    #     check=True
    # )

    # ------------------------------------------------------------------
    # C) External CSV + add M87* EHT row + GW row.
    # print("=== Running R13+ (external CSV + EHT M87 + GW) ===")
    # run_r13p(
    #     conf_path=conf_path,
    #     ext_csv=ext_csv,             
    #     eps_grid="0.98,1.02,201",
    #     Lq_grid="0,5e7,201",
    #     add_eht="m87",               
    #     eht_DGR=42.0,               
    #     eht_sigma=3.0,               
    #     eht_mass=6.5e9,              
    #     eht_kappa=0.1,
    #     eht_p=2.0,
    #     add_gw=True,
    #     gw_value=0.0,
    #     gw_sigma=0.1,
    #     gw_rs_ref=1.0e4,
    #     gw_alpha=1.0,
    #     gw_p=2.0,
    #     gw_gr_coeff=0.0,
    #     check=True
    # )

    # print("=== Running R13+ (external CSV + EHT M87 + GW) ===")
    # run_r13p(
    #     conf_path=conf_path,
    #     ext_csv=ext_csv,
    #     eps_grid="0.98,1.02,201",
    #     Lq_grid="0,5e7,201",
    #     add_eht="m87",
    #     eht_DGR=42.0,
    #     eht_sigma=1.0,       # tighten from 3.0 to 1.0 microas (or your preferred value)
    #     eht_mass=6.5e9,
    #     eht_kappa=0.1,
    #     eht_p=2.0,
    #     add_gw=True,
    #     gw_value=0.02,
    #     gw_sigma=0.01,       # tighten GW toy uncertainty
    #     gw_rs_ref=1.0e4,
    #     gw_alpha=1.0,
    #     gw_p=2.0,
    #     gw_gr_coeff=0.0,
    #     check=True
    # )

    # print("=== Running R13+ (external CSV + optionally EHT + GW rows) ===")
    # run_r13p(
    #     conf_path=conf_path,
    #     ext_csv=ext_csv,
    #     eps_grid="0.98,1.02,201",
    #     Lq_grid="0,5e7,201",
    #     add_eht="m87",          # or "sgrA" or "none"
    #     eht_DGR=42.0,
    #     eht_sigma=1.5,          # realistic ring uncertainty in microas
    #     eht_mass=6.5e9,
    #     eht_kappa=0.1,
    #     eht_p=2.0,
    #     add_gw=True,
    #     gw_value=0.02,          # toy
    #     gw_sigma=0.01,          # toy
    #     gw_rs_ref=1.0e4,
    #     gw_alpha=1.0,
    #     gw_p=2.0,
    #     gw_gr_coeff=0.0,
    #     check=True
    # )

    # print("=== Running R13+ (external CSV + EHT + GW + forecast) ===")
    # run_r13p(
    #     conf_path=conf_path,
    #     ext_csv=os.path.join(REPO_ROOT, "entanglement_validation", "external_datasets", "r13_external_from_R12.csv"),
    #     eps_grid="0.98,1.02,201",
    #     Lq_grid="1e4,5e7,201",            # start above zero to avoid a degenerate UL printout
    #     add_eht="m87",                    # or "sgrA"
    #     eht_DGR=42.0,
    #     eht_sigma=1.5,                    # tighten to 1.0 or 0.8 to see stronger effects
    #     eht_mass=6.5e9,
    #     eht_kappa=0.1,
    #     eht_p=2.0,
    #     add_gw=True,
    #     gw_value=0.02,
    #     gw_sigma=0.01,
    #     gw_rs_ref=1.0e4,
    #     gw_alpha=1.0,
    #     gw_p=2.0,
    #     gw_gr_coeff=0.0,
    #     forecast_multipliers="1.0,0.75,0.5,0.33,0.25",
    #     check=True
    # )


    # print("=== Stress-test sensitivity to the model (scan k and p) ===")
    # for kappa in (0.05, 0.1, 0.2):
    #     for p in (2.0, 4.0):
    #         run_r13p(
    #             conf_path=conf_path,
    #             ext_csv=ext_csv,
    #             eps_grid="0.99,1.01,201",
    #             Lq_grid="0,3e7,301",
    #             add_eht="m87", eht_DGR=42.0, eht_sigma=1.0, eht_mass=6.5e9,
    #             eht_kappa=kappa, eht_p=p,
    #             add_gw=False,
    #             check=False
    #         )

    
    # # --- R13+: strong-field fractional + epsilon prior (Results show EDG > GR) ---
    # print("=== R13+: rs-scaled strong-field test (M87*, wide Lq, epsilon prior) ===")
    # run_r13p(
    #     conf_path=conf_path,
    #     # optional: include your external perihelion/gamma/Shapiro CSV to anchor epsilon
    #     ext_csv=os.path.join(REPO_ROOT, "entanglement_validation", "external_datasets", "r13_external_from_R12.csv"),
    #     eps_grid="0.98,1.02,401",
    #     Lq_grid="auto",               # auto-range from EHT fractional row + rs
    #     use_gamma=True,               # Gaussian prior on epsilon (Cassini-like)
    #     gamma_mu=1.0,
    #     gamma_sigma=2.3e-5,
    #     add_eht="m87",                # "m87" or "sgrA"
    #     eht_mode="fractional",        # use rs-scaled fractional offset
    #     eht_mu_frac=0.02,             # set to your assumed mean fractional offset
    #     eht_sigma_frac=0.005,          # fractional sigma (e.g., 1 microas / 42 microas ~ 0.024)
    #     eht_mass=6.5e9,               # M87*
    #     eht_kappa=0.1,
    #     eht_p=2.0,
    #     add_gw=False,                 # set True to add a GW fractional row
    #     check=True
    # )

    # print("=== Complete R13+ rs-scaled strong-field test (M87*, wide Lq, epsilon prior) ===")

    # print("=== R13+: A) Relax epsilon prior and repeat fractional EHT (M87*) ===")
    # run_r13p(
    #     conf_path=conf_path,
    #     # no ext_csv -> it uses the minimal default set unless your file expects one
    #     eps_grid="0.98,1.02,401",
    #     Lq_grid="0,1.2e13,401",     # wide strong-field range in meters
    #     add_eht="m87",              # adds the EHT fractional row for M87*
    #     # fractional EHT inputs (what you wanted to try):
    #     eht_mu_frac=0.02,           # +2% offset relative to GR ring diameter
    #     eht_sigma_frac=0.005,       # 0.5% uncertainty
    #     # EDG strong-field mapping knobs
    #     eht_kappa=0.1,
    #     eht_p=2.0,
    #     # leave GW off for now unless you also pass the GW kwargs
    #     check=True
    # )


    # print("=== Complete. R13+: A) Relax epsilon prior and repeat fractional EHT (M87*) ===")

    # print("\nR13+ completed. Check entanglement_validation/validation_outputs for:")
    # print("  - R13p_posterior_grid.csv")
    # print("  - R13p_best_fit.txt")
    # print("  - R13p_marginals.csv")
    # print("  - R13p_posterior_heatmap.png")
    # print("  - R13p_constraints.md / .tex")

    print("\nValidations completed. Check entanglement_validation/validation_outputs.")

if __name__ == "__main__":
    main()
