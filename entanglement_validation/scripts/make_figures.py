"""
make_figures.py
---------------
Orchestrates all validation runs (R1 to R7) in sequence using config.yaml.

Usage (repo root recommended):
    python -m entanglement_validation.scripts.make_figures

You can also run directly from VS Code by opening this file and running it.
"""

import os
import sys
import subprocess


def run_script(script_filename, conf_path):
    """
    Run a single validation script as a subprocess with --conf pointing to config.yaml.
    """
    script_path = os.path.join(os.path.dirname(__file__), script_filename)
    print("\n=== Running {} ===".format(script_filename))
    try:
        subprocess.run(
            [sys.executable, script_path, "--conf", conf_path],
            check=True
        )
        print("--- {} completed successfully ---".format(script_filename))
    except subprocess.CalledProcessError as e:
        print("*** {} failed with error code {} ***".format(script_filename, e.returncode))
        sys.exit(1)


def resolve_config_path():
    """
    Resolve the config.yaml path regardless of whether you run from repo root
    or from inside entanglement_validation/scripts.
    """
    # scripts dir -> entanglement_validation
    validation_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # entanglement_validation/config.yaml
    conf_path = os.path.join(validation_dir, "config.yaml")

    if not os.path.isfile(conf_path):
        print("Could not find config.yaml at: {}".format(conf_path))
        sys.exit(1)
    return conf_path


def main():
    conf_path = resolve_config_path()

    scripts = [
        "validation_r1_newtonian.py",
        "validation_r2_bianchi.py",
        "validation_r3_scaling.py",
        "validation_r4_perihelion.py",
        "validation_r5_overview.py",
        "validation_r6_lrl.py",
        "validation_r7_light_bending.py",
    ]

    for script in scripts:
        run_script(script, conf_path)

    print("\nAll validations completed. Check the output directory set in config.yaml.")


if __name__ == "__main__":
    main()
