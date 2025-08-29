# cli.py
# Command-line entry for processing MI CSVs.
import argparse, os
from .units import UnitsConfig
from . import etensor as et

def main():
    p = argparse.ArgumentParser(description="Build curvature, Einstein-like tensors, and Page-curve from MI CSVs.")
    sub = p.add_subparsers(dest="cmd")
    b = sub.add_parser("from-mi", help="Process MI CSVs into curvature/Einstein and optional Page-curve.")
    b.add_argument("--glob", required=True, help="Glob for MI CSVs, e.g., 'spacetime_outputs/mi_t*.csv'")
    b.add_argument("--out", required=True, help="Output directory")
    b.add_argument("--alpha", type=float, default=1.0, help="Coupling multiplier for G(i)")
    b.add_argument("--ell0", type=float, default=1.0, help="Length scale for MI->distance")
    b.add_argument("--horizonA", type=str, default=None, help="Comma-separated node ids for horizon A set")
    args = p.parse_args()
    if args.cmd == "from-mi":
        units = UnitsConfig(ell0=args.ell0, chi=1.0)
        A = [int(x) for x in args.horizonA.split(",")] if args.horizonA else None
        os.makedirs(args.out, exist_ok=True)
        et.process_folder(args.glob, args.out, alpha=args.alpha, units=units, horizon_A=A)
        print(f"[from-mi] Wrote outputs to {args.out}")
    else:
        p.print_help()

if __name__ == "__main__":
    main()
