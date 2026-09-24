"""Experiment 1: reference-history ablation for actual IPO or DPO."""
from run_cyclic_history import main

if __name__ == "__main__":
    main(required_scheme="lagged_reference")
