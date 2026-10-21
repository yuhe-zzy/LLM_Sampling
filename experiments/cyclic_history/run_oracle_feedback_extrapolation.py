"""Positive-loss, full-preference-matrix-assisted feedback extrapolation.

This is deliberately not named a two-sampler estimator.
"""
from run_cyclic_history import main

if __name__ == "__main__":
    main(required_scheme="oracle_feedback_extrapolation")
