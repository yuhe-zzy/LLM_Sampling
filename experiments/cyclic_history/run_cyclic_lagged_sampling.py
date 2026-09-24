"""Experiment 2: feedback extrapolation, with positive IPO/DPO training losses.

IPO matches identity-payoff extrapolation under the recorded joint pair law.
DPO extrapolates its actual BT-projection feedback, NOT elementwise-logit PsiPO.
"""
from run_cyclic_history import main

if __name__ == "__main__":
    main(required_scheme="lagged_sampling")
