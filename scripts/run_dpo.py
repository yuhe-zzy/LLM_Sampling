"""Sequence-sum DPO on fixed transitive or cyclic pairs, without a reward oracle."""
from run_preference_oracle_core import run_experiment

if __name__ == "__main__":
    run_experiment("dpo", nonoracle=True)
