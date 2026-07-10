"""
run_experiment_real_drift.py
=============================
Real drift experiments: validate the privacy-preserving bound and
recertification trigger against ACTUAL population shifts, not injected
synthetic drift.

*** Rewritten against the corrected src/drift.py and src/bound.py (see
their module docstrings for the itemized diff). Two changes from the
previous version of this file: ***

1. Experiment C (Folktables) now calls the group-conditional drift
   estimator and the full four-term Theorem-2 bound, instead of the old
   joint-histogram TV distance and Lipschitz-based envelope.

2. Experiment D is NEW: it validates the framework on REAL cross-center
   drift in the FLamby Fed-Heart-Disease dataset (the four raw UCI
   Cleveland / Hungarian / Switzerland / VA hospital files). The previous
   version of this repository loaded FLamby data in run_experiment.py,
   printed a baseline certificate, and then explicitly stopped -- its own
   log said "That is a different experiment script", but this script only
   supported Folktables. Experiment D closes that gap: it certifies a
   model on one real hospital site's population and tests recertification
   against the other real sites, which is exactly the "hospitals joining
   a consortium with different demographics" scenario from the paper's
   Introduction, using real data instead of a synthetic proxy for it.

  - EXPERIMENT C (Folktables, real US Census microdata):
      TEMPORAL drift: same state, different years (e.g., CA 2016 vs 2018)
      GEOGRAPHIC drift: same year, different states (e.g., CA vs TX in 2016)

  - EXPERIMENT D (FLamby Fed-Heart-Disease, real hospital data):
      CROSS-CENTER drift: certify on one real hospital's population,
      test recertification against each of the other real hospitals'
      populations -- a real analogue of "a new hospital with a different
      patient population joins the consortium."

USAGE
-----
    # Temporal drift: same state, different years
    python run_experiment_real_drift.py --baseline_year 2016 --current_year 2018 --state CA

    # Geographic drift: same year, different states
    python run_experiment_real_drift.py --baseline_state CA --current_state TX --year 2016

    # Multiple state comparisons
    python run_experiment_real_drift.py --compare_all_states --year 2016

    # Real cross-center drift on FLamby Fed-Heart-Disease
    python run_experiment_real_drift.py --flamby

Outputs are written to ./outputs/ as CSV + report.
"""

import argparse
import os
import numpy as np
import pandas as pd

from src import data as D
from src import fairness as F
from src import drift as DR
from src import bound as B

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

SEED = 42
ACS_FEATURE_COLS = ["age", "education", "hours", "occ_code"]

EPS_TOTAL_DEFAULT = 2.0
DELTA_TOTAL_DEFAULT = 1e-5
ALPHA1_TOTAL_DEFAULT = 0.05
ALPHA2_TOTAL_DEFAULT = 0.05


# ------------------------------------------------------------------
# Display utilities
# ------------------------------------------------------------------
def display_dataframe(df, title="", max_rows=None):
    if title:
        print(f"\n{title}")
        print("-" * 70)
    pd.set_option('display.max_rows', max_rows if max_rows else len(df))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df.to_string(index=False))
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')


def calibrate_epsilon_max(eps0, epsilon_max=None, abs_margin=0.02, rel_margin=0.05):
    """Choose a realistic fairness tolerance relative to the baseline
    fairness level, unless the caller supplies one explicitly."""
    if epsilon_max is not None:
        return float(epsilon_max)
    return max(float(eps0) + abs_margin, float(eps0) * (1.0 + rel_margin))


# ------------------------------------------------------------------
# Shared core: certify on a baseline population, evaluate against a
# "current" population, using the CORRECTED Section 2.4-2.7 pipeline.
# This one function backs Experiment C (temporal/geographic) AND
# Experiment D (real FLamby cross-center drift) -- one implementation,
# not three copy-pasted ones that can silently diverge.
# ------------------------------------------------------------------
def run_snapshot_comparison(baseline_fed, current_fed, feature_cols,
                             baseline_label, current_label,
                             eps_total=EPS_TOTAL_DEFAULT, delta_total=DELTA_TOTAL_DEFAULT,
                             alpha1_total=ALPHA1_TOTAL_DEFAULT, alpha2_total=ALPHA2_TOTAL_DEFAULT,
                             xi=0.0, epsilon_max=None, gamma=0.02, privacy_mode="flat",
                             comparison_type="generic"):
    print(f"\n{'='*70}")
    print(f"{comparison_type.upper()} DRIFT: {baseline_label} -> {current_label}")
    print(f"{'='*70}")

    print(f"\nTraining federated model on {baseline_label} baseline...")
    scaler, model = F.federated_average_train(baseline_fed, seed=SEED, feature_cols=feature_cols)
    eps0 = F.demographic_parity(baseline_fed, scaler, model, feature_cols=feature_cols)
    acc0 = F.accuracy(baseline_fed, scaler, model, feature_cols=feature_cols)
    print(f"Baseline model ({baseline_label}):")
    print(f"  • Accuracy: {acc0:.4f}")
    print(f"  • Certified fairness eps0 = F(P0, h): {eps0:.4f}")

    effective_epsilon_max = calibrate_epsilon_max(eps0, epsilon_max)
    print(f"  • Fairness tolerance (eps_max): {effective_epsilon_max:.4f}")
    if eps0 > effective_epsilon_max:
        print(f"  [WARN] eps0 already exceeds eps_max; per Algorithm 1 this baseline "
              f"model should not have been certified in the first place.")

    print(f"\nMeasuring true fairness on {current_label} data...")
    dp_true = F.demographic_parity(current_fed, scaler, model, feature_cols=feature_cols)
    fairness_gap = dp_true - eps0
    true_violation = dp_true > effective_epsilon_max
    print(f"  • F(P_t, h): {dp_true:.4f}   (gap vs baseline: {fairness_gap:+.4f})")
    print(f"  • True violation (F(P_t,h) > eps_max)? {true_violation}")

    # Freeze bin edges on the UNION of baseline+current so both populations
    # are binned consistently even if their marginal ranges differ (needed
    # across real states/years/centers, which can have quite different
    # feature ranges).
    combined = {**{f"base_{k}": v for k, v in baseline_fed.items()},
                **{f"cur_{k}": v for k, v in current_fed.items()}}
    edges = DR.freeze_bin_edges(combined, feature_cols)

    print(f"\nComputing group-conditional drift (Definition 3)...")
    delta0_true = DR.true_group_drift(baseline_fed, current_fed, 0, feature_cols, edges)
    delta1_true = DR.true_group_drift(baseline_fed, current_fed, 1, feature_cols, edges)
    print(f"  • True (non-private) Delta_0: {delta0_true:.4f}   Delta_1: {delta1_true:.4f}")
    det_bound = B.deterministic_bound(eps0, delta0_true, delta1_true)

    eps_dp, delta_dp = B.per_round_privacy_budget(eps_total, delta_total, horizon_T=1, mode=privacy_mode)
    rng = np.random.default_rng(SEED)
    group0, group1, delta_hat_total = DR.private_drift_estimate(
        baseline_fed, current_fed, feature_cols, edges,
        eps_dp, delta_dp, alpha1_total, alpha2_total, xi=xi, rng=rng)
    priv_bound = B.composed_certification_bound(eps0, group0, group1)
    print(f"  • Privacy-composed drift estimate (Delta_hat_0+Delta_hat_1): {delta_hat_total:.4f}")

    print(f"\nBound validation:")
    print(f"  • Deterministic bound (Theorem 1, true drift): {det_bound:.4f}"
          f"   valid (>= true)? {det_bound >= dp_true - 1e-9}")
    print(f"  • Privacy-composed bound (Theorem 2): {priv_bound:.4f}"
          f"   valid (>= true)? {priv_bound >= dp_true - 1e-9}")

    triggered, B_t, state = B.recertification_trigger(eps0, group0, group1, effective_epsilon_max, gamma=gamma)
    print(f"\nRecertification decision:")
    print(f"  • State: {state.value}   (trigger fires: {triggered})")
    print(f"  • True violation? {true_violation}   Correct decision? {triggered == true_violation}")

    return {
        "comparison_type": comparison_type,
        "baseline": baseline_label,
        "current": current_label,
        "eps0": eps0,
        "dp_true": dp_true,
        "fairness_gap": fairness_gap,
        "delta0_true": delta0_true,
        "delta1_true": delta1_true,
        "delta_hat": delta_hat_total,
        "det_bound": det_bound,
        "priv_bound": priv_bound,
        "det_valid": det_bound >= dp_true - 1e-9,
        "priv_valid": priv_bound >= dp_true - 1e-9,
        "epsilon_max": effective_epsilon_max,
        "true_violation": true_violation,
        "state": state.value,
        "trigger_decision": triggered,
        "correct_decision": triggered == true_violation,
    }


# ------------------------------------------------------------------
# EXPERIMENT C: Folktables (real US Census microdata)
# ------------------------------------------------------------------
def experiment_c_temporal(baseline_year, current_year, state, **kwargs):
    print(f"\nLoading baseline data ({state}, {baseline_year})...")
    baseline_fed = D.load_folktables([state], year=baseline_year, task="income")
    print(f"Loading current data ({state}, {current_year})...")
    current_fed = D.load_folktables([state], year=current_year, task="income")
    return run_snapshot_comparison(
        baseline_fed, current_fed, ACS_FEATURE_COLS,
        baseline_label=f"{state} {baseline_year}", current_label=f"{state} {current_year}",
        comparison_type="temporal", **kwargs)


def experiment_c_geographic(baseline_state, current_state, year, **kwargs):
    print(f"\nLoading baseline data ({baseline_state}, {year})...")
    baseline_fed = D.load_folktables([baseline_state], year=year, task="income")
    print(f"Loading current data ({current_state}, {year})...")
    current_fed = D.load_folktables([current_state], year=year, task="income")
    return run_snapshot_comparison(
        baseline_fed, current_fed, ACS_FEATURE_COLS,
        baseline_label=f"{baseline_state} {year}", current_label=f"{current_state} {year}",
        comparison_type="geographic", **kwargs)


def experiment_c_all_pairs(year, **kwargs):
    states = ["CA", "TX", "NY", "FL", "PA", "IL", "OH", "GA"]
    results_list = []
    for i, baseline_state in enumerate(states):
        for current_state in states[i + 1:]:
            try:
                results_list.append(experiment_c_geographic(baseline_state, current_state, year, **kwargs))
            except Exception as e:
                print(f"[WARN] Failed to compare {baseline_state} vs {current_state}: {e}")
    return results_list


# ------------------------------------------------------------------
# EXPERIMENT D: FLamby Fed-Heart-Disease (real cross-center drift)
# ------------------------------------------------------------------
def experiment_d_flamby(baseline_center=0, **kwargs):
    """
    Certifies the model on ONE real FLamby Fed-Heart-Disease center and
    evaluates recertification against each OTHER real center -- a genuine
    (not synthetic) analogue of the paper's "a new hospital with different
    demographics joins the consortium" scenario, using real UCI heart
    disease data (Cleveland/Hungarian/Switzerland/VA) already staged in
    ./data/flamby.

    Returns a list of result dicts, one per (baseline_center, other_center)
    pair.
    """
    print("\nLoading FLamby Fed-Heart-Disease (all 4 real hospital centers)...")
    fed = D.load_flamby_heart_disease()
    center_ids = sorted(fed.keys())
    baseline_key = f"center_{baseline_center}"
    if baseline_key not in fed:
        raise ValueError(f"Unknown baseline_center={baseline_center}; available: {center_ids}")

    baseline_fed = {baseline_key: fed[baseline_key]}
    results = []
    for key in center_ids:
        if key == baseline_key:
            continue
        current_fed = {key: fed[key]}
        results.append(run_snapshot_comparison(
            baseline_fed, current_fed, D.FLAMBY_HEART_FEATURE_COLS,
            baseline_label=f"FLamby {baseline_key}", current_label=f"FLamby {key}",
            comparison_type="flamby_cross_center", **kwargs))
    return results


# ------------------------------------------------------------------
def write_results_report(df_results, args, outdir=OUTDIR, filename="experiment_c_report.txt"):
    report_path = os.path.join(outdir, filename)
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("REAL DRIFT VALIDATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("DETAILED RESULTS\n" + "-" * 80 + "\n")
        f.write(df_results.to_string(index=False) + "\n\n")
        f.write("SUMMARY STATISTICS\n" + "-" * 80 + "\n")
        f.write(f"Total comparisons: {len(df_results)}\n")
        f.write(f"Valid deterministic bounds: {df_results['det_valid'].sum()} / {len(df_results)}\n")
        f.write(f"Valid privacy-composed bounds: {df_results['priv_valid'].sum()} / {len(df_results)}\n")
        f.write(f"True violations: {df_results['true_violation'].sum()} / {len(df_results)}\n")
        f.write(f"Correct trigger decisions: {df_results['correct_decision'].sum()} / {len(df_results)}\n")
        f.write(f"Trigger accuracy: {df_results['correct_decision'].mean()*100:.1f}%\n")
        f.write(f"\nReport generated: {pd.Timestamp.now()}\n")
        f.write("=" * 80 + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Real drift experiments (Folktables and/or FLamby)")
    parser.add_argument("--baseline_year", type=str)
    parser.add_argument("--current_year", type=str)
    parser.add_argument("--state", type=str)
    parser.add_argument("--baseline_state", type=str)
    parser.add_argument("--current_state", type=str)
    parser.add_argument("--year", type=str, default="2016")
    parser.add_argument("--compare_all_states", action="store_true")
    parser.add_argument("--flamby", action="store_true", help="Run Experiment D: real FLamby cross-center drift")
    parser.add_argument("--flamby_baseline_center", type=int, default=0)
    parser.add_argument("--eps_total", type=float, default=EPS_TOTAL_DEFAULT)
    parser.add_argument("--epsilon_max", type=float, default=None,
                         help="Fairness tolerance (defaults to eps0 + 0.02 / 5%% above eps0)")
    parser.add_argument("--privacy_mode", choices=["flat", "basic", "advanced"], default="flat")
    args = parser.parse_args()

    print("=" * 70)
    print("REAL DRIFT VALIDATION")
    print("=" * 70)

    results_list = []
    out_prefix = "experiment_c"

    if args.flamby:
        out_prefix = "experiment_d_flamby"
        results_list = experiment_d_flamby(
            baseline_center=args.flamby_baseline_center,
            eps_total=args.eps_total, epsilon_max=args.epsilon_max,
            privacy_mode=args.privacy_mode)
    elif args.compare_all_states:
        results_list = experiment_c_all_pairs(args.year, eps_total=args.eps_total,
                                               epsilon_max=args.epsilon_max, privacy_mode=args.privacy_mode)
    elif args.baseline_year and args.current_year and args.state:
        results_list = [experiment_c_temporal(
            args.baseline_year, args.current_year, args.state,
            eps_total=args.eps_total, epsilon_max=args.epsilon_max, privacy_mode=args.privacy_mode)]
    elif args.baseline_state and args.current_state:
        results_list = [experiment_c_geographic(
            args.baseline_state, args.current_state, args.year,
            eps_total=args.eps_total, epsilon_max=args.epsilon_max, privacy_mode=args.privacy_mode)]
    else:
        print("\nNo drift parameters specified. Running default: CA 2016 vs 2018")
        results_list = [experiment_c_temporal(
            "2016", "2018", "CA",
            eps_total=args.eps_total, epsilon_max=args.epsilon_max, privacy_mode=args.privacy_mode)]

    if results_list:
        df_results = pd.DataFrame(results_list)
        csv_path = os.path.join(OUTDIR, f"{out_prefix}_real_drift.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")

        report_path = write_results_report(df_results, args, OUTDIR, filename=f"{out_prefix}_report.txt")
        print(f"✓ Report saved to: {report_path}")

        print("\n" + "=" * 70)
        print("Summary Results")
        print("=" * 70)
        display_dataframe(df_results, "Real Drift Comparison Results")

        print("\n" + "-" * 70)
        print(f"Total comparisons: {len(df_results)}")
        print(f"Valid deterministic bounds: {df_results['det_valid'].sum()} / {len(df_results)}")
        print(f"Valid privacy-composed bounds: {df_results['priv_valid'].sum()} / {len(df_results)}")
        print(f"True violations: {df_results['true_violation'].sum()} / {len(df_results)}")
        print(f"Correct trigger decisions: {df_results['correct_decision'].sum()} / {len(df_results)}")
        print(f"Trigger accuracy: {df_results['correct_decision'].mean()*100:.1f}%")
        print("\n" + "=" * 70)
        print(f"All outputs written to: {OUTDIR}")
        print("=" * 70)


if __name__ == "__main__":
    main()
