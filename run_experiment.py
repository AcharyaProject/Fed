"""
run_experiment.py
==================
End-to-end experiment runner for:

  "Continuous Fairness Assurance for Federated Learning: Privacy-
   Preserving Drift Monitoring and Provably Correct Recertification"

*** Rewritten to call the corrected src/drift.py and src/bound.py, which
now implement Definition 3 / Theorem 1 / Theorem 2 / Section 2.7-2.8
exactly (group-conditional drift, boundedness-only stability bound, the
full four-term composed bound, and Bonferroni-split confidence budgets
across the monitoring horizon). See src/drift.py and src/bound.py module
docstrings for the itemized diff against the previous version. ***

Runs two things:

  EXPERIMENT A (Tier 1 - bound validation):
      Injects controlled, KNOWN drift levels and checks that the
      deterministic bound and the privacy-composed bound both stay
      >= the true measured fairness violation (i.e. the bound is a
      valid, non-vacuous upper envelope), across a sweep of drift
      magnitudes.

  EXPERIMENT B (Tier 2/3 - trigger vs. baselines):
      Simulates a federation drifting over many rounds (gradual drift
      in some states + a sudden shock in one state, mimicking a new
      hospital/site joining) and compares:
        - our privacy-preserving, drift-triggered recertification
        - fixed-interval recertification
        - always-audit (upper bound on cost)
      on (i) number of expensive cryptographic audits performed and
      (ii) detection lag (rounds between the TRUE fairness violation
      and the policy actually catching it).

USAGE
-----
    python run_experiment.py                     # synthetic data (offline, default)
    python run_experiment.py --data real          # real Folktables data (needs internet)
    python run_experiment.py --data flamby_heart  # real FLamby Fed-Heart-Disease data
                                                   # (certification only -- see
                                                   # run_experiment_real_drift.py for the
                                                   # real cross-center drift experiment)

Outputs are written to ./outputs/ as CSV + PNG figures.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import data as D
from src import fairness as F
from src import drift as DR
from src import bound as B
from src import baselines as BL

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

STATES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]  # 8 simulated cross-silo clients
SEED = 42

# Table-1-style defaults
EPS_TOTAL_DEFAULT = 2.0
DELTA_TOTAL_DEFAULT = 1e-5
ALPHA1_TOTAL_DEFAULT = 0.05
ALPHA2_TOTAL_DEFAULT = 0.05
COARSENING_XI_DEFAULT = 0.0  # 0 for purely-binned continuous features unless a
                              # smoothness bound is separately justified (Section 2.2)


# ------------------------------------------------------------------
# Display utilities
# ------------------------------------------------------------------
def display_dataframe(df, title="", max_rows=None):
    """Pretty print a dataframe with a title."""
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


def display_summary_stats(df, title=""):
    """Display summary statistics of a dataframe."""
    if title:
        print(f"\n{title}")
        print("-" * 70)
    print(df.describe().to_string())


# ------------------------------------------------------------------
# Data loading (real Folktables if available & requested, else synthetic)
# ------------------------------------------------------------------
def get_certified_baseline(use_real=False, use_flamby=False, n_per_client=20000):
    if use_flamby:
        try:
            print("Attempting to load FLamby Fed-Heart-Disease "
                  "(requires flamby installed + dataset already downloaded)...")
            fed = D.load_flamby_heart_disease()
            print("Loaded FLamby Fed-Heart-Disease for centers:", list(fed.keys()))
            return fed, list(fed.keys()), "flamby_heart"
        except Exception as e:
            print(f"[WARN] FLamby load failed ({e}). Falling back to synthetic ACS-like data.")
    elif use_real:
        try:
            print("Attempting to download real ACS/Folktables data "
                  "(requires internet access to census.gov)...")
            real_states = ["CA", "TX", "NY", "FL", "PA", "IL", "OH", "GA"]
            fed = D.load_folktables(real_states, year="2016", task="income")
            print("Loaded real Folktables data for:", list(fed.keys()))
            return fed, real_states, "real"
        except Exception as e:
            print(f"[WARN] Real data load failed ({e}). "
                  f"Falling back to synthetic ACS-like data.")
    fed = D.make_synthetic_federation(STATES, n_per_client=n_per_client, seed=SEED)
    return fed, STATES, "synthetic"


# ------------------------------------------------------------------
# EXPERIMENT A: bound validation across a drift sweep
# ------------------------------------------------------------------
def experiment_a(baseline_fed, states, feature_cols, scaler, model, eps0,
                  eps_total=EPS_TOTAL_DEFAULT, delta_total=DELTA_TOTAL_DEFAULT,
                  alpha1_total=ALPHA1_TOTAL_DEFAULT, alpha2_total=ALPHA2_TOTAL_DEFAULT,
                  xi=COARSENING_XI_DEFAULT):
    rows = []
    rng = np.random.default_rng(SEED + 1)
    drift_levels = np.linspace(0.0, 1.0, 11)

    edges = DR.freeze_bin_edges(baseline_fed, feature_cols)
    # Single-shot validation sweep (not a multi-round horizon), so no
    # Bonferroni split is needed here -- use the total budgets directly
    # at T=1, matching Algorithm 1's per-round formulas at T=1.
    eps_dp, delta_dp = B.per_round_privacy_budget(eps_total, delta_total, horizon_T=1, mode="basic")

    for lvl in drift_levels:
        schedule = {st: lvl for st in states}  # uniform drift across all clients
        current_fed = D.make_drifted_snapshot(states, schedule, n_per_client=4000, seed=SEED + 100)

        dp_true = F.demographic_parity(current_fed, scaler, model, feature_cols=feature_cols)

        # True (non-private) group-conditional drift, for theory validation only.
        delta0_true = DR.true_group_drift(baseline_fed, current_fed, 0, feature_cols, edges)
        delta1_true = DR.true_group_drift(baseline_fed, current_fed, 1, feature_cols, edges)
        det_bound = B.deterministic_bound(eps0, delta0_true, delta1_true)

        group0, group1, delta_hat_total = DR.private_drift_estimate(
            baseline_fed, current_fed, feature_cols, edges,
            eps_dp, delta_dp, alpha1_total, alpha2_total, xi=xi, rng=rng)
        priv_bound = B.composed_certification_bound(eps0, group0, group1)

        rows.append(dict(
            drift_level=lvl, dp_true=dp_true,
            delta0_true=delta0_true, delta1_true=delta1_true,
            det_bound=det_bound,
            delta_hat_0=group0["delta_hat"], delta_hat_1=group1["delta_hat"],
            eta_0=group0["eta"], eta_1=group1["eta"],
            sigma_0=group0["sigma"], sigma_1=group1["sigma"],
            xi_0=group0["xi"], xi_1=group1["xi"],
            priv_bound=priv_bound,
            det_valid=det_bound >= dp_true - 1e-9,
            priv_valid=priv_bound >= dp_true - 1e-9,
        ))
        print(f"  drift={lvl:.1f}  DP_true={dp_true:.4f}  det_bound={det_bound:.4f}"
              f"  priv_bound={priv_bound:.4f}"
              f"  (delta_hat={delta_hat_total:.4f}, eta0+eta1={group0['eta']+group1['eta']:.4f},"
              f" sigma0+sigma1={group0['sigma']+group1['sigma']:.4f})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "experiment_a_bound_validation.csv"), index=False)

    display_dataframe(df, "EXPERIMENT A: Detailed Results")
    display_summary_stats(df, "EXPERIMENT A: Summary Statistics")

    plt.figure(figsize=(7, 5))
    plt.plot(df.drift_level, df.dp_true, "o-", label="True DP(P_t, h)", color="black", linewidth=2)
    plt.plot(df.drift_level, df.det_bound, "s--", label="Deterministic bound (Theorem 1, true drift)")
    plt.plot(df.drift_level, df.priv_bound, "^--", label=f"Privacy-composed bound (Theorem 2, eps_total={eps_total})")
    plt.xlabel("Injected drift level")
    plt.ylabel("Demographic parity difference")
    plt.title("Experiment A: Bound validity across injected drift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "experiment_a_bound_validation.png"), dpi=150)
    plt.close()

    print(f"\n  Deterministic bound valid (>= true) in {df.det_valid.mean()*100:.0f}% of settings")
    print(f"  Privacy-composed bound valid (>= true) in {df.priv_valid.mean()*100:.0f}% of settings")
    return df


# ------------------------------------------------------------------
# EXPERIMENT B: recertification trigger vs. baselines over many rounds
# ------------------------------------------------------------------
def build_round_schedule(states, n_rounds, shock_state, shock_round):
    """
    Builds a per-round drift schedule dict[state] -> drift_level in [0,1]:
      - most states drift slowly and linearly over time (realistic
        gradual demographic change)
      - one state has a sudden shock at `shock_round` (e.g. a new site
        with a very different population joins), simulating the
        "non-ideal, failure-prone" scenario the paper calls out.
    """
    schedules = []
    for t in range(n_rounds):
        sched = {}
        for st in states:
            gradual = min(1.0, 0.35 * (t / n_rounds))
            sched[st] = gradual
        if t >= shock_round:
            sched[shock_state] = min(1.0, 0.9)
        schedules.append(sched)
    return schedules


def experiment_b(baseline_fed, states, feature_cols, scaler, model, eps0,
                  eps_total=EPS_TOTAL_DEFAULT, delta_total=DELTA_TOTAL_DEFAULT,
                  alpha1_total=ALPHA1_TOTAL_DEFAULT, alpha2_total=ALPHA2_TOTAL_DEFAULT,
                  xi=COARSENING_XI_DEFAULT, epsilon_max=0.15, gamma=0.02,
                  n_rounds=40, fixed_interval=5, privacy_mode="flat",
                  n_per_client=20000):
    """
    NOTE ON n_per_client: the previous version of this experiment used
    n_per_client=3000. With the CORRECTED bound (which now includes the
    sigma_a finite-sample term and xi_a coarsening term that the old
    bound.py omitted -- see bound.py's module docstring), 3000 samples per
    client, split across m=16 histogram bins (4 features x 4 bins) with a
    strict Bonferroni confidence split over T=40 rounds, pushes sigma_a
    alone close to eps_max=0.15 even at ZERO drift, so the trigger fires
    on almost every round regardless of the true drift level. That is the
    sound bound correctly reporting "not enough samples/confidence budget
    to certify this tightly" -- it is not a bug, but it also is not an
    informative demo of drift-triggered audit savings. Raising
    n_per_client to 20000 brings sigma_a down to a level where the trigger
    tracks the true drift trajectory instead of firing unconditionally;
    see the chat discussion for the arithmetic.
    """
    rng = np.random.default_rng(SEED + 2)
    schedules = build_round_schedule(states, n_rounds, shock_state=states[-1], shock_round=n_rounds // 2)

    edges = DR.freeze_bin_edges(baseline_fed, feature_cols)

    # Section 2.8 / Corollary 1: Bonferroni-split confidence budgets across
    # the T-round horizon. The privacy budget is handled separately:
    # `privacy_mode="flat"` uses the paper's own Table-1 constant (eps_dp=2.0)
    # directly each round; `"basic"`/`"advanced"` instead compose a fixed
    # TOTAL privacy budget across all T rounds (see per_round_privacy_budget's
    # docstring for the tradeoff).
    alpha1_t, alpha2_t = B.per_round_confidence_budgets(alpha1_total, alpha2_total, n_rounds)
    eps_dp, delta_dp = B.per_round_privacy_budget(eps_total, delta_total, n_rounds, mode=privacy_mode)

    rows = []
    for t, sched in enumerate(schedules):
        current_fed = D.make_drifted_snapshot(states, sched, n_per_client=n_per_client, seed=SEED + 200 + t)
        dp_true = F.demographic_parity(current_fed, scaler, model, feature_cols=feature_cols)
        violation = dp_true > epsilon_max

        group0, group1, delta_hat_total = DR.private_drift_estimate(
            baseline_fed, current_fed, feature_cols, edges,
            eps_dp, delta_dp, alpha1_t, alpha2_t, xi=xi, rng=rng)
        triggered, B_t, state = B.recertification_trigger(eps0, group0, group1, epsilon_max, gamma=gamma)

        fixed_audit = BL.fixed_interval_policy(t, fixed_interval)
        always_audit = BL.always_audit_policy(t)

        rows.append(dict(round=t, dp_true=dp_true, violation=violation,
                          delta_hat=delta_hat_total, our_bound=B_t, state=state.value,
                          our_audit=triggered, fixed_audit=fixed_audit, always_audit=always_audit))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "experiment_b_trigger_vs_baselines.csv"), index=False)

    # --- summary metrics ---
    def detection_lag(policy_col):
        """Rounds between the FIRST true violation and the first audit at/after it."""
        viol_idx = df.index[df["violation"]].tolist()
        if not viol_idx:
            return np.nan
        first_viol = viol_idx[0]
        audit_idx = df.index[(df[policy_col]) & (df.index >= first_viol)].tolist()
        if not audit_idx:
            return len(df) - first_viol  # never caught within horizon
        return audit_idx[0] - first_viol

    summary = pd.DataFrame({
        "policy": ["Ours (drift-triggered)", "Fixed-interval", "Always-audit"],
        "n_audits": [df.our_audit.sum(), df.fixed_audit.sum(), df.always_audit.sum()],
        "detection_lag_rounds": [detection_lag("our_audit"),
                                  detection_lag("fixed_audit"),
                                  detection_lag("always_audit")],
    })
    summary.to_csv(os.path.join(OUTDIR, "experiment_b_summary.csv"), index=False)

    display_dataframe(df, "EXPERIMENT B: Full Trace Data (all rounds)")
    print("\n")
    display_dataframe(summary, "EXPERIMENT B: Policy Comparison Summary")

    print(f"\nKey Metrics:")
    print(f"  • Total rounds: {len(df)}")
    print(f"  • Total violations detected: {df.violation.sum()}")
    print(f"  • Our trigger audits: {df.our_audit.sum()}")
    print(f"  • Fixed-interval audits: {df.fixed_audit.sum()}")
    print(f"  • Always-audit audits: {df.always_audit.sum()}")

    # --- plots ---
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(df["round"], df.dp_true, color="black", linewidth=2, label="True DP(P_t, h)")
    axes[0].axhline(epsilon_max, color="red", linestyle=":", label="Tolerance epsilon_max")
    axes[0].scatter(df["round"][df.our_audit], df.dp_true[df.our_audit],
                     marker="^", color="tab:blue", s=60, label="Our trigger audits", zorder=5)
    axes[0].scatter(df["round"][df.fixed_audit], df.dp_true[df.fixed_audit] + 0.005,
                     marker="s", color="tab:orange", s=30, label="Fixed-interval audits", zorder=4)
    axes[0].set_ylabel("Demographic parity")
    axes[0].set_title("Experiment B: true fairness trajectory & when each policy audits")
    axes[0].legend(loc="upper left")

    axes[1].plot(df["round"], df.our_bound, color="tab:blue", label="Our privacy-composed bound (Theorem 2)")
    axes[1].axhline(epsilon_max, color="red", linestyle=":")
    axes[1].set_xlabel("Federated round")
    axes[1].set_ylabel("Bound value")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "experiment_b_trigger_trace.png"), dpi=150)
    plt.close()

    bar_fig, bax = plt.subplots(1, 2, figsize=(10, 4))
    bax[0].bar(summary.policy, summary.n_audits, color=["tab:blue", "tab:orange", "tab:gray"])
    bax[0].set_ylabel("# expensive cryptographic audits")
    bax[0].set_title("Audit cost")
    bax[0].tick_params(axis="x", rotation=15)

    bax[1].bar(summary.policy, summary.detection_lag_rounds, color=["tab:blue", "tab:orange", "tab:gray"])
    bax[1].set_ylabel("Detection lag (rounds)")
    bax[1].set_title("Violation detection speed")
    bax[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "experiment_b_summary_bars.png"), dpi=150)
    plt.close()

    return df, summary


# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["synthetic", "real", "flamby_heart"], default="synthetic")
    parser.add_argument("--eps_total", type=float, default=EPS_TOTAL_DEFAULT, help="Total DP privacy budget over the horizon")
    parser.add_argument("--epsilon_max", type=float, default=0.15, help="Regulator's fairness tolerance")
    parser.add_argument("--gamma", type=float, default=0.02, help="Uncertain-state margin below epsilon_max")
    parser.add_argument("--n_rounds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--privacy_mode", choices=["flat", "basic", "advanced"], default="flat",
                         help="How eps_dp is set across the T-round horizon (see bound.py "
                              "per_round_privacy_budget docstring). 'flat' matches Table 1.")
    parser.add_argument("--n_per_client_b", type=int, default=20000,
                         help="Samples per client per round in Experiment B (see experiment_b's "
                              "docstring for why this is larger than the original 3000).")
    args = parser.parse_args()

    global SEED
    SEED = args.seed

    print("=" * 70)
    print("STEP 1: Load / generate certified baseline federation")
    print("=" * 70)
    baseline_fed, states, source = get_certified_baseline(
        use_real=(args.data == "real"), use_flamby=(args.data == "flamby_heart"),
        n_per_client=args.n_per_client_b)
    print(f"Data source used: {source}  |  clients: {states}")

    feature_cols = D.FLAMBY_HEART_FEATURE_COLS if source == "flamby_heart" else None
    resolved_feature_cols = feature_cols or F.FEATURE_COLS

    print("\n" + "=" * 70)
    print("STEP 2: Train the certified federated model h (FedAvg, logistic reg.)")
    print("=" * 70)
    scaler, model = F.federated_average_train(baseline_fed, seed=SEED, feature_cols=feature_cols)
    eps0 = F.demographic_parity(baseline_fed, scaler, model, feature_cols=feature_cols)
    acc0 = F.accuracy(baseline_fed, scaler, model, feature_cols=feature_cols)
    print(f"Certified model: accuracy={acc0:.3f}, eps0=F(P0,h)={eps0:.4f}")
    if eps0 > args.epsilon_max:
        print(f"[WARN] eps0={eps0:.4f} already exceeds epsilon_max={args.epsilon_max:.4f}. "
              f"Per Algorithm 1 line 3-4, certification should abort and h should be "
              f"retrained before proceeding; continuing anyway for experimental purposes.")

    if source in ("real", "flamby_heart"):
        print("\n[NOTE] Experiments A and B below inject SYNTHETIC drift and are only "
              "meaningful for the synthetic generator, which has a controllable drift dial. "
              "Real data (Folktables/FLamby) has no such dial -- validating the bound on real "
              "drift means comparing two real snapshots directly (e.g. two FLamby centers, or "
              "two Folktables states/years). See run_experiment_real_drift.py, which now "
              "supports both Folktables (Experiment C) and FLamby (Experiment D).")
        print("All outputs written to:", OUTDIR)
        return

    print("\n" + "=" * 70)
    print("STEP 3: EXPERIMENT A — bound validation across injected drift")
    print("=" * 70)
    df_a = experiment_a(baseline_fed, states, resolved_feature_cols, scaler, model, eps0,
                         eps_total=args.eps_total)

    print("\n" + "=" * 70)
    print("STEP 4: EXPERIMENT B — trigger vs. fixed-interval vs. always-audit")
    print("=" * 70)
    df_b, summary_b = experiment_b(baseline_fed, states, resolved_feature_cols, scaler, model, eps0,
                                    eps_total=args.eps_total, epsilon_max=args.epsilon_max,
                                    gamma=args.gamma, n_rounds=args.n_rounds,
                                    privacy_mode=args.privacy_mode, n_per_client=args.n_per_client_b)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nAll outputs written to: {OUTDIR}")
    print("\nGenerated files:")
    output_files = [
        ("experiment_a_bound_validation.csv", "Detailed bound validation results"),
        ("experiment_a_bound_validation.png", "Bound validation plot"),
        ("experiment_b_trigger_vs_baselines.csv", "Full trigger trace across all rounds"),
        ("experiment_b_summary.csv", "Policy comparison summary"),
        ("experiment_b_trigger_trace.png", "Trigger vs baseline trace plot"),
        ("experiment_b_summary_bars.png", "Policy comparison bar charts"),
    ]
    for fname, desc in output_files:
        fpath = os.path.join(OUTDIR, fname)
        status = "✓" if os.path.exists(fpath) else "✗"
        print(f"  {status} {fname:40s} - {desc}")


if __name__ == "__main__":
    main()
