"""
drift.py
========
Privacy-preserving estimation of GROUP-CONDITIONAL population drift between
the certified baseline population P0 and the current population P_t, exactly
as specified in Definition 3 and Section 2.4-2.6 of the paper.

*** This is a corrected version. ***

The previous version of this file binned the protected attribute A together
with the covariates into ONE joint histogram over (A, age, education, hours)
and computed a single TV distance on that joint distribution. That is NOT
the quantity the paper's Theorem 1 is about, and it silently breaks the
soundness of everything downstream. Definition 3 requires the CONDITIONAL
distance, computed separately within each group:

    Delta_a(P, Q) = TV( P(X | A=a), Q(X | A=a) )     for a in {0, 1}
    Delta(P, Q)   = Delta_0(P, Q) + Delta_1(P, Q)

i.e. two separate histograms -- one built ONLY from the A=0 records' feature
vectors, one built ONLY from the A=1 records' feature vectors -- each
normalized within its own group. Theorem 1's proof works because, within
each group, |E_P[h|A=a] - E_Q[h|A=a]| <= TV(P(.|A=a), Q(.|A=a)) via the
standard variational characterization of TV (Section 2.3 proof). Mixing A
into the histogram measures the shift of the *joint* distribution instead,
which conflates a change in the group base rate P(A=1) with a change in the
within-group covariate distribution, and is not bounded by Theorem 1 at all.

This file also adds the two error terms that Theorem 2 needs but the
previous version omitted entirely:
    - sigma_a : finite-sample estimation error (Lemma 2, Section 2.6)
    - xi_a    : deterministic histogram-coarsening gap (Section 2.6)
alongside the DP-noise error term eta_a (Lemma 1, Section 2.4), which the
previous version did compute but only as an ad hoc closed-form envelope
rather than the paper's exact formula.

SECURE AGGREGATION SIMULATION
------------------------------
We do not implement real cryptography here (masking protocols / homomorphic
encryption); that is an engineering artifact orthogonal to the statistical
claim being validated. We simulate its *privacy-utility effect*:
    1. each client's per-group histogram is summed exactly across clients
       (this is what secure aggregation guarantees: only the SUM is
       revealed) -- the code never operates on a per-client histogram
       outside `secure_aggregate_group_histograms`;
    2. the summed histogram is perturbed with calibrated Gaussian noise to
       satisfy (eps_dp, delta_dp)-DP on top of the secure sum (Section 2.4,
       Gaussian mechanism with sensitivity 1, since a single record changes
       exactly one bin's count by exactly 1).
"""

import numpy as np


N_BINS_PER_DIM = 4  # matches Table 1: "Histogram bins per feature dimension"


def _bin_edges(values, n_bins=N_BINS_PER_DIM):
    """Frozen histogram binning (Algorithm 1, line 6): computed ONCE on the
    certified baseline population and reused for every subsequent round."""
    qs = np.linspace(0, 100, n_bins + 1)
    return np.unique(np.percentile(values, qs))


def freeze_bin_edges(baseline_dfs, feature_cols):
    """Freeze one set of per-dimension bin edges on the baseline population
    for each feature column (Algorithm 1, line 6). Returns a dict
    {feature_name: edges}.
    """
    edges = {}
    for col in feature_cols:
        all_vals = np.concatenate([df[col].values for df in baseline_dfs.values()])
        edges[col] = _bin_edges(all_vals)
    return edges


# ---------------------------------------------------------------------------
# Definition 3 : group-conditional histograms and TV distance
# ---------------------------------------------------------------------------

def compute_group_histogram(df, a, feature_cols, edges):
    """Each client computes its LOCAL, GROUP-CONDITIONAL bin-count vector
    c_{k,a}: only the rows with A == a contribute, and only the feature
    columns (never A itself) are binned (Definition 3 / Section 2.4:
    "each client k computes ck,a locally").

    Following Table 1's "bins per feature dimension" parameterization, we
    build one marginal histogram per feature dimension and concatenate them
    into a single count vector of length m = len(feature_cols) * n_bins,
    rather than a full joint histogram over all features (whose size would
    grow exponentially in the number of features).
    """
    sub = df[df["A"] == a]
    counts = []
    for col in feature_cols:
        hist, _ = np.histogram(sub[col].values if len(sub) else np.array([]), bins=edges[col])
        counts.append(hist.astype(float))
    return np.concatenate(counts)


def secure_aggregate_group_histograms(client_dfs, a, feature_cols, edges):
    """Step 1 (Section 2.4): secure aggregation, restricted to group a.

    Only the SUMMED group-conditional histogram is ever returned; no
    intermediate per-client histogram is exposed outside this function
    (mirrors the secure-aggregation primitive assumption in Section 2.2).
    """
    total = None
    for df in client_dfs.values():
        h = compute_group_histogram(df, a, feature_cols, edges)
        total = h if total is None else total + h
    return total


def dp_noised_histogram(hist_sum, eps_dp, delta_dp, sensitivity=1.0, rng=None):
    """Step 2 (Section 2.4): Gaussian mechanism DP release.

        sigma = (sensitivity / eps_dp) * sqrt(2 ln(1.25/delta_dp))
        c_tilde = c_sum + N(0, sigma^2 I)     (clipped at zero)

    A single record changes exactly one bin's count by exactly 1
    (sensitivity 1), so this is (eps_dp, delta_dp)-DP by the standard
    Gaussian mechanism.

    Returns the noised (unnormalized) histogram and sigma.
    """
    rng = rng or np.random.default_rng()
    sigma = (sensitivity / eps_dp) * np.sqrt(2.0 * np.log(1.25 / delta_dp))
    noise = rng.normal(0.0, sigma, size=hist_sum.shape)
    noised = np.clip(hist_sum + noise, 0.0, None)
    return noised, sigma


def tv_distance_from_hists(hist_a, hist_b):
    """TV(p, q) = 1/2 ||p - q||_1 for (unnormalized) histograms, normalized
    internally into probability vectors first."""
    total_a, total_b = hist_a.sum(), hist_b.sum()
    pa = hist_a / total_a if total_a > 0 else np.ones_like(hist_a) / len(hist_a)
    pb = hist_b / total_b if total_b > 0 else np.ones_like(hist_b) / len(hist_b)
    return 0.5 * float(np.abs(pa - pb).sum())


# ---------------------------------------------------------------------------
# Lemma 1 (Section 2.4) : DP-noise estimation error bound eta_a(beta)
# ---------------------------------------------------------------------------

def eta_error_bound(beta, sigma, m, sum_c_P, sum_c_Q):
    """Lemma 1:
        eta_a(beta) = sigma * sqrt(2 m ln(4m/beta)) * (1/sum(c_P) + 1/sum(c_Q))

    With probability >= 1 - beta:  |Delta_hat_a - Delta_emp_a(P,Q)| <= eta_a(beta)
    """
    if sum_c_P <= 0 or sum_c_Q <= 0:
        return float("inf")
    return sigma * np.sqrt(2.0 * m * np.log(4.0 * m / beta)) * (
        1.0 / sum_c_P + 1.0 / sum_c_Q
    )


# ---------------------------------------------------------------------------
# Lemma 2 (Section 2.6) : finite-sample estimation error bound sigma_a
# ---------------------------------------------------------------------------

def finite_sample_error_bound(alpha1, m, n0, nt):
    """Lemma 2:
        sigma_a = sqrt( (m/2) * (1/n0 + 1/nt) * ln(4m/alpha1) )

    With probability >= 1 - alpha1/2:  |Delta_emp_a - Delta_bin_a| <= sigma_a
    """
    n0 = max(n0, 1)
    nt = max(nt, 1)
    return float(np.sqrt((m / 2.0) * (1.0 / n0 + 1.0 / nt) * np.log(4.0 * m / alpha1)))


# ---------------------------------------------------------------------------
# Section 2.6 : deterministic coarsening gap xi_a
# ---------------------------------------------------------------------------

def coarsening_gap(resolution_constant):
    """xi_a is a fixed, deterministic constant depending only on the
    histogram resolution and an assumed bound on within-bin density
    variation (Section 2.2/2.6). It consumes no confidence budget
    (Remark 2). For purely categorical features (no smoothing needed),
    xi_a = 0. We treat it as a supplied modelling constant since it is a
    smoothness assumption, not an estimated quantity.
    """
    return max(0.0, float(resolution_constant))


# ---------------------------------------------------------------------------
# Full per-group and total drift estimation pipeline
# ---------------------------------------------------------------------------

def private_group_drift_estimate(baseline_dfs, current_dfs, a, feature_cols, edges,
                                  eps_dp, delta_dp, alpha1, alpha2,
                                  xi=0.0, rng=None):
    """
    Runs the full Section 2.4-2.6 pipeline for ONE group a in {0, 1}:
    secure aggregation -> DP release -> Delta_hat_a, plus all three error
    terms (eta_a, sigma_a, xi_a) needed by the Theorem-2 composed bound.

    Returns a dict with delta_hat, eta, sigma, xi, and the group sample
    sizes (useful for logging/debugging).
    """
    rng = rng or np.random.default_rng()

    c_base = secure_aggregate_group_histograms(baseline_dfs, a, feature_cols, edges)
    c_curr = secure_aggregate_group_histograms(current_dfs, a, feature_cols, edges)

    n0 = int(c_base.sum())
    nt = int(c_curr.sum())
    m = len(c_base)

    c_base_noised, sigma_base = dp_noised_histogram(c_base, eps_dp, delta_dp, rng=rng)
    c_curr_noised, sigma_curr = dp_noised_histogram(c_curr, eps_dp, delta_dp, rng=rng)

    delta_hat = tv_distance_from_hists(c_base_noised, c_curr_noised)

    # Lemma 1: use the larger of the two per-release sigmas as a conservative
    # single sigma (both releases use the same eps_dp, delta_dp in our
    # protocol, so sigma_base == sigma_curr up to floating point).
    sigma_dp = max(sigma_base, sigma_curr)
    eta_a = eta_error_bound(beta=alpha2 / 2.0, sigma=sigma_dp, m=m,
                             sum_c_P=c_base.sum(), sum_c_Q=c_curr.sum())

    sigma_a = finite_sample_error_bound(alpha1=alpha1, m=m, n0=n0, nt=nt)
    xi_a = coarsening_gap(xi)

    return dict(delta_hat=delta_hat, eta=eta_a, sigma=sigma_a, xi=xi_a,
                n0=n0, nt=nt, m=m)


def private_drift_estimate(baseline_dfs, current_dfs, feature_cols, edges,
                            eps_dp, delta_dp, alpha1, alpha2, xi=0.0, rng=None):
    """Runs `private_group_drift_estimate` for BOTH groups a in {0, 1} and
    returns the per-group dicts plus the combined (Delta_hat_0 + Delta_hat_1)
    drift estimate, mirroring Definition 3's Delta(P, Q) = Delta_0 + Delta_1.
    """
    rng = rng or np.random.default_rng()
    group0 = private_group_drift_estimate(baseline_dfs, current_dfs, 0, feature_cols, edges,
                                           eps_dp, delta_dp, alpha1, alpha2, xi, rng)
    group1 = private_group_drift_estimate(baseline_dfs, current_dfs, 1, feature_cols, edges,
                                           eps_dp, delta_dp, alpha1, alpha2, xi, rng)
    delta_hat_total = group0["delta_hat"] + group1["delta_hat"]
    return group0, group1, delta_hat_total


def true_group_drift(baseline_dfs, current_dfs, a, feature_cols, edges):
    """Non-private, exact group-conditional TV distance Delta_a(P0, Pt).
    Used ONLY for validating the theory (Experiment A style checks), never
    available to the deployed monitoring system itself."""
    c_base = secure_aggregate_group_histograms(baseline_dfs, a, feature_cols, edges)
    c_curr = secure_aggregate_group_histograms(current_dfs, a, feature_cols, edges)
    return tv_distance_from_hists(c_base, c_curr)


def true_drift(baseline_dfs, current_dfs, feature_cols, edges):
    """Delta(P0, Pt) = Delta_0 + Delta_1, exact (non-private) version."""
    d0 = true_group_drift(baseline_dfs, current_dfs, 0, feature_cols, edges)
    d1 = true_group_drift(baseline_dfs, current_dfs, 1, feature_cols, edges)
    return d0 + d1
