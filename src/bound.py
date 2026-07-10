"""
bound.py
========
Implements:
  1. The deterministic fairness-stability bound (Theorem 1, exact):
        |F(Q,h) - F(P,h)| <= Delta_0(P,Q) + Delta_1(P,Q)
     with NO Lipschitz constant and NO feature-norm scaling. Theorem 1
     needs only that h : R^d -> [0,1] is bounded (Section 2.2), and its
     proof (the variational characterization of TV) gives coefficient 1,
     not 2*L*B. Remark 1 in the paper explicitly rejects a Lipschitz/
     Wasserstein-based alternative for exactly this reason: it would need
     a private estimator for a harder statistic, defeating the whole
     point of using TV + secure aggregation in the first place.

  2. The privacy-composed bound (Theorem 2, exact, all four terms):
        F(P_t, h) <= eps0 + sum_{a in {0,1}} (Delta_hat_a + xi_a + sigma_a + eta_a)
     i.e. Eq. (2)-(3): the deterministic coarsening gap (xi), the
     finite-sample estimation error (sigma), and the DP-noise error (eta)
     are ALL required per group -- not just delta_hat + eta.

  3. Section 2.8's confidence-budget (Bonferroni) and privacy-budget
     (composition) splitting across a T-round monitoring horizon, which
     Corollary 1's lifecycle-wide guarantee depends on.

  4. The recertification trigger and lifecycle state machine (Section 2.7 /
     Theorem 3): Certified / Monitored / Uncertain / Expired / Recertified,
     not just a boolean "audit now" flag.

*** This is a corrected version of the original bound.py, which used a
Lipschitz/feature-norm envelope `g(delta) = 2*L*B*delta` (Theorem-1-style
but NOT Theorem 1), and a two-term `dp0 + g(delta_hat + eta)` composed bound
that omitted xi and sigma (NOT Theorem 2). See the accompanying chat
message for the itemized diff. ***
"""

import math
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Theorem 1 : deterministic fairness-stability bound (boundedness only)
# ---------------------------------------------------------------------------

def fairness_stability_bound(delta0, delta1):
    """Theorem 1:  |F(Q,h) - F(P,h)| <= Delta_0(P,Q) + Delta_1(P,Q).

    delta0, delta1 are the (exact or estimated) group-conditional TV
    distances from drift.py. No Lipschitz constant, no feature norm --
    just the sum, exactly as proven.
    """
    return delta0 + delta1


def deterministic_bound(eps0, delta0_true, delta1_true):
    """Theorem-1 bound using the TRUE (non-private) group-conditional
    drifts -- used only for validating the theory (Experiment A), never
    available to the deployed system itself, which only ever sees the
    privacy-composed estimate below."""
    return eps0 + fairness_stability_bound(delta0_true, delta1_true)


# ---------------------------------------------------------------------------
# Theorem 2 : composed privacy-aware certification bound B_t (all 4 terms)
# ---------------------------------------------------------------------------

def group_bound_term(group_components: dict) -> float:
    """One group's contribution to B_t: Delta_hat_a + xi_a + sigma_a + eta_a.

    `group_components` is the dict returned by
    drift.private_group_drift_estimate (keys: delta_hat, xi, sigma, eta).
    """
    return (group_components["delta_hat"] + group_components["xi"]
            + group_components["sigma"] + group_components["eta"])


def composed_certification_bound(eps0, group0_components, group1_components):
    """Theorem 2 (Eq. 2-3):

        B_t = eps0 + (Delta_hat_0 + xi_0 + sigma_0 + eta_0)
                    + (Delta_hat_1 + xi_1 + sigma_1 + eta_1)

    Valid with probability >= 1 - alpha1 - alpha2 (union bound over the two
    groups' finite-sample and DP-noise failure events; see the Theorem-2
    proof in Section 2.5).
    """
    return eps0 + group_bound_term(group0_components) + group_bound_term(group1_components)


def private_bound(eps0, group0_components, group1_components):
    """Alias kept for call-site compatibility with the previous API name."""
    return composed_certification_bound(eps0, group0_components, group1_components)


# ---------------------------------------------------------------------------
# Section 2.8 : confidence-budget (Bonferroni) and privacy-budget composition
# ---------------------------------------------------------------------------

def per_round_confidence_budgets(alpha1_total, alpha2_total, horizon_T):
    """Corollary 1 / Section 2.8: simple Bonferroni split of the total
    confidence budgets across the T-round monitoring horizon.

        alpha1_t = alpha1_total / T
        alpha2_t = alpha2_total / T

    This is what makes the lifecycle-wide guarantee
        Pr[ exists t <= T : tau_t = 0 and F(Pt,h) > eps_max ] <= alpha1_total + alpha2_total
    hold via a union bound over rounds (Corollary 1's proof).
    """
    return alpha1_total / horizon_T, alpha2_total / horizon_T


def per_round_privacy_budget(eps_total, delta_total, horizon_T, mode="basic"):
    """Section 2.8: derive a per-round (eps_dp, delta_dp) from a total
    T-round privacy budget (eps_total, delta_total), or use a flat
    per-round constant directly.

    mode="flat": use (eps_total, delta_total) unchanged as the per-round
        budget, ignoring T. This is what the paper's own Table 1 reports
        ("DP privacy budget eps_dp = 2.0") -- a deployer may simply fix a
        constant per-round budget rather than composing a total budget
        across the horizon; Section 2.8 frames privacy and confidence as
        "two distinct and independently adjustable dials", and this is
        the simpler of the two ways to set the privacy dial.
    mode="basic": basic composition, eps_dp = eps_total / T (conservative;
        matches "accessing the estimator once per round at constant
        (eps_dp, delta_dp) across T rounds combines... to (T*eps_dp, T*delta_dp)"
        read in reverse to solve for the per-round budget).
    mode="advanced": the paper's stated advanced/Renyi-DP-accountant-style
        approximation, eps_dp ~= eps_total / sqrt(T * ln(1/delta_total)),
        which permits a looser (larger, i.e. less noisy) per-round budget
        for the same total (eps_total, delta_total) -- consistent with
        Section 2.8's remark that "deterioration is nearer to
        O(eps_dp * sqrt(T ln(1/delta)))" under an enhanced-composition
        accountant.

    All three are legitimate readings of Section 2.8 / Table 1; which one
    a deployer should use depends on whether they want a fixed per-round
    privacy cost (flat) or a fixed total privacy cost over the horizon
    (basic/advanced). We expose all three rather than picking one silently.
    """
    if mode == "flat":
        return eps_total, delta_total
    delta_dp = delta_total / horizon_T
    if mode == "basic":
        eps_dp = eps_total / horizon_T
    elif mode == "advanced":
        eps_dp = eps_total / math.sqrt(horizon_T * math.log(1.0 / delta_total))
    else:
        raise ValueError(f"Unknown privacy composition mode: {mode!r}")
    return eps_dp, delta_dp


# ---------------------------------------------------------------------------
# Section 2.7 / Theorem 3 : recertification trigger and certificate lifecycle
# ---------------------------------------------------------------------------

class CertificateState(Enum):
    CERTIFIED = "Certified"
    MONITORED = "Monitored"
    UNCERTAIN = "Uncertain"
    EXPIRED = "Expired"
    RECERTIFIED = "Recertified"


def recertification_trigger(eps0, group0_components, group1_components,
                             eps_max, gamma=0.0):
    """Section 2.7:

        B_t   = composed_certification_bound(eps0, group0, group1)
        tau_t = 1[B_t > eps_max]
        state = Expired    if tau_t == 1
                Uncertain   if tau_t == 0 and B_t > eps_max - gamma
                Monitored   otherwise

    Theorem 3: if tau_t == 0 (state is Monitored or Uncertain, i.e. NOT
    Expired) then Pr[F(Pt,h) <= eps_max] >= 1 - alpha1 - alpha2, at the
    confidence levels used to build B_t's components.

    Returns (triggered: bool, B_t: float, state: CertificateState).
    """
    B_t = composed_certification_bound(eps0, group0_components, group1_components)
    if B_t > eps_max:
        return True, B_t, CertificateState.EXPIRED
    if B_t > eps_max - gamma:
        return False, B_t, CertificateState.UNCERTAIN
    return False, B_t, CertificateState.MONITORED
