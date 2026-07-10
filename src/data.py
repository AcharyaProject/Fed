"""
data.py
=======
Data loading for the federated fairness recertification experiments.

Two sources are supported:

1. REAL DATA (recommended for the paper): Folktables / ACS PUMS.
   Folktables provides real US Census microdata, naturally indexed by
   STATE and YEAR -- exactly the two axes of real, non-synthetic
   population drift we want (geographic drift across states, temporal
   drift across years). This requires internet access to census.gov,
   which is NOT available in this sandboxed environment, so the loader
   below will attempt it and gracefully fall back if it fails.

   To run with real data on your own machine:
       pip install folktables
       python run_experiment.py --data real

2. SYNTHETIC FALLBACK (used automatically here): a generator that
   mimics the structure of the ACS Income task (age, education,
   occupation code, hours worked, protected attribute = sex or race,
   label = income > $50k) with an explicit, controllable drift
   parameter so the whole pipeline is reproducible without internet
   access. This is what actually runs in this sandbox; swap in (1)
   for the real experiments in the paper.
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. REAL DATA LOADER (Folktables / ACS PUMS)
# ----------------------------------------------------------------------
def load_folktables(states, year, task="income", root_dir=None):
    """
    Loads real ACS PUMS data for a list of states and a given year using
    the `folktables` package. Requires internet access to census.gov.

    Parameters
    ----------
    states : list[str]   e.g. ["CA", "TX", "NY"]
    year   : str          e.g. "2018"
    task   : str           one of {"income", "employment", "coverage"}
    root_dir : str        directory to cache downloaded data (defaults to ./data)

    Returns
    -------
    dict[state] -> pandas.DataFrame with columns:
        features..., 'A' (protected attribute), 'Y' (label)
    """
    import os
    from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage

    if root_dir is None:
        root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(root_dir, exist_ok=True)

    task_map = {
        "income": ACSIncome,
        "employment": ACSEmployment,
        "coverage": ACSPublicCoverage,
    }
    acs_task = task_map[task]

    # ACSIncome's native feature columns are AGEP, COW, SCHL, MAR, OCCP,
    # POBP, RELP, WKHP, SEX, RAC1P. We rename the ones we use to match
    # our internal feature names (age, education, hours, occ_code) so
    # fairness.py / drift.py / bound.py work unchanged on real data.
    rename_map = {"AGEP": "age", "SCHL": "education", "WKHP": "hours", "OCCP": "occ_code"}

    data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person",
                                 root_dir=root_dir)
    out = {}
    for st in states:
        acs_data = data_source.get_data(states=[st], download=True)
        features, label, group = acs_task.df_to_pandas(acs_data)
        df = features.rename(columns=rename_map)[list(rename_map.values())].copy()
        df["A"] = group.values.ravel()
        df["Y"] = label.values.ravel().astype(int)
        out[st] = df.reset_index(drop=True)
    return out


# ----------------------------------------------------------------------
# 1b. REAL DATA LOADER (FLamby -- Fed-Heart-Disease)
# ----------------------------------------------------------------------
FLAMBY_HEART_FEATURE_COLS = ["age", "trestbps", "chol", "thalach"]


def load_flamby_heart_disease():
    """
    Loads the real FLamby Fed-Heart-Disease dataset: 4 real hospital
    sites (natural, not synthetic, cross-silo partitions) from the
    classic UCI Heart Disease data. Requires FLamby to be installed
    and the dataset already downloaded via FLamby's own download script
    (see README / setup instructions) -- this function does NOT trigger
    the download itself, since FLamby's download script needs to be run
    once, standalone, per its own licensing/setup flow.

    Protected attribute: sex (A=1 male, A=0 female) -- a real, commonly
    studied fairness axis in cardiac risk prediction.
    Label: presence of heart disease (Y=1) vs absence (Y=0).
    Features: age, resting blood pressure (trestbps), cholesterol (chol),
    max heart rate achieved (thalach).

    COLUMN ORDER: FLamby's Fed-Heart-Disease preserves the standard UCI
    Heart Disease (Cleveland/Hungarian/Switzerland/VA) 13-feature column
    order: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
    oldpeak, slope, ca, thal (index 0-12) -- this is the order documented
    across UCI-derived heart-disease reproductions and is what the
    previous version of this function assumed too (correctly, as far as
    the standard UCI ordering goes). The change here is NOT the indices
    themselves but that we now verify the assumption at load time instead
    of silently trusting it: since sex is binary in the raw data (0/1),
    we assert that column actually IS binary before using it as A, so a
    wrong index (e.g. off-by-one from a differently-preprocessed FLamby
    version) fails loudly here rather than silently producing a nonsense
    protected attribute and a meaningless fairness number downstream.
    """
    from flamby.datasets.fed_heart_disease import FedHeartDisease

    SEX_COL_IDX = 1  # standard UCI order; verified at runtime below

    out = {}
    for center in range(4):  # FLamby Fed-Heart-Disease has 4 real centers
        train_ds = FedHeartDisease(center=center, train=True)
        rows, labels = [], []
        for i in range(len(train_ds)):
            x, y = train_ds[i]
            rows.append(x.numpy().ravel())
            labels.append(y.item())
        X = np.array(rows)
        Y = np.array(labels).astype(int)

        sex_col = X[:, SEX_COL_IDX]
        uniq = np.unique(np.round(sex_col))
        if not set(uniq.tolist()).issubset({0.0, 1.0}):
            raise ValueError(
                f"load_flamby_heart_disease: column index {SEX_COL_IDX} (assumed "
                f"to be 'sex') is not binary for center {center} (unique values: "
                f"{uniq}). The installed FLamby version's feature ordering or "
                f"normalization may differ from the standard UCI order this "
                f"loader assumes -- verify against your local "
                f"flamby.datasets.fed_heart_disease.dataset before trusting "
                f"downstream fairness numbers."
            )

        df = pd.DataFrame({
            "age": X[:, 0], "trestbps": X[:, 3],
            "chol": X[:, 4], "thalach": X[:, 7],
        })
        df["A"] = sex_col.round().astype(int)
        df["Y"] = Y
        out[f"center_{center}"] = df
    return out


# ----------------------------------------------------------------------
# 2. SYNTHETIC ACS-LIKE GENERATOR (offline fallback, drift-controllable)
# ----------------------------------------------------------------------
def _make_client_distribution(rng, base_params, drift=0.0):
    """
    Returns a dict of distribution parameters for one client, obtained by
    shifting `base_params` by an amount controlled by `drift` in [0, 1].
    drift=0 reproduces the certified baseline distribution exactly;
    drift=1 is a strong, deliberately injected shift.

    Both the protected-attribute base rate AND the strength of the proxy
    correlation (proxy_gap_*) are allowed to shift with drift -- e.g. a
    new client/site joining with both a different demographic mix and a
    different occupational/hours structure, which is what actually moves
    demographic parity for a model that excludes A but has learned its
    proxies.
    """
    p = dict(base_params)
    p["p_A1"] = np.clip(base_params["p_A1"] + drift * base_params["drift_dir_A"], 0.05, 0.95)
    p["age_mean"] = base_params["age_mean"] + drift * base_params["drift_dir_age"]
    p["educ_mean"] = base_params["educ_mean"] + drift * base_params["drift_dir_educ"]
    p["hours_mean"] = base_params["hours_mean"] + drift * base_params["drift_dir_hours"]
    p["proxy_gap_age"] = base_params["proxy_gap_age"] + drift * base_params["drift_dir_proxy_age"]
    p["proxy_gap_educ"] = base_params["proxy_gap_educ"] + drift * base_params["drift_dir_proxy_educ"]
    p["proxy_gap_hours"] = base_params["proxy_gap_hours"] + drift * base_params["drift_dir_proxy_hours"]
    return p


def generate_synthetic_client(rng, n, params, label_rule="income"):
    """
    Generates one client's data (age, education, hours, occ_code, A, Y).

    Proxy-discrimination design: A is drawn first, and age/education/hours
    are then generated as A-conditional draws (mirroring real proxy
    structure in tasks like ACS Income, where hours-worked/occupation
    correlate with sex/race even though the model never sees A directly).
    This is what makes demographic-parity violations possible for a model
    trained WITHOUT A as a feature -- with A-independent features, the
    model would trivially satisfy demographic parity regardless of drift.

    The label is generated by a fixed ground-truth rule that does not
    depend on drift parameters directly (only through the A-conditional
    feature means), so measured fairness change is attributable to the
    changing population, not to the label rule changing.
    """
    A = rng.binomial(1, params["p_A1"], size=n)

    age_mean_0 = params["age_mean"] - params["proxy_gap_age"] / 2
    age_mean_1 = params["age_mean"] + params["proxy_gap_age"] / 2
    educ_mean_0 = params["educ_mean"] - params["proxy_gap_educ"] / 2
    educ_mean_1 = params["educ_mean"] + params["proxy_gap_educ"] / 2
    hours_mean_0 = params["hours_mean"] - params["proxy_gap_hours"] / 2
    hours_mean_1 = params["hours_mean"] + params["proxy_gap_hours"] / 2

    age = np.where(A == 1, rng.normal(age_mean_1, 10, size=n),
                   rng.normal(age_mean_0, 10, size=n)).clip(18, 75)
    educ = np.where(A == 1, rng.normal(educ_mean_1, 2.5, size=n),
                    rng.normal(educ_mean_0, 2.5, size=n)).clip(1, 20)
    hours = np.where(A == 1, rng.normal(hours_mean_1, 8, size=n),
                     rng.normal(hours_mean_0, 8, size=n)).clip(5, 80)
    occ_code = rng.integers(1, 12, size=n)

    # Fixed ground-truth rule, deliberately does not reference A directly;
    # any disparity in predictions arises through the proxy features above.
    z = (
        0.05 * (educ - 10)
        + 0.03 * (age - 40)
        + 0.06 * (hours - 40)
        + 0.10 * (occ_code % 3)
    )
    prob = 1 / (1 + np.exp(-z))
    Y = rng.binomial(1, prob)

    df = pd.DataFrame({
        "age": age, "education": educ, "hours": hours,
        "occ_code": occ_code, "A": A, "Y": Y,
    })
    return df


def base_acs_like_params(seed_state):
    """Fixed 'true' baseline distribution parameters per synthetic state."""
    rng = np.random.default_rng(abs(hash(seed_state)) % (2**32))
    return {
        "p_A1": rng.uniform(0.35, 0.55),
        "drift_dir_A": rng.uniform(-0.3, 0.3),
        "age_mean": rng.uniform(35, 45),
        "drift_dir_age": rng.uniform(-8, 8),
        "educ_mean": rng.uniform(9, 13),
        "drift_dir_educ": rng.uniform(-3, 3),
        "hours_mean": rng.uniform(35, 42),
        "drift_dir_hours": rng.uniform(-6, 6),
        # Baseline proxy correlation strength (real-world analogue: e.g.
        # average weekly hours differing by group due to occupational
        # segregation). Certified baseline has a modest, realistic gap;
        # drift can substantially widen or narrow it per client.
        "proxy_gap_age": rng.uniform(1, 4),
        "drift_dir_proxy_age": rng.uniform(2, 10),
        "proxy_gap_educ": rng.uniform(0.5, 1.5),
        "drift_dir_proxy_educ": rng.uniform(1, 4),
        "proxy_gap_hours": rng.uniform(2, 5),
        "drift_dir_proxy_hours": rng.uniform(6, 16),
    }


def make_synthetic_federation(states, n_per_client=4000, seed=0):
    """
    Builds the CERTIFIED baseline federation (drift=0 for every client).
    Returns dict[state] -> DataFrame.
    """
    rng = np.random.default_rng(seed)
    fed = {}
    for st in states:
        params = base_acs_like_params(st)
        fed[st] = generate_synthetic_client(rng, n_per_client, params)
    return fed


def make_drifted_snapshot(states, drift_schedule, n_per_client=4000, seed=0):
    """
    Builds a snapshot of the federation at a given point in (drift) time.

    drift_schedule : dict[state] -> drift value in [0, 1]
        0.0 = identical to the certified baseline population
        1.0 = maximally shifted synthetic population for that state

    Returns dict[state] -> DataFrame
    """
    rng = np.random.default_rng(seed)
    fed = {}
    for st in states:
        base_params = base_acs_like_params(st)
        drifted_params = _make_client_distribution(rng, base_params, drift=drift_schedule.get(st, 0.0))
        fed[st] = generate_synthetic_client(rng, n_per_client, drifted_params)
    return fed
