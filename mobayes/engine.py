"""Minimal Bayesian belief engine for diagnostic dialogue.

Mirrors the math of MoBayes (arXiv:2604.20022, §3) on a single discrete
disease universe with discrete-valued features. ~120 LoC for teaching.
"""
from __future__ import annotations
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

_EPS = 1e-20

# ---- Soft-evidence vocabulary (paper §3.2, Eq. 1-4) ---------------------
_CONFIDENCE: Dict[str, float] = {
    'very_likely':   1.00,
    'likely':        0.80,
    'uncertain':     0.50,
    'unlikely':      0.25,
    'very_unlikely': 0.05,
}

def resolve_confidence(label):
    """Map a soft-evidence label (or raw float in [0,1]) to a Jeffrey weight."""
    if isinstance(label, (int, float)):
        return float(max(0.0, min(1.0, label)))
    return _CONFIDENCE.get(str(label).strip().lower(), 1.0)


# ---- Knowledge base loader -----------------------------------------------
def _parse_values(raw):
    """features.csv stores values as a JSON-encoded list string."""
    if isinstance(raw, list):
        return raw
    if pd.isna(raw):
        return []
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else [v]
    except (json.JSONDecodeError, TypeError):
        return [raw]


def load_kb(data_dir: str) -> 'WorldModel':
    """Load the four-file KB (diseases, features, priors, likelihood_counts)
    and return an initialised WorldModel.

    Schema (matches DDxPlus / AgentClinic kbs/ format):
      diseases.csv         : disease_id, parent_id, ...
      features.csv         : feature_id, name, type, values, ...
      priors.csv           : disease_id, prior
      likelihood_counts.csv: disease_id, feature_id, value, count
    """
    dd = pd.read_csv(os.path.join(data_dir, 'diseases.csv'))
    df = pd.read_csv(os.path.join(data_dir, 'features.csv'))
    dp = pd.read_csv(os.path.join(data_dir, 'priors.csv'))
    dl = pd.read_csv(os.path.join(data_dir, 'likelihood_counts.csv'))

    # Leaves only: drop nodes that appear as anyone's parent_id.
    parents = set(dd['parent_id'].dropna().unique()) if 'parent_id' in dd.columns else set()
    leaves = [d for d in dd['disease_id'].tolist() if d not in parents]
    d2i = {d: i for i, d in enumerate(leaves)}
    K = len(leaves)

    # Prior vector
    prior = np.zeros(K, dtype=float)
    for _, r in dp.iterrows():
        if r['disease_id'] in d2i:
            prior[d2i[r['disease_id']]] = float(r['prior'])
    s = prior.sum()
    prior = prior / s if s > 0 else np.ones(K) / K

    # Schema: feature_id -> dict(name, type, values)
    schema: Dict[str, dict] = {}
    for _, r in df.iterrows():
        fid = r['feature_id']
        schema[fid] = {
            'name': r['name'],
            'type': r.get('type', 'binary'),
            'values': _parse_values(r.get('values', '[]')),
        }

    # Dirichlet-Categorical likelihoods: P(feature=v | disease) =
    #   (count + 1) / (sum_v count + |values|)
    counts: Dict[str, Dict[str, np.ndarray]] = {fid: {} for fid in schema}
    for _, r in dl.iterrows():
        fid, v, did = r['feature_id'], str(r['value']), r['disease_id']
        if fid in counts and did in d2i:
            counts[fid].setdefault(v, np.zeros(K))[d2i[did]] += float(r['count'])

    likelihood: Dict[str, Dict[str, np.ndarray]] = {}
    for fid, vs in counts.items():
        values = [str(v) for v in schema[fid]['values']]
        if not values:
            values = sorted(vs.keys())
        # zero-init every disease/value pair, then fill
        like = {v: np.zeros(K) for v in values}
        for v in values:
            if v in vs:
                like[v] = vs[v].copy()
        # Laplace +1 smoothing, normalise per disease
        totals = sum(like[v] for v in values) + len(values)
        for v in values:
            like[v] = (like[v] + 1.0) / totals
        likelihood[fid] = like

    return WorldModel(prior, leaves, schema, likelihood)


# ---- Engine --------------------------------------------------------------
class WorldModel:
    """Belief over diseases + closed-form Jeffrey-style soft update."""

    def __init__(self, prior: np.ndarray, leaves: List[str],
                 schema: dict, likelihood: dict):
        self.prior = prior.astype(float).copy()
        self.leaves = list(leaves)
        self.schema = schema
        self.likelihood = likelihood
        self.num_diseases = len(leaves)
        self.belief = self.prior.copy()
        self.asked: set = set()

    # ---- belief access ----
    def posterior(self) -> np.ndarray:
        return self.belief.copy()

    def entropy(self) -> float:
        return self.compute_posterior_entropy(self.belief)

    @staticmethod
    def compute_posterior_entropy(p: np.ndarray) -> float:
        p = p[p > _EPS]
        return float(-np.sum(p * np.log2(p)))

    def unasked_features(self) -> List[str]:
        return [f for f in self.schema if f not in self.asked]

    # ---- closed-form update ----
    def _likelihood_vec(self, fid: str, value: str) -> np.ndarray:
        return self.likelihood[fid].get(str(value), np.ones(self.num_diseases) / self.num_diseases)

    def update(self, fid: str, value, confidence: float = 1.0) -> None:
        """Jeffrey conditioning with soft evidence weight c in [0, 1].

        c=1   -> standard Bayes:  b *= P(v|d) ; renormalise
        c=0   -> no-op           (evidence ignored)
        else  -> mix:  b *= c*P(v|d) + (1-c)*uniform_over_values
        """
        c = float(np.clip(confidence, 0.0, 1.0))
        L = self._likelihood_vec(fid, value)
        if c < 1.0:
            neutral = np.ones(self.num_diseases) / self.num_diseases
            L = c * L + (1.0 - c) * neutral
        new = self.belief * L
        s = new.sum()
        self.belief = new / s if s > 0 else self.belief
        self.asked.add(fid)

    # ---- predictive distribution over feature values ----
    def predict_feature_distribution(self, fid: str) -> Dict[str, float]:
        """P(X_f = v | E_t) = sum_d P(v|d) * b_t(d)."""
        dist = {}
        for v, L in self.likelihood.get(fid, {}).items():
            dist[v] = float(np.sum(L * self.belief))
        return dist

    def simulate_update(self, fid: str, value) -> np.ndarray:
        """Counterfactual: posterior IF (fid, value) were observed, no asking."""
        L = self._likelihood_vec(fid, value)
        new = self.belief * L
        s = new.sum()
        return new / s if s > 0 else self.belief


# ---- EIG sweep -----------------------------------------------------------
def eig_over(wm: WorldModel,
             candidates: Optional[Iterable[str]] = None) -> Tuple[Optional[str], float]:
    """Pick the feature with highest expected information gain.

    EIG(f) = H(b_t) - E_v[H(b_{t+1} | X_f = v)]
           = current_entropy - sum_v P(v|E_t) * H(simulate(f, v))
    """
    if candidates is None:
        candidates = wm.unasked_features()
    cur_h = wm.entropy()
    best_fid, best_eig = None, -np.inf
    for fid in candidates:
        pred = wm.predict_feature_distribution(fid)
        exp_h = 0.0
        for v, p_v in pred.items():
            if p_v > 1e-9:
                sim = wm.simulate_update(fid, v)
                exp_h += p_v * wm.compute_posterior_entropy(sim)
        eig = cur_h - exp_h
        if eig > best_eig:
            best_fid, best_eig = fid, eig
    return best_fid, best_eig


# ---- Stopping ------------------------------------------------------------
class StoppingRules:
    """τ-thresholded commit + hard budget ceiling + warm-up floor."""

    def __init__(self, confidence_threshold: float = 0.85,
                 max_questions: int = 8,
                 min_questions_before_abstain: int = 0):
        self.tau = float(confidence_threshold)
        self.max_q = int(max_questions)
        self.min_q = int(min_questions_before_abstain)

    def should_stop(self, wm: WorldModel, num_questions: int) -> Tuple[bool, str]:
        if num_questions >= self.max_q:
            return True, 'MAX_QUESTIONS'
        if num_questions >= self.min_q and np.max(wm.posterior()) >= self.tau:
            return True, 'CONFIDENT'
        return False, 'CONTINUE'
