"""Lightweight inline MoBayes engine for the ACL 2026 tutorial notebook.

Self-contained: no external imports beyond numpy / pandas. Re-implements the
minimum surface area the §6 audit demo needs:

  - load_kb(data_dir)              -> WorldModel  (Dirichlet-Categorical likelihoods)
  - WorldModel.update(fid, v, c)   -> in-place Jeffrey-style soft update
  - WorldModel.posterior()         -> current b_t
  - WorldModel.entropy()           -> H(b_t) in bits
  - WorldModel.predict_feature_distribution(fid) -> {value: P(value | E_t)}
  - WorldModel.simulate_update(fid, v) -> counterfactual posterior
  - resolve_confidence(label)      -> Jeffrey weight in [0, 1]
  - eig_over(wm, candidates)       -> (best_fid, best_eig)
  - StoppingRules(...).should_stop(num_q) -> (done, reason)

The full MoBayes engine (taxonomy, hierarchical EIG, top-k focused EIG, etc.)
lives at https://github.com/<anon>/MoBayes ; this file is a teaching subset.
"""
from .engine import (
    load_kb,
    WorldModel,
    StoppingRules,
    resolve_confidence,
    eig_over,
)

__all__ = [
    'load_kb', 'WorldModel', 'StoppingRules', 'resolve_confidence', 'eig_over',
]
