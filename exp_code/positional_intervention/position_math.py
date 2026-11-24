"""Host-side two-row probability relocation; no sampler or temperature changes."""
import math
import numpy as np


def logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return -math.inf
    maximum = float(finite.max())
    return maximum + math.log(float(np.exp(finite - maximum).sum()))


def probabilities(logits):
    """Normalize across the complete vocabulary, not just top-k candidates."""
    x = np.asarray(logits, dtype=np.float64)
    if np.isnan(x).any() or np.isposinf(x).any():
        raise ValueError('NaN/+inf logits')
    maximum = x.max(axis=-1, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ValueError('Row has no allowed tokens')
    p = np.exp(x - maximum)
    return p / p.sum(axis=-1, keepdims=True)


def relocate_rows(rows, token_id, fraction):
    """Return requested logits and diagnostics; input rows are never mutated.

    delta = fraction * min(p_source(v), 1-p_target(v)). Preserve the
    conditional distribution over all other allowed tokens within each row.
    Keep the old logit gauge to limit BF16 rounding. Forbidden -inf entries
    stay forbidden. A zero-mass operation returns the original object exactly.
    """
    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 2 or x.shape[1] < 2:
        raise ValueError('Expected two vocabulary rows')
    if not isinstance(token_id, (int, np.integer)) or not 0 <= token_id < x.shape[1]:
        raise ValueError('Invalid token ID')
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('fraction must be finite and in [0,1]')
    before = probabilities(x)
    if not np.isfinite(x[:, token_id]).all():
        raise ValueError('Anchor is forbidden at source or destination')
    other = np.arange(x.shape[1]) != token_id
    normalizers = [logsumexp(row) for row in x]
    tail_logs = [logsumexp(row[other]) for row in x]
    tail = np.exp(np.array(tail_logs) - normalizers)
    old = before[:, token_id]
    delta = float(fraction * min(old[0], tail[1]))
    info = dict(fraction=float(fraction), requested_delta=delta,
                anchor_before=old.tolist())
    if delta == 0:
        return rows, dict(info, requested_anchor_after=old.tolist(), exact_noop=True)
    if not np.isfinite(tail_logs[0]):
        raise ValueError('Source has no allowed non-anchor support to receive mass')
    new_anchor = np.array([max(0., old[0]-delta), min(1., old[1]+delta)])
    new_tail = np.array([min(1., tail[0]+delta), max(0., tail[1]-delta)])
    out = x.copy()
    for i in range(2):
        if new_tail[i] == 0:
            out[i, other] = -np.inf
        else:
            out[i, other] += math.log(new_tail[i]) + normalizers[i] - tail_logs[i]
        out[i, token_id] = (math.log(new_anchor[i]) + normalizers[i]
                            if new_anchor[i] > 0 else -np.inf)
    return out, dict(info, requested_anchor_after=new_anchor.tolist(), exact_noop=False)


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    positive = p > 0
    return float(-np.sum(p[positive] * np.log(p[positive])))
