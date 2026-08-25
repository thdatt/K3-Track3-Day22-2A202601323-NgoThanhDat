from __future__ import annotations

from typing import cast

import numpy as np


def _log_sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Numerically stable log-sigmoid via logaddexp (no overflow at -inf)."""
    arr = np.asarray(x, dtype=float)
    result = -np.logaddexp(0.0, -arr)
    return cast(np.ndarray, result)


def dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float,
) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    DPO re-expresses the Bradley-Terry preference model in terms of the
    implicit reward ``r = log(policy/ref)``. The loss is the negative
    log-likelihood of the preference (chosen > rejected):

        loss = -log_sigmoid(beta * ((policy_chosen - ref_chosen)
                                    - (policy_rejected - ref_rejected)))

    Full NLL is the mean over the batch, computed with logaddexp so that
    extreme log-ratios stay finite.
    """
    policy_ratio = np.asarray(policy_chosen_logps, dtype=float) - np.asarray(
        policy_rejected_logps, dtype=float
    )
    ref_ratio = np.asarray(ref_chosen_logps, dtype=float) - np.asarray(
        ref_rejected_logps, dtype=float
    )
    implicit_reward = beta * (policy_ratio - ref_ratio)
    nll = -_log_sigmoid(implicit_reward)
    return float(np.mean(nll))


def orpo_loss(
    sft_nll: np.ndarray,
    chosen_logps: np.ndarray,
    rejected_logps: np.ndarray,
    lambda_orpo: float,
) -> float:
    """Compute a simplified ORPO-style objective.

    Combines an SFT term (negative log-likelihood of the chosen response)
    with an odds-ratio preference penalty:

        loss = mean(sft_nll) - lambda_orpo * log_sigmoid(odds_ratio)

    where ``odds_ratio = chosen_logps - rejected_logps``. Minimizing
    ``-log_sigmoid`` pushes the chosen odds above the rejected odds.
    """
    sft = float(np.mean(np.asarray(sft_nll, dtype=float)))
    odds_ratio = np.asarray(chosen_logps, dtype=float) - np.asarray(rejected_logps, dtype=float)
    preference_penalty = float(np.mean(-_log_sigmoid(odds_ratio)))
    return sft + lambda_orpo * preference_penalty
