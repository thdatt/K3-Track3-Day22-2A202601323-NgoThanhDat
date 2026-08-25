import numpy as np
import pytest

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_is_positive() -> None:
    # Identical policy and ref log-ratios => reward diff 0 => loss = -log(sigmoid(0)) = log(2) > 0
    loss = dpo_loss(
        policy_chosen_logps=np.array([-0.5, -0.6]),
        policy_rejected_logps=np.array([-1.5, -1.6]),
        ref_chosen_logps=np.array([-0.5, -0.6]),
        ref_rejected_logps=np.array([-1.5, -1.6]),
        beta=1.0,
    )
    assert loss > 0.0


def test_dpo_loss_smaller_when_policy_prefers_chosen() -> None:
    aligned = dpo_loss(
        np.array([-0.1]),
        np.array([-3.0]),
        np.array([-0.5]),
        np.array([-1.0]),
        beta=1.0,
    )
    misaligned = dpo_loss(
        np.array([-3.0]),
        np.array([-0.1]),
        np.array([-0.5]),
        np.array([-1.0]),
        beta=1.0,
    )
    assert aligned < misaligned


def test_dpo_loss_is_finite_for_extreme_logits() -> None:
    loss = dpo_loss(
        np.array([20.0]),
        np.array([-20.0]),
        np.array([-20.0]),
        np.array([20.0]),
        beta=10.0,
    )
    assert np.isfinite(loss)


def test_dpo_loss_averages_batch() -> None:
    loss = dpo_loss(
        np.array([-0.5, -0.5]),
        np.array([-1.5, -1.5]),
        np.array([-0.6, -0.6]),
        np.array([-1.0, -1.0]),
        beta=0.1,
    )
    single = dpo_loss(
        np.array([-0.5]), np.array([-1.5]), np.array([-0.6]), np.array([-1.0]), beta=0.1
    )
    assert loss == pytest.approx(single)


def test_orpo_loss_combines_sft_and_preference() -> None:
    loss = orpo_loss(
        sft_nll=np.array([1.0]),
        chosen_logps=np.array([-0.5]),
        rejected_logps=np.array([-1.5]),
        lambda_orpo=0.1,
    )
    assert np.isfinite(loss)
    assert loss > 0.0


def test_orpo_loss_smaller_when_chosen_preferred() -> None:
    pref = orpo_loss(
        sft_nll=np.array([1.0]),
        chosen_logps=np.array([-0.5]),
        rejected_logps=np.array([-1.5]),
        lambda_orpo=0.1,
    )
    anti = orpo_loss(
        sft_nll=np.array([1.0]),
        chosen_logps=np.array([-1.5]),
        rejected_logps=np.array([-0.5]),
        lambda_orpo=0.1,
    )
    assert pref < anti


def test_orpo_loss_grows_with_lambda() -> None:
    small = orpo_loss(np.array([1.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=0.0)
    large = orpo_loss(np.array([1.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=5.0)
    # lambda is non-negative; larger lambda should not reduce loss below SFT-only
    assert large >= small - 1e-9
