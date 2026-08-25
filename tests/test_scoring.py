from preference_lab.scoring import score_text


def test_score_increases_with_length() -> None:
    short = score_text("ok")
    long = score_text("a much more detailed and longer response here")
    assert long > short


def test_score_penalizes_empty() -> None:
    assert score_text("") == 0.0
    assert score_text("   ") == 0.0


def test_score_is_deterministic() -> None:
    assert score_text("the quick brown fox") == score_text("the quick brown fox")


def test_score_is_finite() -> None:
    assert score_text("a " * 1000) < float("inf")


def test_score_respects_structural_richness() -> None:
    # More sentences / structure should generally not hurt vs a bare word dump
    structured = score_text("First, we consider X. Then, we evaluate Y. Finally, we decide.")
    assert structured > 0.0


def test_response_features_have_fixed_shape() -> None:
    from preference_lab.scoring import FEATURE_NAMES, response_features

    features = response_features("A concise but structured answer. It explains why.")
    assert features.shape == (len(FEATURE_NAMES),)
