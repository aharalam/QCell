import numpy as np

from synthetic_control import ANGLE_COLUMNS, planted_control, synthetic_features


def test_same_seed_gives_the_same_planted_values_and_labels():
    a, b = planted_control(300, seed=11), planted_control(300, seed=11)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], planted_control(300, seed=12)[0])


def test_label_is_exactly_the_documented_interaction_rule():
    planted, y = planted_control(500, seed=1)
    assert planted.shape == (500, 2)
    assert planted.min() >= -1 and planted.max() <= 1
    assert np.array_equal(y, ((planted[:, 0] > 0) != (planted[:, 1] > 0)).astype(int))


def test_neither_planted_feature_predicts_the_label_alone():
    planted, y = planted_control(4000, seed=2)
    for column in (0, 1):
        agreement = np.mean((planted[:, column] > 0).astype(int) == y)
        assert abs(agreement - 0.5) < 0.04
    assert 0.4 < y.mean() < 0.6


def test_planted_values_fill_the_first_two_feature_columns_as_fixed_angles():
    planted, _ = planted_control(50, seed=3)
    nuisance = np.random.default_rng(0).uniform(0, np.pi, size=(50, 6))
    features = synthetic_features(planted, nuisance)
    assert features.shape == (50, 8)
    assert ANGLE_COLUMNS == (0, 1)
    assert np.allclose(features[:, :2], (planted + 1) / 2 * np.pi)  # fixed map, nothing fitted
    assert np.array_equal(features[:, 2:], nuisance)
