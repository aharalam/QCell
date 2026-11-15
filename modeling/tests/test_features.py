import numpy as np
import pytest

from features import FoldFeatures


def counts(n_cells=40, n_genes=60, seed=0):
    rng = np.random.default_rng(seed)
    return rng.poisson(rng.uniform(0.2, 6.0, size=n_genes), size=(n_cells, n_genes)).astype(float)


def test_output_is_one_angle_per_component_within_zero_to_pi():
    data = counts()
    out = FoldFeatures(n_hvg=30, n_components=8).fit(data).transform(data)
    assert out.shape == (40, 8)
    assert out.min() >= 0 and out.max() <= np.pi
    assert np.all(np.isfinite(out))


def test_held_out_cells_beyond_the_training_range_are_clipped_not_wrapped():
    train = counts(seed=1)
    extreme = train[:3] * 50
    out = FoldFeatures(n_hvg=30, n_components=4).fit(train).transform(extreme)
    assert out.min() >= 0 and out.max() <= np.pi


def test_only_the_most_variable_training_genes_are_kept():
    rng = np.random.default_rng(2)
    data = np.full((50, 20), 50.0)
    data[:, [3, 7, 11]] = rng.poisson(30, size=(50, 3))  # vary by roughly 18%
    data[:, 0] = rng.integers(49, 52, size=50)  # varies by about 2%
    fitted = FoldFeatures(n_hvg=3, n_components=2).fit(data)
    assert sorted(fitted.gene_index_.tolist()) == [3, 7, 11]


def test_genes_that_never_vary_do_not_produce_nan():
    data = counts(seed=3)
    data[:, :10] = 0.0
    out = FoldFeatures(n_hvg=60, n_components=5).fit(data).transform(data)
    assert np.all(np.isfinite(out))


def test_fitted_statistics_come_from_the_cells_given_to_fit_only():
    train, other = counts(seed=4), counts(seed=5) * 3
    alone = FoldFeatures(n_hvg=30, n_components=6).fit(train)
    again = FoldFeatures(n_hvg=30, n_components=6).fit(train)
    again.transform(other)  # transforming other cells must not change the fit
    assert again.transform(train) == pytest.approx(alone.transform(train))
    pooled = FoldFeatures(n_hvg=30, n_components=6).fit(np.vstack([train, other]))
    assert not np.allclose(pooled.transform(train), alone.transform(train))
