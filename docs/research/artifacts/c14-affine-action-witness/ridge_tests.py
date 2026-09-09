"""Synthetic-only ridge tests. No C12/C13 artifacts are accessed."""
import copy
import unittest
from unittest.mock import patch

import ridge

np = ridge.np


class RidgeTests(unittest.TestCase):
    def positive(self):
        # Disjoint nuisance examples; the first four features linearly encode y.
        train_y = np.tile(np.arange(4), 12)
        eval_y = np.tile(np.arange(4), 7)
        rng = np.random.default_rng(17)
        train = np.column_stack((np.eye(4)[train_y], rng.normal(size=(len(train_y), 3))))
        fresh = np.column_stack((np.eye(4)[eval_y], rng.normal(size=(len(eval_y), 3))))
        return train, train_y, fresh, eval_y

    def test_disjoint_linear_four_action_positive(self):
        train, labels, fresh, expected = self.positive()
        fitted = ridge.fit(train, labels)
        scores = ridge.predict(fresh, fitted)
        np.testing.assert_array_equal(scores.argmax(axis=1), expected)
        self.assertEqual(scores.dtype, np.float64)
        self.assertLessEqual(fitted["relative_normal_equation_residual"], 1e-9)
        self.assertGreater(fitted["objective"], 0.)
        self.assertFalse({tuple(x) for x in train} & {tuple(x) for x in fresh})

    def test_independent_augmented_least_squares_coefficients_scores_argmax(self):
        train, labels, fresh, _ = self.positive()
        fitted = ridge.fit(train, labels)
        n, d = train.shape
        Z = (train - train.mean(axis=0)) / np.maximum(train.std(axis=0), 1e-6)
        Y = np.eye(4)[labels]
        design = np.vstack((Z / np.sqrt(n), np.sqrt(.01) * np.eye(d)))
        targets = np.vstack(((Y - Y.mean(axis=0)) / np.sqrt(n), np.zeros((d, 4))))
        alternative, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
        np.testing.assert_allclose(fitted["coefficients"], alternative, atol=1e-9, rtol=1e-7)
        for X in (train, fresh):
            scores = ridge.predict(X, fitted)
            reference = (X - fitted["mean"]) / fitted["scale"] @ alternative + Y.mean(axis=0)
            np.testing.assert_allclose(scores, reference, atol=1e-9, rtol=1e-7)
            np.testing.assert_array_equal(scores.argmax(axis=1), reference.argmax(axis=1))

    def test_unpenalized_intercept_and_summed_output_objective(self):
        X = np.zeros((8, 3))
        labels = np.array([0, 0, 0, 0, 1, 1, 2, 3])
        fitted = ridge.fit(X, labels)
        np.testing.assert_array_equal(fitted["intercept"], [.5, .25, .125, .125])
        np.testing.assert_array_equal(fitted["coefficients"], np.zeros((3, 4)))
        self.assertEqual(fitted["objective"], .65625)  # 1 - (.5² + .25² + 2*.125²).
        self.assertEqual(fitted["relative_normal_equation_residual"], 0.)
        self.assertEqual(fitted["condition_number"], 1.)

    def test_exact_zero_rhs_never_calls_solver(self):
        X = np.arange(20, dtype=np.float64).reshape(10, 2)
        with patch.object(np.linalg, "solve", side_effect=AssertionError("unnecessary solve")):
            fitted = ridge.fit(X, np.zeros(10, dtype=np.int64))
        np.testing.assert_array_equal(fitted["coefficients"], np.zeros((2, 4)))
        self.assertEqual(fitted["objective"], 0.)

    def test_group_only_balanced_negative_exactly_quarter(self):
        rng = np.random.default_rng(5)
        X = np.repeat(rng.normal(size=(9, 6)), 16, axis=0)
        labels = np.tile(np.repeat(np.arange(4), 4), 9)
        fitted = ridge.fit(X, labels)
        np.testing.assert_allclose(fitted["coefficients"], 0., atol=1e-12, rtol=0.)
        for maps in (16, 8):
            fresh = np.repeat(rng.normal(size=(7, 6)), maps, axis=0)
            target = np.tile(np.repeat(np.arange(4), maps // 4), 7)
            prediction = ridge.predict(fresh, fitted).argmax(axis=1).reshape(7, maps)
            np.testing.assert_array_equal(prediction, np.repeat(prediction[:, :1], maps, axis=1))
            self.assertEqual(np.count_nonzero(prediction.ravel() == target), len(target) // 4)

    def test_training_only_population_std_and_floor_keep_dimensions(self):
        X = np.column_stack((np.arange(8), np.full(8, 7.), np.arange(8) * 1e-9))
        fitted = ridge.fit(X, np.arange(8) % 4)
        np.testing.assert_array_equal(fitted["mean"], X.mean(axis=0))
        np.testing.assert_array_equal(fitted["std"], X.std(axis=0, ddof=0))
        np.testing.assert_array_equal(fitted["scale"], np.maximum(X.std(axis=0, ddof=0), 1e-6))
        self.assertEqual(fitted["clamped_dimensions"], 2)
        self.assertEqual(fitted["coefficients"].shape, (3, 4))
        original = copy.deepcopy(fitted)
        alone = ridge.predict(X[:1], fitted)
        with_outlier = ridge.predict(np.vstack((X[:1], [1e8, -1e8, 1e8])), fitted)[:1]
        np.testing.assert_allclose(alone, with_outlier, atol=1e-12, rtol=1e-12)
        for name in ("mean", "std", "scale", "coefficients", "intercept"):
            np.testing.assert_array_equal(fitted[name], original[name])
        np.testing.assert_array_equal(fitted["quantiles"]["probabilities"], [0., .01, .1, .5, .9, .99, 1.])
        np.testing.assert_allclose(fitted["quantiles"]["scale"],
                                   np.quantile(fitted["scale"], ridge.QUANTILES, method="linear"))

    def test_affine_reconstruction_in_original_features(self):
        train, labels, fresh, _ = self.positive()
        fitted = ridge.fit(train, labels)
        weight = fitted["coefficients"] / fitted["scale"][:, None]
        bias = fitted["intercept"] - fitted["mean"] @ weight
        for X in (train, fresh):
            reference = X @ weight + bias
            scores = ridge.predict(X, fitted)
            np.testing.assert_allclose(scores, reference, atol=1e-12, rtol=1e-12)
            np.testing.assert_array_equal(scores.argmax(axis=1), reference.argmax(axis=1))

    def test_row_duplication_preserves_mean_loss_regularization(self):
        train, labels, _, _ = self.positive()
        original = ridge.fit(train, labels)
        repeated = ridge.fit(np.repeat(train, 3, axis=0), np.repeat(labels, 3))
        np.testing.assert_allclose(original["coefficients"], repeated["coefficients"], atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(original["objective"], repeated["objective"], places=13)

    def test_general_dimension_and_single_row(self):
        for d in (1, 9):
            fitted = ridge.fit(np.full((1, d), 3.), np.array([2]))
            np.testing.assert_array_equal(ridge.predict(np.full((2, d), 3.), fitted), [[0., 0., 1., 0.]] * 2)
            self.assertEqual(fitted["clamped_dimensions"], d)

    def test_rejects_invalid_inputs_and_labels(self):
        good = np.arange(12, dtype=np.float64).reshape(4, 3)
        for X in (np.zeros((0, 3)), np.zeros((4, 0)), good.ravel(), good.astype(complex),
                  np.full((4, 3), np.nan), np.full((4, 3), np.inf), np.ones((4, 3), dtype=bool)):
            with self.assertRaises(ValueError):
                ridge.fit(X, np.arange(4))
        for labels in ([0, 1, 2], [0, 1, 2, 4], [-1, 1, 2, 3], [0., 1., 2., 3.],
                       [True, False, True, False], [[0], [1], [2], [3]]):
            with self.assertRaises(ValueError):
                ridge.fit(good, labels)

    def test_prediction_rejects_corrupt_coefficients_scale_and_shapes(self):
        train, labels, fresh, _ = self.positive()
        fitted = ridge.fit(train, labels)
        for name, bad in (("scale", np.zeros(7)), ("coefficients", np.full((7, 4), np.nan)),
                          ("mean", np.zeros(6)), ("intercept", np.zeros(3))):
            corrupted = dict(fitted, **{name: bad})
            with self.assertRaises(ValueError):
                ridge.predict(fresh, corrupted)

    def test_wrong_solver_solution_fails_normal_equation_guard(self):
        train, labels, _, _ = self.positive()
        with patch.object(np.linalg, "solve", return_value=np.zeros((7, 4))):
            with self.assertRaisesRegex(ValueError, "normal equation"):
                ridge.fit(train, labels)


if __name__ == "__main__":
    unittest.main()
