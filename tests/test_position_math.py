import unittest
import numpy as np

from exp_code.positional_intervention.position_math import probabilities, relocate_rows


class PositionMathTests(unittest.TestCase):
    def test_known_transfer_mass_and_conditional_ratios(self):
        original = np.array([[.6, .3, .1, 0.], [.2, .2, .6, 0.]])
        with np.errstate(divide='ignore'):
            logits = np.log(original)
        changed, info = relocate_rows(logits, 0, .5)
        actual = probabilities(changed)
        np.testing.assert_allclose(actual, [[.3, .525, .175, 0.], [.5, .125, .375, 0.]], atol=1e-14)
        np.testing.assert_allclose(actual.sum(axis=1), 1.)
        self.assertAlmostEqual(actual[:, 0].sum(), .8)
        self.assertAlmostEqual(info['requested_delta'], .3)
        self.assertTrue(np.isneginf(changed[:, 3]).all())
        np.testing.assert_allclose(probabilities(logits), original)

    def test_zero_fraction_is_exact_identity(self):
        logits = np.array([[37., 35., -np.inf], [-4., -3., -np.inf]], dtype=np.float32)
        changed, info = relocate_rows(logits, 0, 0.)
        self.assertIs(changed, logits)
        self.assertTrue(info['exact_noop'])

    def test_full_transfer_and_capacity_limit(self):
        for values, expected in (([[.4, .6], [.2, .8]], [0., .6]),
                                 ([[.8, .2], [.9, .1]], [.7, 1.])):
            changed, _ = relocate_rows(np.log(values), 0, 1.)
            np.testing.assert_allclose(probabilities(changed)[:, 0], expected, atol=1e-14)

    def test_extreme_source_retains_nonanchor_ratios(self):
        logits = np.array([[1000., 0., -1., -np.inf], [0., 0., -1., -np.inf]])
        changed, _ = relocate_rows(logits, 0, .5)
        actual = probabilities(changed)
        np.testing.assert_allclose(actual.sum(axis=1), 1.)
        self.assertAlmostEqual(actual[0, 1]/actual[0, 2], np.e)
        self.assertTrue(np.isneginf(changed[:, 3]).all())

    def test_saturated_target_is_exact_noop(self):
        logits = np.array([[0., 0., -1.], [1000., 0., -1.]])
        changed, info = relocate_rows(logits, 0, .5)
        self.assertIs(changed, logits)
        self.assertEqual(info['requested_delta'], 0.)

    def test_forbidden_and_nonfinite_values_rejected(self):
        for rows in ([[0., 1.], [-np.inf, 0.]],
                     [[np.inf, 1.], [1., 0.]], [[np.nan, 1.], [1., 0.]],
                     [[0., -np.inf], [0., 0.]]):
            with self.assertRaises(ValueError):
                relocate_rows(np.array(rows), 0, .5)
        for fraction in (-.1, 1.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                relocate_rows(np.zeros((2, 3)), 0, fraction)

    def test_random_distributions_preserve_mass_and_other_ratios(self):
        rng = np.random.default_rng(230)
        for _ in range(50):
            logits = rng.normal(size=(2, 11))*4
            before = probabilities(logits)
            changed, _ = relocate_rows(logits, 3, float(rng.uniform(.05, .95)))
            after = probabilities(changed)
            np.testing.assert_allclose(after.sum(axis=1), 1., atol=2e-14)
            self.assertAlmostEqual(after[:, 3].sum(), before[:, 3].sum(), places=12)
            other = np.arange(11) != 3
            np.testing.assert_allclose(after[:, other]/after[:, other].sum(axis=1, keepdims=True),
                before[:, other]/before[:, other].sum(axis=1, keepdims=True), atol=1e-13)


if __name__ == '__main__':
    unittest.main()
