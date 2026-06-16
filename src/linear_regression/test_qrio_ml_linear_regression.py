"""
Tests for linear regression from scratch.

Author: Sineth Sankalpa, AI/ML Engineer
Org: qriomatrix.com
Domain: test
Module: test_qrio_ml_linear_regression
"""

import pytest
from qrio_ml_linear_regression import QrioLinearRegression


class TestQrioMlLinearRegression:
    def test_qrio_fit_perfect_line_recovers_params(self):
        """QTCD-ID: QRIO-TC-0001

        Objective: Verify exact univariate line recovery (easy).
        Given: Targets follow y = 2x + 1 for five evenly spaced x values.
        When: The model is fit with the normal equation method.
        Then: Learned intercept is 1.0 and slope is 2.0 within tolerance.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLinearRegression()
        qrio_features = [0.0, 1.0, 2.0, 3.0, 4.0]
        qrio_targets = [1.0, 3.0, 5.0, 7.0, 9.0]
        qrio_model.fit(qrio_features, qrio_targets)

        assert qrio_model.qrio_weights is not None
        assert abs(qrio_model.qrio_weights[0] - 1.0) < 1e-6
        assert abs(qrio_model.qrio_weights[1] - 2.0) < 1e-6

    def test_qrio_predict_single_point_returns_expected_value(self):
        """QTCD-ID: QRIO-TC-0002

        Objective: Verify single-point prediction after training (easy).
        Given: A model trained on y = 3x - 2.
        When: predict is called for x = 4.0.
        Then: The prediction equals 10.0 within tolerance.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLinearRegression()
        qrio_model.fit([0.0, 1.0, 2.0], [-2.0, 1.0, 4.0])
        qrio_prediction = qrio_model.predict([4.0])

        assert len(qrio_prediction) == 1
        assert abs(qrio_prediction[0] - 10.0) < 1e-6

    def test_qrio_score_perfect_fit_returns_one(self):
        """QTCD-ID: QRIO-TC-0003

        Objective: Verify R-squared equals 1.0 on noiseless data (easy).
        Given: Training data that lies exactly on a straight line.
        When: score is evaluated on the same training set.
        Then: The returned R-squared value is 1.0.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLinearRegression()
        qrio_features = [1.0, 2.0, 3.0, 4.0]
        qrio_targets = [4.0, 6.0, 8.0, 10.0]
        qrio_model.fit(qrio_features, qrio_targets)

        assert qrio_model.score(qrio_features, qrio_targets) == 1.0

    def test_qrio_fit_multivariate_recovers_weights(self):
        """QTCD-ID: QRIO-TC-0004

        Objective: Verify multivariate regression weight recovery (medium).
        Given: Targets follow y = 1 + 2*x1 + 3*x2 across four samples.
        When: The model is fit on two-dimensional feature rows.
        Then: Intercept and both feature weights match the generating function.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLinearRegression()
        qrio_features = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
        qrio_targets = [1.0, 3.0, 4.0, 6.0]
        qrio_model.fit(qrio_features, qrio_targets)

        assert qrio_model.qrio_weights is not None
        assert abs(qrio_model.qrio_weights[0] - 1.0) < 1e-6
        assert abs(qrio_model.qrio_weights[1] - 2.0) < 1e-6
        assert abs(qrio_model.qrio_weights[2] - 3.0) < 1e-6

    def test_qrio_compute_mse_perfect_fit_returns_zero(self):
        """QTCD-ID: QRIO-TC-0005

        Objective: Verify zero MSE on perfectly fitted training data (medium).
        Given: A model trained on a noiseless linear relationship.
        When: qrio_compute_mse is called on the training set.
        Then: The mean squared error is 0.0.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLinearRegression()
        qrio_features = [0.0, 2.0, 4.0]
        qrio_targets = [5.0, 9.0, 13.0]
        qrio_model.fit(qrio_features, qrio_targets)

        assert qrio_model.qrio_compute_mse(qrio_features, qrio_targets) == 0.0

    def test_qrio_gradient_descent_converges_near_normal_solution(self):
        """QTCD-ID: QRIO-TC-0006

        Objective: Verify gradient descent approximates normal equation (medium).
        Given: The same univariate dataset for both fitting methods.
        When: One model uses normal equation and another uses gradient descent.
        Then: Both models produce predictions within 0.05 of each other.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_features = [0.0, 1.0, 2.0, 3.0, 5.0]
        qrio_targets = [2.0, 4.0, 6.0, 8.0, 12.0]
        qrio_normal = QrioLinearRegression(qrio_method="normal")
        qrio_gd = QrioLinearRegression(
            qrio_method="gradient_descent",
            qrio_learning_rate=0.05,
            qrio_max_iterations=5000,
        )
        qrio_normal.fit(qrio_features, qrio_targets)
        qrio_gd.fit(qrio_features, qrio_targets)

        qrio_eval_features = [1.5, 2.5, 4.0]
        qrio_normal_preds = qrio_normal.predict(qrio_eval_features)
        qrio_gd_preds = qrio_gd.predict(qrio_eval_features)

        for normal_value, gd_value in zip(qrio_normal_preds, qrio_gd_preds):
            assert abs(normal_value - gd_value) < 0.05

    def test_qrio_no_intercept_fits_through_origin(self):
        """QTCD-ID: QRIO-TC-0007

        Objective: Verify regression without intercept passes through origin (medium).
        Given: Data generated by y = 4x with qrio_fit_intercept disabled.
        When: The model is fit and evaluated at x = 0.
        Then: The learned slope is 4.0 and prediction at zero is 0.0.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLinearRegression(qrio_fit_intercept=False)
        qrio_features = [0.0, 1.0, 2.0, 3.0]
        qrio_targets = [0.0, 4.0, 8.0, 12.0]
        qrio_model.fit(qrio_features, qrio_targets)

        assert qrio_model.qrio_weights is not None
        assert len(qrio_model.qrio_weights) == 1
        assert abs(qrio_model.qrio_weights[0] - 4.0) < 1e-6
        assert abs(qrio_model.predict([0.0])[0]) < 1e-6

    def test_qrio_fit_noisy_data_achieves_high_r2(self):
        """QTCD-ID: QRIO-TC-0008

        Objective: Verify robust fit on mildly noisy observations (hard).
        Given: Ten samples around y = 1.5x + 0.5 with small Gaussian-like noise.
        When: The model is fit and scored on the noisy training set.
        Then: R-squared exceeds 0.95 and MSE remains below 0.05.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLinearRegression()
        qrio_features = [float(index) for index in range(10)]
        qrio_targets = [
            0.5 + 1.5 * feature + noise
            for feature, noise in zip(
                qrio_features,
                [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02, 0.02, -0.01],
            )
        ]
        qrio_model.fit(qrio_features, qrio_targets)

        assert qrio_model.score(qrio_features, qrio_targets) > 0.95
        assert qrio_model.qrio_compute_mse(qrio_features, qrio_targets) < 0.05

    def test_qrio_fit_mismatched_targets_raises_value_error(self):
        """QTCD-ID: QRIO-TC-0009

        Objective: Verify invalid training shapes raise ValueError (hard).
        Given: Three feature rows and only two target values.
        When: fit is called with mismatched lengths.
        Then: A ValueError is raised.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLinearRegression()

        with pytest.raises(ValueError, match="qrio_targets length must match"):
            qrio_model.fit([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_qrio_predict_before_fit_raises_runtime_error(self):
        """QTCD-ID: QRIO-TC-0010

        Objective: Verify unfitted model cannot predict (hard).
        Given: A newly instantiated model with no training step.
        When: predict is called on feature input.
        Then: A RuntimeError is raised.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLinearRegression()

        with pytest.raises(RuntimeError, match="Model is not fitted"):
            qrio_model.predict([1.0, 2.0])
