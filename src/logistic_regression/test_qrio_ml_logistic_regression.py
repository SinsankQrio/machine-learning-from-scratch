"""
Tests for logistic regression from scratch.

Author: Sineth Sankalpa, AI/ML Engineer
Org: qriomatrix.com
Domain: test
Module: test_qrio_ml_logistic_regression
"""

import pytest
from qrio_ml_logistic_regression import QrioLogisticRegression


class TestQrioMlLogisticRegression:
    def test_qrio_fit_separable_data_learns_boundary(self):
        """QTCD-ID: QRIO-TC-0001

        Objective: Verify model separates linearly separable classes (easy).
        Given: Labels split at x = 2.5 for five evenly spaced x values.
        When: The model is fit on univariate features.
        Then: Predictions match all training labels.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLogisticRegression(qrio_learning_rate=0.5)
        qrio_features = [0.0, 1.0, 2.0, 3.0, 4.0]
        qrio_labels = [0, 0, 0, 1, 1]
        qrio_model.fit(qrio_features, qrio_labels)

        assert qrio_model.qrio_weights is not None
        assert qrio_model.predict(qrio_features) == qrio_labels

    def test_qrio_predict_single_sample_returns_class(self):
        """QTCD-ID: QRIO-TC-0002

        Objective: Verify single-point class prediction after training (easy).
        Given: A model trained on data with decision boundary near x = 1.5.
        When: predict is called for x = 3.0.
        Then: The predicted label is 1.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLogisticRegression(qrio_learning_rate=0.5)
        qrio_features = [0.0, 1.0, 2.0, 3.0, 4.0]
        qrio_labels = [0, 0, 0, 1, 1]
        qrio_model.fit(qrio_features, qrio_labels)
        qrio_prediction = qrio_model.predict([3.0])

        assert len(qrio_prediction) == 1
        assert qrio_prediction[0] == 1

    def test_qrio_predict_proba_returns_valid_probability(self):
        """QTCD-ID: QRIO-TC-0003

        Objective: Verify predicted probabilities lie in (0, 1) (easy).
        Given: A fitted model on binary classification data.
        When: predict_proba is called on training features.
        Then: All probabilities are strictly between 0 and 1.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLogisticRegression(qrio_learning_rate=0.5)
        qrio_features = [0.0, 1.0, 2.0, 3.0, 4.0]
        qrio_labels = [0, 0, 0, 1, 1]
        qrio_model.fit(qrio_features, qrio_labels)
        qrio_probabilities = qrio_model.predict_proba(qrio_features)

        assert len(qrio_probabilities) == len(qrio_features)
        for probability in qrio_probabilities:
            assert 0.0 < probability < 1.0

    def test_qrio_score_perfect_classification_returns_one(self):
        """QTCD-ID: QRIO-TC-0004

        Objective: Verify accuracy equals 1.0 on separable training data (easy).
        Given: Linearly separable univariate classification data.
        When: score is evaluated on the same training set.
        Then: The returned accuracy is 1.0.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLogisticRegression(qrio_learning_rate=0.5)
        qrio_features = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        qrio_labels = [0, 0, 0, 1, 1, 1]
        qrio_model.fit(qrio_features, qrio_labels)

        assert qrio_model.score(qrio_features, qrio_labels) == 1.0

    def test_qrio_fit_multivariate_classifies_correctly(self):
        """QTCD-ID: QRIO-TC-0005

        Objective: Verify multivariate logistic regression (medium).
        Given: Two-dimensional features where label equals 1 when x1 + x2 > 1.
        When: The model is fit on four corner samples.
        Then: All training samples are classified correctly.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLogisticRegression(qrio_learning_rate=0.5)
        qrio_features = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
        qrio_labels = [0, 0, 0, 1]
        qrio_model.fit(qrio_features, qrio_labels)

        assert qrio_model.predict(qrio_features) == qrio_labels

    def test_qrio_compute_log_loss_decreases_after_training(self):
        """QTCD-ID: QRIO-TC-0006

        Objective: Verify training reduces log loss (medium).
        Given: Randomly initialized weights versus a fitted model.
        When: qrio_compute_log_loss is compared before and after fit.
        Then: Post-training log loss is lower than the unfitted baseline.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_features = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        qrio_labels = [0, 0, 0, 1, 1, 1]
        qrio_unfitted = QrioLogisticRegression()
        qrio_unfitted.qrio_weights = [0.0, 0.0]
        qrio_unfitted_loss = qrio_unfitted.qrio_compute_log_loss(
            qrio_features,
            qrio_labels,
        )

        qrio_model = QrioLogisticRegression(qrio_learning_rate=0.5)
        qrio_model.fit(qrio_features, qrio_labels)
        qrio_fitted_loss = qrio_model.qrio_compute_log_loss(qrio_features, qrio_labels)

        assert qrio_fitted_loss < qrio_unfitted_loss

    def test_qrio_no_intercept_fits_without_bias(self):
        """QTCD-ID: QRIO-TC-0007

        Objective: Verify regression without intercept (medium).
        Given: Labels follow sign(x) with qrio_fit_intercept disabled.
        When: The model is fit on symmetric positive and negative x values.
        Then: Predictions match training labels and only one weight is learned.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLogisticRegression(
            qrio_fit_intercept=False,
            qrio_learning_rate=0.5,
        )
        qrio_features = [-2.0, -1.0, 1.0, 2.0]
        qrio_labels = [0, 0, 1, 1]
        qrio_model.fit(qrio_features, qrio_labels)

        assert qrio_model.qrio_weights is not None
        assert len(qrio_model.qrio_weights) == 1
        assert qrio_model.predict(qrio_features) == qrio_labels

    def test_qrio_fit_noisy_boundary_achieves_high_accuracy(self):
        """QTCD-ID: QRIO-TC-0008

        Objective: Verify robust fit with one mislabeled sample (hard).
        Given: Nine correctly labeled points and one flipped label near boundary.
        When: The model is fit and scored on the noisy training set.
        Then: Accuracy is at least 0.8.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLogisticRegression(qrio_learning_rate=0.3)
        qrio_features = [float(index) for index in range(10)]
        qrio_labels = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
        qrio_model.fit(qrio_features, qrio_labels)

        assert qrio_model.score(qrio_features, qrio_labels) >= 0.8

    def test_qrio_fit_mismatched_labels_raises_value_error(self):
        """QTCD-ID: QRIO-TC-0009

        Objective: Verify invalid training shapes raise ValueError (hard).
        Given: Three feature rows and only two label values.
        When: fit is called with mismatched lengths.
        Then: A ValueError is raised.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLogisticRegression()

        with pytest.raises(ValueError, match="qrio_labels length must match"):
            qrio_model.fit([1.0, 2.0, 3.0], [0, 1])

    def test_qrio_predict_before_fit_raises_runtime_error(self):
        """QTCD-ID: QRIO-TC-0010

        Objective: Verify unfitted model cannot predict (hard).
        Given: A newly instantiated model with no training step.
        When: predict is called on feature input.
        Then: A RuntimeError is raised.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioLogisticRegression()

        with pytest.raises(RuntimeError, match="Model is not fitted"):
            qrio_model.predict([1.0, 2.0])
