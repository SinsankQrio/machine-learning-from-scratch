"""
Tests for multi-layer perceptron from scratch.

Author: Sineth Sankalpa, AI/ML Engineer
Org: qriomatrix.com
Domain: test
Module: test_qrio_ml_multi_layer_perceptron
"""

import pytest
from qrio_ml_multi_layer_perceptron import QrioMultiLayerPerceptron


class TestQrioMlMultiLayerPerceptron:
    def test_qrio_fit_xor_learns_nonlinear_boundary(self):
        """QTCD-ID: QRIO-TC-0001

        Objective: Verify MLP learns the XOR function (hard).
        Given: Four binary samples with XOR labels.
        When: The model is fit with a single hidden layer of four neurons.
        Then: All training samples are classified correctly.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioMultiLayerPerceptron(
            qrio_hidden_layers=[4],
            qrio_learning_rate=0.5,
            qrio_max_iterations=20000,
        )
        qrio_features = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
        qrio_labels = [0, 1, 1, 0]
        qrio_model.fit(qrio_features, qrio_labels)

        assert qrio_model.qrio_layer_weights is not None
        assert qrio_model.predict(qrio_features) == qrio_labels

    def test_qrio_fit_separable_data_classifies_correctly(self):
        """QTCD-ID: QRIO-TC-0002

        Objective: Verify binary classification on separable data (easy).
        Given: Univariate features with labels split at x = 2.5.
        When: The model is fit on five evenly spaced samples.
        Then: Predictions match all training labels.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioMultiLayerPerceptron(qrio_hidden_layers=[3])
        qrio_features = [0.0, 1.0, 2.0, 3.0, 4.0]
        qrio_labels = [0, 0, 0, 1, 1]
        qrio_model.fit(qrio_features, qrio_labels)

        assert qrio_model.predict(qrio_features) == qrio_labels

    def test_qrio_predict_single_sample_returns_class(self):
        """QTCD-ID: QRIO-TC-0003

        Objective: Verify single-point class prediction after training (easy).
        Given: A model trained on separable univariate data.
        When: predict is called for x = 4.0.
        Then: The predicted label is 1.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioMultiLayerPerceptron(qrio_hidden_layers=[3])
        qrio_features = [0.0, 1.0, 2.0, 3.0, 4.0]
        qrio_labels = [0, 0, 0, 1, 1]
        qrio_model.fit(qrio_features, qrio_labels)
        qrio_prediction = qrio_model.predict([4.0])

        assert len(qrio_prediction) == 1
        assert qrio_prediction[0] == 1

    def test_qrio_predict_proba_returns_valid_probabilities(self):
        """QTCD-ID: QRIO-TC-0004

        Objective: Verify predicted probabilities lie in (0, 1) (easy).
        Given: A fitted model on binary classification data.
        When: predict_proba is called on training features.
        Then: All probabilities are strictly between 0 and 1.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioMultiLayerPerceptron(qrio_hidden_layers=[3])
        qrio_features = [0.0, 1.0, 2.0, 3.0, 4.0]
        qrio_labels = [0, 0, 0, 1, 1]
        qrio_model.fit(qrio_features, qrio_labels)
        qrio_probabilities = qrio_model.predict_proba(qrio_features)

        assert len(qrio_probabilities) == len(qrio_features)
        for probabilities in qrio_probabilities:
            assert 0.0 < probabilities[0] < 1.0

    def test_qrio_score_perfect_classification_returns_one(self):
        """QTCD-ID: QRIO-TC-0005

        Objective: Verify accuracy equals 1.0 on learned training data (easy).
        Given: XOR training data and a fitted two-hidden-neuron model.
        When: score is evaluated on the same training set.
        Then: The returned accuracy is 1.0.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioMultiLayerPerceptron(
            qrio_hidden_layers=[4],
            qrio_learning_rate=0.5,
        )
        qrio_features = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
        qrio_labels = [0, 1, 1, 0]
        qrio_model.fit(qrio_features, qrio_labels)

        assert qrio_model.score(qrio_features, qrio_labels) == 1.0

    def test_qrio_fit_multivariate_classifies_correctly(self):
        """QTCD-ID: QRIO-TC-0006

        Objective: Verify multivariate binary classification (medium).
        Given: Two-dimensional features where label is 1 when x1 + x2 > 1.
        When: The model is fit on four corner samples.
        Then: All training samples are classified correctly.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioMultiLayerPerceptron(qrio_hidden_layers=[4])
        qrio_features = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
        qrio_labels = [0, 0, 0, 1]
        qrio_model.fit(qrio_features, qrio_labels)

        assert qrio_model.predict(qrio_features) == qrio_labels

    def test_qrio_compute_cross_entropy_loss_decreases_after_training(self):
        """QTCD-ID: QRIO-TC-0007

        Objective: Verify training reduces cross-entropy loss (medium).
        Given: Randomly initialized weights versus a fitted XOR model.
        When: qrio_compute_cross_entropy_loss is compared before and after fit.
        Then: Post-training loss is lower than the unfitted baseline.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_features = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
        qrio_labels = [0, 1, 1, 0]
        qrio_unfitted = QrioMultiLayerPerceptron(qrio_hidden_layers=[4])
        qrio_unfitted.qrio_num_classes = 2
        qrio_unfitted.qrio_layer_weights = [
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
            [[0.5, 0.5, 0.5, 0.5, 0.5]],
        ]
        qrio_unfitted_loss = qrio_unfitted.qrio_compute_cross_entropy_loss(
            qrio_features,
            qrio_labels,
        )

        qrio_model = QrioMultiLayerPerceptron(
            qrio_hidden_layers=[4],
            qrio_learning_rate=0.5,
        )
        qrio_model.fit(qrio_features, qrio_labels)
        qrio_fitted_loss = qrio_model.qrio_compute_cross_entropy_loss(
            qrio_features,
            qrio_labels,
        )

        assert qrio_fitted_loss < qrio_unfitted_loss

    def test_qrio_fit_three_class_softmax_classifies_correctly(self):
        """QTCD-ID: QRIO-TC-0008

        Objective: Verify multi-class softmax output (hard).
        Given: Three clusters in 2-D space with labels 0, 1, and 2.
        When: The model is fit with two hidden neurons.
        Then: All training samples are classified correctly.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioMultiLayerPerceptron(
            qrio_hidden_layers=[6],
            qrio_learning_rate=0.3,
        )
        qrio_features = [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
            [2.0, 0.0],
            [2.1, 0.0],
        ]
        qrio_labels = [0, 0, 1, 1, 2, 2]
        qrio_model.fit(qrio_features, qrio_labels)

        assert qrio_model.predict(qrio_features) == qrio_labels

    def test_qrio_fit_mismatched_labels_raises_value_error(self):
        """QTCD-ID: QRIO-TC-0009

        Objective: Verify invalid training shapes raise ValueError (hard).
        Given: Three feature rows and only two label values.
        When: fit is called with mismatched lengths.
        Then: A ValueError is raised.

        Author: Sineth Sankalpa, AI/ML Engineer
        Org: qriomatrix.com
        """
        qrio_model = QrioMultiLayerPerceptron()

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
        qrio_model = QrioMultiLayerPerceptron()

        with pytest.raises(RuntimeError, match="Model is not fitted"):
            qrio_model.predict([[1.0, 2.0]])
