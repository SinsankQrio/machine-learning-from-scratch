"""
Binary logistic regression from scratch using gradient descent.

Author: Sineth Sankalpa, AI/ML Engineer
Org: qriomatrix.com
Domain: application
Module: qrio_ml_logistic_regression
"""

from __future__ import annotations

import math

QRIO_DEFAULT_LEARNING_RATE = 0.1
QRIO_DEFAULT_MAX_ITERATIONS = 10000
QRIO_DEFAULT_TOLERANCE = 1e-6
QRIO_DEFAULT_THRESHOLD = 0.5
QRIO_MIN_SAMPLES = 1


class QrioLogisticRegression:
    """Binary logistic regression with sigmoid activation.

    QrioMatrix | qriomatrix.com
    """

    def __init__(
        self,
        qrio_learning_rate: float = QRIO_DEFAULT_LEARNING_RATE,
        qrio_max_iterations: int = QRIO_DEFAULT_MAX_ITERATIONS,
        qrio_tolerance: float = QRIO_DEFAULT_TOLERANCE,
        qrio_fit_intercept: bool = True,
        qrio_threshold: float = QRIO_DEFAULT_THRESHOLD,
    ) -> None:
        """Initialize the logistic regression model.

        Args:
            qrio_learning_rate: Step size for gradient descent.
            qrio_max_iterations: Maximum gradient-descent iterations.
            qrio_tolerance: Convergence threshold for weight updates.
            qrio_fit_intercept: Whether to learn a bias term.
            qrio_threshold: Decision threshold for class labels.

        QrioMatrix | qriomatrix.com
        """
        self.qrio_learning_rate = qrio_learning_rate
        self.qrio_max_iterations = qrio_max_iterations
        self.qrio_tolerance = qrio_tolerance
        self.qrio_fit_intercept = qrio_fit_intercept
        self.qrio_threshold = qrio_threshold
        self.qrio_weights: list[float] | None = None

    def fit(
        self,
        qrio_features: list[list[float]] | list[float],
        qrio_labels: list[int],
    ) -> QrioLogisticRegression:
        """Fit model weights to binary classification data.

        Args:
            qrio_features: Feature matrix or 1-D feature vector.
            qrio_labels: Binary labels (0 or 1) aligned with feature rows.

        Returns:
            Fitted model instance.

        QrioMatrix | qriomatrix.com
        """
        qrio_design_matrix = _qrio_to_design_matrix(
            qrio_features,
            self.qrio_fit_intercept,
        )
        _qrio_validate_training_inputs(qrio_design_matrix, qrio_labels)
        self.qrio_weights = _qrio_solve_logistic_gradient_descent(
            qrio_design_matrix,
            qrio_labels,
            self.qrio_learning_rate,
            self.qrio_max_iterations,
            self.qrio_tolerance,
        )
        return self

    def predict_proba(
        self,
        qrio_features: list[list[float]] | list[float],
    ) -> list[float]:
        """Predict positive-class probabilities for feature rows.

        Args:
            qrio_features: Feature matrix or 1-D feature vector.

        Returns:
            Probabilities in the range (0, 1).

        QrioMatrix | qriomatrix.com
        """
        if self.qrio_weights is None:
            raise RuntimeError(
                "Model is not fitted. Call fit() before predict_proba()."
            )

        qrio_design_matrix = _qrio_to_design_matrix(
            qrio_features,
            self.qrio_fit_intercept,
        )
        qrio_logits = _qrio_matrix_vector_multiply(
            qrio_design_matrix,
            self.qrio_weights,
        )
        return [_qrio_sigmoid(logit) for logit in qrio_logits]

    def predict(
        self,
        qrio_features: list[list[float]] | list[float],
    ) -> list[int]:
        """Predict binary class labels for feature rows.

        Args:
            qrio_features: Feature matrix or 1-D feature vector.

        Returns:
            Predicted labels (0 or 1).

        QrioMatrix | qriomatrix.com
        """
        qrio_probabilities = self.predict_proba(qrio_features)
        return [
            1 if probability >= self.qrio_threshold else 0
            for probability in qrio_probabilities
        ]

    def score(
        self,
        qrio_features: list[list[float]] | list[float],
        qrio_labels: list[int],
    ) -> float:
        """Compute classification accuracy on a dataset.

        Args:
            qrio_features: Feature matrix or 1-D feature vector.
            qrio_labels: Ground-truth binary labels.

        Returns:
            Fraction of correctly classified samples.

        QrioMatrix | qriomatrix.com
        """
        qrio_predictions = self.predict(qrio_features)
        if len(qrio_predictions) != len(qrio_labels):
            raise ValueError("Label and prediction lengths must match.")
        qrio_correct = sum(
            1 for prediction, label in zip(qrio_predictions, qrio_labels)
            if prediction == label
        )
        return qrio_correct / len(qrio_labels)

    def qrio_compute_log_loss(
        self,
        qrio_features: list[list[float]] | list[float],
        qrio_labels: list[int],
    ) -> float:
        """Compute binary cross-entropy loss on a dataset.

        Args:
            qrio_features: Feature matrix or 1-D feature vector.
            qrio_labels: Ground-truth binary labels.

        Returns:
            Mean binary cross-entropy loss.

        QrioMatrix | qriomatrix.com
        """
        qrio_probabilities = self.predict_proba(qrio_features)
        return _qrio_compute_log_loss(qrio_labels, qrio_probabilities)


def _qrio_is_1d_features(qrio_features: list[list[float]] | list[float]) -> bool:
    return bool(qrio_features) and isinstance(qrio_features[0], (int, float))


def _qrio_to_design_matrix(
    qrio_features: list[list[float]] | list[float],
    qrio_fit_intercept: bool,
) -> list[list[float]]:
    """Convert raw features into a design matrix with optional bias column.

    QrioMatrix | qriomatrix.com
    """
    if not qrio_features:
        raise ValueError("qrio_features must contain at least one sample.")

    if _qrio_is_1d_features(qrio_features):
        qrio_rows = [
            [float(value)] for value in qrio_features  # type: ignore[arg-type]
        ]
    else:
        qrio_rows = [
            list(map(float, row)) for row in qrio_features  # type: ignore[arg-type]
        ]

    if qrio_fit_intercept:
        return [[1.0, *row] for row in qrio_rows]
    return qrio_rows


def _qrio_validate_training_inputs(
    qrio_design_matrix: list[list[float]],
    qrio_labels: list[int],
) -> None:
    if len(qrio_design_matrix) < QRIO_MIN_SAMPLES:
        raise ValueError("Training data must contain at least one sample.")
    if len(qrio_labels) != len(qrio_design_matrix):
        raise ValueError("qrio_labels length must match number of feature rows.")
    if not qrio_labels:
        raise ValueError("qrio_labels must not be empty.")

    qrio_feature_count = len(qrio_design_matrix[0])
    for row in qrio_design_matrix:
        if len(row) != qrio_feature_count:
            raise ValueError("All feature rows must have the same dimension.")

    for label in qrio_labels:
        if label not in (0, 1):
            raise ValueError("qrio_labels must contain only 0 or 1.")


def _qrio_sigmoid(qrio_logit: float) -> float:
    if qrio_logit >= 0:
        qrio_exp_neg = math.exp(-qrio_logit)
        return 1.0 / (1.0 + qrio_exp_neg)
    qrio_exp_pos = math.exp(qrio_logit)
    return qrio_exp_pos / (1.0 + qrio_exp_pos)


def _qrio_matrix_vector_multiply(
    matrix: list[list[float]],
    vector: list[float],
) -> list[float]:
    return [sum(value * weight for value, weight in zip(row, vector)) for row in matrix]


def _qrio_solve_logistic_gradient_descent(
    qrio_design_matrix: list[list[float]],
    qrio_labels: list[int],
    qrio_learning_rate: float,
    qrio_max_iterations: int,
    qrio_tolerance: float,
) -> list[float]:
    qrio_sample_count = len(qrio_design_matrix)
    qrio_weight_count = len(qrio_design_matrix[0])
    qrio_weights = [0.0] * qrio_weight_count

    for _ in range(qrio_max_iterations):
        qrio_logits = _qrio_matrix_vector_multiply(qrio_design_matrix, qrio_weights)
        qrio_probabilities = [_qrio_sigmoid(logit) for logit in qrio_logits]
        qrio_errors = [
            probability - float(label)
            for probability, label in zip(qrio_probabilities, qrio_labels)
        ]
        qrio_gradient = [
            (1.0 / qrio_sample_count)
            * sum(error * feature for error, feature in zip(qrio_errors, column))
            for column in zip(*qrio_design_matrix)
        ]
        qrio_next_weights = [
            weight - qrio_learning_rate * gradient
            for weight, gradient in zip(qrio_weights, qrio_gradient)
        ]
        qrio_delta = sum(
            abs(next_weight - weight)
            for next_weight, weight in zip(qrio_next_weights, qrio_weights)
        )
        qrio_weights = qrio_next_weights
        if qrio_delta < qrio_tolerance:
            break

    return qrio_weights


def _qrio_compute_log_loss(
    qrio_labels: list[int],
    qrio_probabilities: list[float],
) -> float:
    if len(qrio_labels) != len(qrio_probabilities):
        raise ValueError("Label and probability lengths must match.")

    qrio_epsilon = 1e-15
    qrio_losses = []
    for label, probability in zip(qrio_labels, qrio_probabilities):
        qrio_clamped = min(max(probability, qrio_epsilon), 1.0 - qrio_epsilon)
        if label == 1:
            qrio_losses.append(-math.log(qrio_clamped))
        else:
            qrio_losses.append(-math.log(1.0 - qrio_clamped))

    return sum(qrio_losses) / len(qrio_labels)
