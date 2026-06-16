"""
Univariate and multivariate linear regression from scratch.

Author: Sineth Sankalpa, AI/ML Engineer
Org: qriomatrix.com
Domain: application
Module: qrio_ml_linear_regression
"""

from __future__ import annotations

QRIO_DEFAULT_LEARNING_RATE = 0.01
QRIO_DEFAULT_MAX_ITERATIONS = 10000
QRIO_DEFAULT_TOLERANCE = 1e-6
QRIO_MIN_SAMPLES = 1


class QrioLinearRegression:
    """Ordinary least squares and gradient-descent linear regression.

    QrioMatrix | qriomatrix.com
    """

    def __init__(
        self,
        qrio_learning_rate: float = QRIO_DEFAULT_LEARNING_RATE,
        qrio_max_iterations: int = QRIO_DEFAULT_MAX_ITERATIONS,
        qrio_tolerance: float = QRIO_DEFAULT_TOLERANCE,
        qrio_fit_intercept: bool = True,
        qrio_method: str = "normal",
    ) -> None:
        """Initialize the regression model.

        Args:
            qrio_learning_rate: Step size for gradient descent.
            qrio_max_iterations: Maximum gradient-descent iterations.
            qrio_tolerance: Convergence threshold for gradient descent.
            qrio_fit_intercept: Whether to learn a bias term.
            qrio_method: ``normal`` or ``gradient_descent``.

        QrioMatrix | qriomatrix.com
        """
        self.qrio_learning_rate = qrio_learning_rate
        self.qrio_max_iterations = qrio_max_iterations
        self.qrio_tolerance = qrio_tolerance
        self.qrio_fit_intercept = qrio_fit_intercept
        self.qrio_method = qrio_method
        self.qrio_weights: list[float] | None = None

    def fit(
        self,
        qrio_features: list[list[float]] | list[float],
        qrio_targets: list[float],
    ) -> QrioLinearRegression:
        """Fit regression weights to training data.

        Args:
            qrio_features: Feature matrix or 1-D feature vector.
            qrio_targets: Target values aligned with feature rows.

        Returns:
            Fitted model instance.

        QrioMatrix | qriomatrix.com
        """
        qrio_design_matrix = _qrio_to_design_matrix(
            qrio_features,
            self.qrio_fit_intercept,
        )
        _qrio_validate_training_inputs(qrio_design_matrix, qrio_targets)

        if self.qrio_method == "normal":
            self.qrio_weights = _qrio_solve_normal_equation(
                qrio_design_matrix,
                qrio_targets,
            )
        elif self.qrio_method == "gradient_descent":
            self.qrio_weights = _qrio_solve_gradient_descent(
                qrio_design_matrix,
                qrio_targets,
                self.qrio_learning_rate,
                self.qrio_max_iterations,
                self.qrio_tolerance,
            )
        else:
            raise ValueError(
                f"Unsupported qrio_method '{self.qrio_method}'. "
                "Use 'normal' or 'gradient_descent'."
            )
        return self

    def predict(
        self,
        qrio_features: list[list[float]] | list[float],
    ) -> list[float]:
        """Predict target values for feature rows.

        Args:
            qrio_features: Feature matrix or 1-D feature vector.

        Returns:
            Predicted target values.

        QrioMatrix | qriomatrix.com
        """
        if self.qrio_weights is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        qrio_design_matrix = _qrio_to_design_matrix(
            qrio_features,
            self.qrio_fit_intercept,
        )
        return _qrio_matrix_vector_multiply(qrio_design_matrix, self.qrio_weights)

    def score(
        self,
        qrio_features: list[list[float]] | list[float],
        qrio_targets: list[float],
    ) -> float:
        """Compute the coefficient of determination (R-squared).

        Args:
            qrio_features: Feature matrix or 1-D feature vector.
            qrio_targets: Ground-truth target values.

        Returns:
            R-squared score in the range (-inf, 1].

        QrioMatrix | qriomatrix.com
        """
        qrio_predictions = self.predict(qrio_features)
        return _qrio_compute_r2(qrio_targets, qrio_predictions)

    def qrio_compute_mse(
        self,
        qrio_features: list[list[float]] | list[float],
        qrio_targets: list[float],
    ) -> float:
        """Compute mean squared error on a dataset.

        Args:
            qrio_features: Feature matrix or 1-D feature vector.
            qrio_targets: Ground-truth target values.

        Returns:
            Mean squared error.

        QrioMatrix | qriomatrix.com
        """
        qrio_predictions = self.predict(qrio_features)
        return _qrio_compute_mse(qrio_targets, qrio_predictions)


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
    qrio_targets: list[float],
) -> None:
    if len(qrio_design_matrix) < QRIO_MIN_SAMPLES:
        raise ValueError("Training data must contain at least one sample.")
    if len(qrio_targets) != len(qrio_design_matrix):
        raise ValueError("qrio_targets length must match number of feature rows.")
    if not qrio_targets:
        raise ValueError("qrio_targets must not be empty.")

    qrio_feature_count = len(qrio_design_matrix[0])
    for row in qrio_design_matrix:
        if len(row) != qrio_feature_count:
            raise ValueError("All feature rows must have the same dimension.")


def _qrio_transpose(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        return []
    return [list(column) for column in zip(*matrix)]


def _qrio_matrix_multiply(
    left: list[list[float]],
    right: list[list[float]],
) -> list[list[float]]:
    qrio_right_transpose = _qrio_transpose(right)
    return [
        [sum(a * b for a, b in zip(row, col)) for col in qrio_right_transpose]
        for row in left
    ]


def _qrio_matrix_vector_multiply(
    matrix: list[list[float]],
    vector: list[float],
) -> list[float]:
    return [sum(value * weight for value, weight in zip(row, vector)) for row in matrix]


def _qrio_invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [row[:] + [1.0 if index == row_index else 0.0 for index in range(size)]
                 for row_index, row in enumerate(matrix)]

    for pivot_index in range(size):
        pivot_row = max(
            range(pivot_index, size),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        pivot_value = augmented[pivot_row][pivot_index]
        if abs(pivot_value) < QRIO_DEFAULT_TOLERANCE:
            raise ValueError("Design matrix is singular and cannot be inverted.")

        augmented[pivot_index], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[pivot_index],
        )
        pivot_row_values = augmented[pivot_index]
        scale = pivot_value
        augmented[pivot_index] = [value / scale for value in pivot_row_values]

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            augmented[row_index] = [
                left - factor * right
                for left, right in zip(augmented[row_index], augmented[pivot_index])
            ]

    return [row[size:] for row in augmented]


def _qrio_solve_normal_equation(
    qrio_design_matrix: list[list[float]],
    qrio_targets: list[float],
) -> list[float]:
    qrio_x_transpose = _qrio_transpose(qrio_design_matrix)
    qrio_xtx = _qrio_matrix_multiply(qrio_x_transpose, qrio_design_matrix)
    qrio_xty = [
        sum(feature * target for feature, target in zip(row, qrio_targets))
        for row in qrio_x_transpose
    ]
    qrio_inverse = _qrio_invert_matrix(qrio_xtx)
    return _qrio_matrix_vector_multiply(qrio_inverse, qrio_xty)


def _qrio_solve_gradient_descent(
    qrio_design_matrix: list[list[float]],
    qrio_targets: list[float],
    qrio_learning_rate: float,
    qrio_max_iterations: int,
    qrio_tolerance: float,
) -> list[float]:
    qrio_sample_count = len(qrio_design_matrix)
    qrio_weight_count = len(qrio_design_matrix[0])
    qrio_weights = [0.0] * qrio_weight_count

    for _ in range(qrio_max_iterations):
        qrio_predictions = _qrio_matrix_vector_multiply(
            qrio_design_matrix,
            qrio_weights,
        )
        qrio_errors = [
            prediction - target
            for prediction, target in zip(qrio_predictions, qrio_targets)
        ]
        qrio_gradient = [
            (2.0 / qrio_sample_count)
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


def _qrio_compute_mse(
    qrio_targets: list[float],
    qrio_predictions: list[float],
) -> float:
    if len(qrio_targets) != len(qrio_predictions):
        raise ValueError("Target and prediction lengths must match.")
    qrio_sample_count = len(qrio_targets)
    return sum(
        (target - prediction) ** 2
        for target, prediction in zip(qrio_targets, qrio_predictions)
    ) / qrio_sample_count


def _qrio_compute_r2(qrio_targets: list[float], qrio_predictions: list[float]) -> float:
    qrio_mean = sum(qrio_targets) / len(qrio_targets)
    qrio_total_variance = sum((target - qrio_mean) ** 2 for target in qrio_targets)
    if qrio_total_variance == 0.0:
        return 1.0 if _qrio_compute_mse(qrio_targets, qrio_predictions) == 0.0 else 0.0
    qrio_residual_variance = sum(
        (target - prediction) ** 2
        for target, prediction in zip(qrio_targets, qrio_predictions)
    )
    return 1.0 - (qrio_residual_variance / qrio_total_variance)
