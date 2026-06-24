"""
Multi-layer perceptron from scratch using backpropagation.

Author: Sineth Sankalpa, AI/ML Engineer
Org: qriomatrix.com
Domain: application
Module: qrio_ml_multi_layer_perceptron
"""

from __future__ import annotations

import math
import random

QRIO_DEFAULT_LEARNING_RATE = 0.5
QRIO_DEFAULT_MAX_ITERATIONS = 20000
QRIO_DEFAULT_TOLERANCE = 1e-6
QRIO_DEFAULT_THRESHOLD = 0.5
QRIO_DEFAULT_HIDDEN_LAYERS = [4]
QRIO_MIN_SAMPLES = 1


class QrioMultiLayerPerceptron:
    """Feed-forward neural network with ReLU hidden and sigmoid/softmax output.

    QrioMatrix | qriomatrix.com
    """

    def __init__(
        self,
        qrio_hidden_layers: list[int] | None = None,
        qrio_learning_rate: float = QRIO_DEFAULT_LEARNING_RATE,
        qrio_max_iterations: int = QRIO_DEFAULT_MAX_ITERATIONS,
        qrio_tolerance: float = QRIO_DEFAULT_TOLERANCE,
        qrio_threshold: float = QRIO_DEFAULT_THRESHOLD,
        qrio_random_seed: int | None = 42,
    ) -> None:
        """Initialize the multi-layer perceptron.

        Args:
            qrio_hidden_layers: Hidden layer neuron counts, e.g. ``[4, 8]``.
            qrio_learning_rate: Step size for gradient descent.
            qrio_max_iterations: Maximum training epochs.
            qrio_tolerance: Convergence threshold for weight updates.
            qrio_threshold: Decision threshold for binary classification.
            qrio_random_seed: Seed for reproducible weight initialization.

        QrioMatrix | qriomatrix.com
        """
        self.qrio_hidden_layers = (
            list(qrio_hidden_layers)
            if qrio_hidden_layers is not None
            else list(QRIO_DEFAULT_HIDDEN_LAYERS)
        )
        self.qrio_learning_rate = qrio_learning_rate
        self.qrio_max_iterations = qrio_max_iterations
        self.qrio_tolerance = qrio_tolerance
        self.qrio_threshold = qrio_threshold
        self.qrio_random_seed = qrio_random_seed
        self.qrio_layer_weights: list[list[list[float]]] | None = None
        self.qrio_num_classes: int = 2

    def fit(
        self,
        qrio_features: list[list[float]] | list[float],
        qrio_labels: list[int],
    ) -> QrioMultiLayerPerceptron:
        """Fit network weights to classification data via backpropagation.

        Args:
            qrio_features: Feature matrix or 1-D feature vector.
            qrio_labels: Integer class labels aligned with feature rows.

        Returns:
            Fitted model instance.

        QrioMatrix | qriomatrix.com
        """
        qrio_design_matrix = _qrio_to_design_matrix(qrio_features)
        _qrio_validate_training_inputs(qrio_design_matrix, qrio_labels)
        self.qrio_num_classes = len(set(qrio_labels))
        qrio_layer_sizes = _qrio_build_layer_sizes(
            len(qrio_design_matrix[0]),
            self.qrio_hidden_layers,
            self.qrio_num_classes,
        )
        self.qrio_layer_weights = _qrio_initialize_weights(
            qrio_layer_sizes,
            self.qrio_random_seed,
        )
        qrio_one_hot = _qrio_one_hot_encode(qrio_labels, self.qrio_num_classes)
        self.qrio_layer_weights = _qrio_train_backpropagation(
            qrio_design_matrix,
            qrio_one_hot,
            self.qrio_layer_weights,
            self.qrio_learning_rate,
            self.qrio_max_iterations,
            self.qrio_tolerance,
            self.qrio_num_classes,
        )
        return self

    def predict_proba(
        self,
        qrio_features: list[list[float]] | list[float],
    ) -> list[list[float]]:
        """Predict class probabilities for feature rows.

        Args:
            qrio_features: Feature matrix or 1-D feature vector.

        Returns:
            Per-sample probability vectors summing to 1.

        QrioMatrix | qriomatrix.com
        """
        if self.qrio_layer_weights is None:
            raise RuntimeError(
                "Model is not fitted. Call fit() before predict_proba()."
            )

        qrio_design_matrix = _qrio_to_design_matrix(qrio_features)
        qrio_probabilities = []
        for qrio_sample in qrio_design_matrix:
            qrio_output = _qrio_forward_pass(
                qrio_sample,
                self.qrio_layer_weights,
                self.qrio_num_classes,
            )["output"]
            qrio_probabilities.append(qrio_output)
        return qrio_probabilities

    def predict(
        self,
        qrio_features: list[list[float]] | list[float],
    ) -> list[int]:
        """Predict class labels for feature rows.

        Args:
            qrio_features: Feature matrix or 1-D feature vector.

        Returns:
            Predicted integer class labels.

        QrioMatrix | qriomatrix.com
        """
        qrio_probabilities = self.predict_proba(qrio_features)
        if self.qrio_num_classes == 2:
            return [
                1 if probabilities[0] >= self.qrio_threshold else 0
                for probabilities in qrio_probabilities
            ]
        return [
            max(range(len(probabilities)), key=probabilities.__getitem__)
            for probabilities in qrio_probabilities
        ]

    def score(
        self,
        qrio_features: list[list[float]] | list[float],
        qrio_labels: list[int],
    ) -> float:
        """Compute classification accuracy on a dataset.

        Args:
            qrio_features: Feature matrix or 1-D feature vector.
            qrio_labels: Ground-truth class labels.

        Returns:
            Fraction of correctly classified samples.

        QrioMatrix | qriomatrix.com
        """
        qrio_predictions = self.predict(qrio_features)
        if len(qrio_predictions) != len(qrio_labels):
            raise ValueError("Label and prediction lengths must match.")
        qrio_correct = sum(
            1
            for prediction, label in zip(qrio_predictions, qrio_labels)
            if prediction == label
        )
        return qrio_correct / len(qrio_labels)

    def qrio_compute_cross_entropy_loss(
        self,
        qrio_features: list[list[float]] | list[float],
        qrio_labels: list[int],
    ) -> float:
        """Compute mean cross-entropy loss on a dataset.

        Args:
            qrio_features: Feature matrix or 1-D feature vector.
            qrio_labels: Ground-truth class labels.

        Returns:
            Mean cross-entropy loss.

        QrioMatrix | qriomatrix.com
        """
        qrio_probabilities = self.predict_proba(qrio_features)
        return _qrio_compute_cross_entropy_loss(qrio_labels, qrio_probabilities)


def _qrio_is_1d_features(qrio_features: list[list[float]] | list[float]) -> bool:
    return bool(qrio_features) and isinstance(qrio_features[0], (int, float))


def _qrio_to_design_matrix(
    qrio_features: list[list[float]] | list[float],
) -> list[list[float]]:
    """Convert raw features into a design matrix with a leading bias column.

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

    return [[1.0, *row] for row in qrio_rows]


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


def _qrio_build_layer_sizes(
    qrio_input_size: int,
    qrio_hidden_layers: list[int],
    qrio_num_classes: int,
) -> list[int]:
    qrio_output_size = 1 if qrio_num_classes == 2 else qrio_num_classes
    return [qrio_input_size, *qrio_hidden_layers, qrio_output_size]


def _qrio_initialize_weights(
    qrio_layer_sizes: list[int],
    qrio_random_seed: int | None,
) -> list[list[list[float]]]:
    if qrio_random_seed is not None:
        random.seed(qrio_random_seed)

    qrio_weights = []
    for qrio_layer_index in range(len(qrio_layer_sizes) - 1):
        qrio_input_size = qrio_layer_sizes[qrio_layer_index]
        qrio_output_size = qrio_layer_sizes[qrio_layer_index + 1]
        qrio_limit = math.sqrt(6.0 / (qrio_input_size + qrio_output_size))
        qrio_layer = [
            [
                random.uniform(-qrio_limit, qrio_limit)
                for _ in range(qrio_input_size + 1)
            ]
            for _ in range(qrio_output_size)
        ]
        qrio_weights.append(qrio_layer)
    return qrio_weights


def _qrio_one_hot_encode(
    qrio_labels: list[int],
    qrio_num_classes: int,
) -> list[list[float]]:
    if qrio_num_classes == 2:
        return [[float(label)] for label in qrio_labels]
    return [
        [
            1.0 if class_index == label else 0.0
            for class_index in range(qrio_num_classes)
        ]
        for label in qrio_labels
    ]


def _qrio_sigmoid(qrio_value: float) -> float:
    if qrio_value >= 0:
        qrio_exp_neg = math.exp(-qrio_value)
        return 1.0 / (1.0 + qrio_exp_neg)
    qrio_exp_pos = math.exp(qrio_value)
    return qrio_exp_pos / (1.0 + qrio_exp_pos)


def _qrio_relu(qrio_value: float) -> float:
    return max(0.0, qrio_value)


def _qrio_softmax(qrio_logits: list[float]) -> list[float]:
    qrio_max_logit = max(qrio_logits)
    qrio_exp_values = [math.exp(logit - qrio_max_logit) for logit in qrio_logits]
    qrio_sum = sum(qrio_exp_values)
    return [value / qrio_sum for value in qrio_exp_values]


def _qrio_linear_layer(
    qrio_input: list[float],
    qrio_weights: list[list[float]],
) -> list[float]:
    return [
        sum(weight * activation for weight, activation in zip(neuron, qrio_input))
        for neuron in qrio_weights
    ]


def _qrio_add_bias(qrio_activation: list[float]) -> list[float]:
    return [1.0, *qrio_activation]


def _qrio_forward_pass(
    qrio_sample: list[float],
    qrio_layer_weights: list[list[list[float]]],
    qrio_num_classes: int,
) -> dict[str, list[list[float]] | list[float]]:
    qrio_activations: list[list[float]] = [qrio_sample]
    qrio_pre_activations: list[list[float]] = []

    qrio_current = qrio_sample
    for qrio_layer_index, qrio_weights in enumerate(qrio_layer_weights):
        qrio_z = _qrio_linear_layer(qrio_current, qrio_weights)
        qrio_pre_activations.append(qrio_z)
        qrio_is_output = qrio_layer_index == len(qrio_layer_weights) - 1
        if qrio_is_output:
            if qrio_num_classes == 2:
                qrio_a = [_qrio_sigmoid(qrio_z[0])]
            else:
                qrio_a = _qrio_softmax(qrio_z)
        else:
            qrio_a = [_qrio_relu(value) for value in qrio_z]
            qrio_a = _qrio_add_bias(qrio_a)
        qrio_activations.append(qrio_a)
        qrio_current = qrio_a

    return {
        "activations": qrio_activations,
        "pre_activations": qrio_pre_activations,
        "output": qrio_activations[-1],
    }


def _qrio_output_delta(
    qrio_output: list[float],
    qrio_target: list[float],
    qrio_num_classes: int,
) -> list[float]:
    if qrio_num_classes == 2:
        return [qrio_output[0] - qrio_target[0]]
    return [
        output - target
        for output, target in zip(qrio_output, qrio_target)
    ]


def _qrio_hidden_delta(
    qrio_next_delta: list[float],
    qrio_next_weights: list[list[float]],
    qrio_pre_activation: list[float],
) -> list[float]:
    qrio_weighted = [
        sum(
            qrio_next_delta[neuron_index]
            * qrio_next_weights[neuron_index][feature_index]
            for neuron_index in range(len(qrio_next_delta))
        )
        for feature_index in range(1, len(qrio_next_weights[0]))
    ]
    return [
        weighted * (1.0 if pre_activation > 0 else 0.0)
        for weighted, pre_activation in zip(qrio_weighted, qrio_pre_activation)
    ]


def _qrio_weight_gradient(
    qrio_delta: list[float],
    qrio_activation: list[float],
) -> list[list[float]]:
    return [
        [delta * activation for activation in qrio_activation]
        for delta in qrio_delta
    ]


def _qrio_apply_gradients(
    qrio_layer_weights: list[list[list[float]]],
    qrio_gradients: list[list[list[float]]],
    qrio_learning_rate: float,
) -> list[list[list[float]]]:
    return [
        [
            [
                weight - qrio_learning_rate * gradient
                for weight, gradient in zip(neuron, neuron_gradient)
            ]
            for neuron, neuron_gradient in zip(layer, layer_gradient)
        ]
        for layer, layer_gradient in zip(qrio_layer_weights, qrio_gradients)
    ]


def _qrio_backprop_sample(
    qrio_sample: list[float],
    qrio_target: list[float],
    qrio_layer_weights: list[list[list[float]]],
    qrio_num_classes: int,
) -> list[list[list[float]]]:
    qrio_forward = _qrio_forward_pass(
        qrio_sample,
        qrio_layer_weights,
        qrio_num_classes,
    )
    qrio_activations = qrio_forward["activations"]
    qrio_pre_activations = qrio_forward["pre_activations"]
    qrio_output = qrio_forward["output"]

    qrio_deltas: list[list[float]] = [[] for _ in qrio_layer_weights]
    qrio_deltas[-1] = _qrio_output_delta(
        qrio_output,  # type: ignore[arg-type]
        qrio_target,
        qrio_num_classes,
    )

    for qrio_layer_index in range(len(qrio_layer_weights) - 2, -1, -1):
        qrio_deltas[qrio_layer_index] = _qrio_hidden_delta(
            qrio_deltas[qrio_layer_index + 1],
            qrio_layer_weights[qrio_layer_index + 1],
            qrio_pre_activations[qrio_layer_index],
        )

    return [
        _qrio_weight_gradient(delta, activation)
        for delta, activation in zip(qrio_deltas, qrio_activations[:-1])
    ]


def _qrio_sum_gradients(
    qrio_left: list[list[list[float]]],
    qrio_right: list[list[list[float]]],
) -> list[list[list[float]]]:
    return [
        [
            [left + right for left, right in zip(left_neuron, right_neuron)]
            for left_neuron, right_neuron in zip(left_layer, right_layer)
        ]
        for left_layer, right_layer in zip(qrio_left, qrio_right)
    ]


def _qrio_scale_gradients(
    qrio_gradients: list[list[list[float]]],
    qrio_scale: float,
) -> list[list[list[float]]]:
    return [
        [
            [gradient * qrio_scale for gradient in neuron]
            for neuron in layer
        ]
        for layer in qrio_gradients
    ]


def _qrio_weight_delta(
    qrio_previous: list[list[list[float]]],
    qrio_updated: list[list[list[float]]],
) -> float:
    return sum(
        abs(new_weight - old_weight)
        for old_layer, new_layer in zip(qrio_previous, qrio_updated)
        for old_neuron, new_neuron in zip(old_layer, new_layer)
        for old_weight, new_weight in zip(old_neuron, new_neuron)
    )


def _qrio_train_backpropagation(
    qrio_design_matrix: list[list[float]],
    qrio_one_hot: list[list[float]],
    qrio_layer_weights: list[list[list[float]]],
    qrio_learning_rate: float,
    qrio_max_iterations: int,
    qrio_tolerance: float,
    qrio_num_classes: int,
) -> list[list[list[float]]]:
    qrio_sample_count = len(qrio_design_matrix)
    qrio_weights = qrio_layer_weights

    for _ in range(qrio_max_iterations):
        qrio_epoch_gradients = [
            [
                [[0.0] * len(neuron) for neuron in layer]
                for layer in qrio_weights
            ]
        ]
        qrio_epoch_gradients = qrio_epoch_gradients[0]

        for qrio_sample, qrio_target in zip(qrio_design_matrix, qrio_one_hot):
            qrio_sample_gradients = _qrio_backprop_sample(
                qrio_sample,
                qrio_target,
                qrio_weights,
                qrio_num_classes,
            )
            qrio_epoch_gradients = _qrio_sum_gradients(
                qrio_epoch_gradients,
                qrio_sample_gradients,
            )

        qrio_epoch_gradients = _qrio_scale_gradients(
            qrio_epoch_gradients,
            1.0 / qrio_sample_count,
        )
        qrio_next_weights = _qrio_apply_gradients(
            qrio_weights,
            qrio_epoch_gradients,
            qrio_learning_rate,
        )
        qrio_delta = _qrio_weight_delta(qrio_weights, qrio_next_weights)
        qrio_weights = qrio_next_weights
        if qrio_delta < qrio_tolerance:
            break

    return qrio_weights


def _qrio_compute_cross_entropy_loss(
    qrio_labels: list[int],
    qrio_probabilities: list[list[float]],
) -> float:
    if len(qrio_labels) != len(qrio_probabilities):
        raise ValueError("Label and probability lengths must match.")

    qrio_epsilon = 1e-15
    qrio_losses = []
    for label, probabilities in zip(qrio_labels, qrio_probabilities):
        if len(probabilities) == 1:
            qrio_probability = min(
                max(probabilities[0], qrio_epsilon),
                1.0 - qrio_epsilon,
            )
            if label == 1:
                qrio_losses.append(-math.log(qrio_probability))
            else:
                qrio_losses.append(-math.log(1.0 - qrio_probability))
        else:
            qrio_class_probability = min(
                max(probabilities[label], qrio_epsilon),
                1.0 - qrio_epsilon,
            )
            qrio_losses.append(-math.log(qrio_class_probability))

    return sum(qrio_losses) / len(qrio_labels)
