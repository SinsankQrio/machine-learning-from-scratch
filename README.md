# Machine Learning from Scratch

Pure Python implementations of core machine learning algorithms — no NumPy, no scikit-learn, no PyTorch. Just lists, loops, and math so you can see exactly how each model learns.

Built for [QrioMatrix](https://qriomatrix.com) with the **Qrio Code Signature (QCS)** and **Qrio Test Case Definition (QTCD)** standards. Every module ships with a full pytest suite and practical test cases (XOR, linear boundaries, multi-class softmax, and more).

---

## Why from scratch?

Frameworks hide the details. This repo makes them visible:

- **Linear regression** — normal equation and gradient descent side by side
- **Logistic regression** — sigmoid activation and binary cross-entropy
- **Multi-layer perceptron** — ReLU hidden layers, backpropagation, sigmoid/softmax output

If you have ever wondered what `model.fit()` actually does, start here.

---

## Project structure

```
machine-learning-from-scratch/
├── src/
│   ├── linear_regression/
│   │   ├── qrio_ml_linear_regression.py
│   │   └── test_qrio_ml_linear_regression.py
│   ├── logistic_regression/
│   │   ├── qrio_ml_logistic_regression.py
│   │   ├── qrio_aws_lambda_deploy_handler.py
│   │   ├── test_qrio_ml_logistic_regression.py
│   │   └── docs/
│   │       └── qrio_aws_lambda_deploy_handler.md
│   └── multi_layer_perceptron/
│       ├── qrio_ml_multi_layer_perceptron.py
│       └── test_qrio_ml_multi_layer_perceptron.py
├── pyproject.toml
└── README.md
```

Each algorithm lives in its own directory. Modules are imported directly (no package wrapper) — pytest adds each `src/<module>/` path automatically via `pyproject.toml`.

---

## Quick start

### Requirements

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/) or pip for environment management

### Setup

```bash
git clone https://github.com/SinsankQrio/machine-learning-from-scratch.git
cd machine-learning-from-scratch

# Using uv (recommended)
uv sync
uv run pip install pytest

# Or with a standard venv
python -m venv .venv
source .venv/bin/activate
pip install pytest
```

### Run all tests

```bash
pytest -v
```

Run tests for a single module:

```bash
pytest src/linear_regression/ -v
pytest src/logistic_regression/ -v
pytest src/multi_layer_perceptron/ -v
```

---

## Modules

### 1. Linear regression

**File:** `src/linear_regression/qrio_ml_linear_regression.py`  
**Class:** `QrioLinearRegression`

Ordinary least squares for univariate and multivariate data. Supports two solvers:

| Method | Description |
|--------|-------------|
| `normal` | Closed-form solution via the normal equation (default) |
| `gradient_descent` | Iterative weight updates with configurable learning rate |

**Example — recover y = 2x + 1:**

```python
from qrio_ml_linear_regression import QrioLinearRegression

model = QrioLinearRegression(qrio_method="normal")
model.fit([0, 1, 2, 3, 4], [1, 3, 5, 7, 9])

print(model.predict([5.0]))   # [11.0]
print(model.score([0, 1, 2], [1, 3, 5]))  # R² ≈ 1.0
```

**Key API**

| Method | Purpose |
|--------|---------|
| `fit(features, targets)` | Train weights |
| `predict(features)` | Return continuous predictions |
| `score(features, targets)` | R² coefficient of determination |
| `qrio_compute_mse(features, targets)` | Mean squared error |

Features accept either a 1-D list (`[x1, x2, …]`) or a 2-D matrix (`[[x1, y1], …]`).

---

### 2. Logistic regression

**File:** `src/logistic_regression/qrio_ml_logistic_regression.py`  
**Class:** `QrioLogisticRegression`

Binary classifier using the sigmoid function and gradient descent.

**Example — classify by threshold:**

```python
from qrio_ml_logistic_regression import QrioLogisticRegression

model = QrioLogisticRegression(qrio_learning_rate=0.5)
model.fit([0, 1, 2, 3, 4], [0, 0, 0, 1, 1])

print(model.predict([3.0]))        # [1]
print(model.predict_proba([3.0]))  # [0.87…]
print(model.score([0, 1, 2, 3, 4], [0, 0, 0, 1, 1]))  # 1.0
```

**Key API**

| Method | Purpose |
|--------|---------|
| `fit(features, labels)` | Train on binary labels (0 or 1) |
| `predict(features)` | Hard class labels |
| `predict_proba(features)` | Positive-class probabilities |
| `score(features, labels)` | Classification accuracy |
| `qrio_compute_log_loss(features, labels)` | Binary cross-entropy loss |

**AWS Lambda deploy handler**

The logistic regression module includes a production-ready Lambda handler at `qrio_aws_lambda_deploy_handler.py`. It exposes stateless `fit` and `predict` endpoints. See the [deploy runbook](src/logistic_regression/docs/qrio_aws_lambda_deploy_handler.md) for packaging and deployment steps.

---

### 3. Multi-layer perceptron

**File:** `src/multi_layer_perceptron/qrio_ml_multi_layer_perceptron.py`  
**Class:** `QrioMultiLayerPerceptron`

Feed-forward neural network trained with backpropagation. Handles non-linear problems that linear models cannot solve (e.g. XOR).

| Layer | Activation |
|-------|------------|
| Hidden | ReLU |
| Output (binary) | Sigmoid |
| Output (multi-class) | Softmax |

**Example — learn XOR:**

```python
from qrio_ml_multi_layer_perceptron import QrioMultiLayerPerceptron

model = QrioMultiLayerPerceptron(
    qrio_hidden_layers=[4],
    qrio_learning_rate=0.5,
    qrio_max_iterations=20000,
)

features = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels   = [0, 1, 1, 0]

model.fit(features, labels)
print(model.predict(features))  # [0, 1, 1, 0]
print(model.score(features, labels))  # 1.0
```

**Key API**

| Method | Purpose |
|--------|---------|
| `fit(features, labels)` | Train via backpropagation |
| `predict(features)` | Integer class labels |
| `predict_proba(features)` | Per-class probability vectors |
| `score(features, labels)` | Classification accuracy |
| `qrio_compute_cross_entropy_loss(features, labels)` | Mean cross-entropy loss |

**Constructor highlights**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `qrio_hidden_layers` | `[4]` | Neurons per hidden layer |
| `qrio_learning_rate` | `0.5` | Gradient descent step size |
| `qrio_max_iterations` | `20000` | Training epoch cap |
| `qrio_random_seed` | `42` | Reproducible weight init (Xavier-style) |

---

## Test suite

Every module has a **QTCD-compliant** pytest suite with Given/When/Then docstrings and unique test IDs (`QRIO-TC-0001`, …).

| Module | Tests | Highlights |
|--------|-------|------------|
| Linear regression | 10 | Normal equation recovery, gradient descent, multivariate R² |
| Logistic regression | 10 | Separable data, multivariate, noisy boundaries, error handling |
| Multi-layer perceptron | 10 | XOR, softmax 3-class, loss reduction, unfitted guards |

```bash
# Run with coverage-style verbosity
pytest -v --tb=short

# Run a specific test by name
pytest -k "xor" -v
```

---

## Design conventions

All application code follows **Qrio Code Signature (QCS)**:

- Module headers include Author, Org, Domain, and Module name
- Public classes use the `Qrio` prefix (e.g. `QrioLinearRegression`)
- Domain variables use the `qrio_` prefix; constants use `QRIO_`
- Private helpers use `_qrio_` prefix
- Imports sorted with isort; linted with flake8 (88-char line limit)

Tests follow **Qrio Test Case Definition (QTCD)**:

- Files named `test_qrio_<module>.py`
- Classes named `TestQrio<ModuleCamelCase>`
- Methods named `test_qrio_<feature>_<scenario>_<expected>`

---

## Algorithm comparison

| | Linear regression | Logistic regression | Multi-layer perceptron |
|---|---|---|---|
| **Task** | Regression | Binary classification | Binary / multi-class |
| **Output** | Continuous | Probability → class | Probability → class |
| **Solver** | Normal equation or GD | Gradient descent | Backpropagation |
| **Non-linear data** | No | No (linear boundary) | Yes |
| **Dependencies** | stdlib only | stdlib only | stdlib only |

---

## Contributing

1. Fork the repo and create a branch with the `qrio/` prefix.
2. Follow QCS for application code and QTCD for tests.
3. Run `pytest -v` and ensure all tests pass.
4. Open a pull request against `main`.

For automated code quality validation, see the MCP servers in `mcp/qrio-code-review` and `mcp/qrio-auto-pr`.

---

## License

This project is maintained by [QrioMatrix](https://qriomatrix.com).

**Author:** Sineth Sankalpa, AI/ML Engineer
