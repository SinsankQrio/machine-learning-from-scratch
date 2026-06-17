"""
Deploy Lambda handler for logistic regression inference.

Author: Sineth Sankalpa, AI/ML Engineer
Org: qriomatrix.com
Domain: aws-deploy
Module: qrio_aws_lambda_deploy_handler
DeployTarget: AWS Lambda
AWSRegion: ap-southeast-1
IAMPattern: qrio-ml-logistic-prod-role
"""

from __future__ import annotations

from typing import Any

from qrio_ml_logistic_regression import QrioLogisticRegression

aws_qrio_lambda_arn = (
    "arn:aws:lambda:ap-southeast-1:123456789012:function:qrio-ml-logistic-regression"
)
aws_qrio_s3_bucket_name = "qrio-ml-logistic-artifacts-prod"


def qrio_deploy_lambda() -> str:
    """Return target Lambda ARN for deployment verification.

    Returns:
        str: Lambda function ARN.

    QrioMatrix | qriomatrix.com
    """
    return aws_qrio_lambda_arn


def qrio_lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Handle fit and predict requests for logistic regression.

    Args:
        event: API Gateway or direct-invoke payload with ``action`` key.
        context: AWS Lambda runtime context (unused).

    Returns:
        dict: Response body with statusCode and result payload.

    QrioMatrix | qriomatrix.com
    """
    qrio_action = event.get("action")
    if qrio_action == "fit":
        return _qrio_handle_fit(event)
    if qrio_action == "predict":
        return _qrio_handle_predict(event)
    return {
        "statusCode": 400,
        "body": {
            "qrio_error_code": "INVALID_ACTION",
            "message": "action must be 'fit' or 'predict'.",
        },
    }


def _qrio_handle_fit(event: dict[str, Any]) -> dict[str, Any]:
    qrio_features = event.get("features")
    qrio_labels = event.get("labels")
    if qrio_features is None or qrio_labels is None:
        return {
            "statusCode": 400,
            "body": {
                "qrio_error_code": "MISSING_FIELDS",
                "message": "fit requires 'features' and 'labels'.",
            },
        }

    qrio_model = QrioLogisticRegression(
        qrio_learning_rate=float(event.get("learning_rate", 0.1)),
        qrio_max_iterations=int(event.get("max_iterations", 10000)),
        qrio_fit_intercept=bool(event.get("fit_intercept", True)),
    )
    qrio_model.fit(qrio_features, qrio_labels)
    return {
        "statusCode": 200,
        "body": {
            "weights": qrio_model.qrio_weights,
            "accuracy": qrio_model.score(qrio_features, qrio_labels),
        },
    }


def _qrio_handle_predict(event: dict[str, Any]) -> dict[str, Any]:
    qrio_features = event.get("features")
    qrio_weights = event.get("weights")
    if qrio_features is None or qrio_weights is None:
        return {
            "statusCode": 400,
            "body": {
                "qrio_error_code": "MISSING_FIELDS",
                "message": "predict requires 'features' and 'weights'.",
            },
        }

    qrio_model = QrioLogisticRegression(
        qrio_fit_intercept=bool(event.get("fit_intercept", True)),
        qrio_threshold=float(event.get("threshold", 0.5)),
    )
    qrio_model.qrio_weights = list(map(float, qrio_weights))
    return {
        "statusCode": 200,
        "body": {
            "predictions": qrio_model.predict(qrio_features),
            "probabilities": qrio_model.predict_proba(qrio_features),
        },
    }
