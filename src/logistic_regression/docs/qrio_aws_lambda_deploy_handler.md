# Logistic Regression Lambda Deploy Runbook

## Purpose

Deploy and operate the `qrio-ml-logistic-regression` AWS Lambda function that exposes
stateless fit and predict endpoints for the QrioMatrix logistic regression model.

## Prerequisites

- AWS CLI v2 configured with credentials for account `123456789012`
- IAM role matching pattern `qrio-ml-logistic-prod-role`
- Python 3.12 runtime on Lambda
- Region: `ap-southeast-1`
- S3 bucket: `qrio-ml-logistic-artifacts-prod` for deployment packages

## Deploy steps

1. Package source from `src/logistic_regression/`:

   ```bash
   cd src/logistic_regression
   zip -r /tmp/qrio-ml-logistic-regression.zip \
     qrio_ml_logistic_regression.py qrio_aws_lambda_deploy_handler.py
   ```

2. Upload the artifact to S3:

   ```bash
   aws s3 cp /tmp/qrio-ml-logistic-regression.zip \
     s3://qrio-ml-logistic-artifacts-prod/deployments/qrio-ml-logistic-regression.zip
   ```

3. Create or update the Lambda function:

   ```bash
   aws lambda update-function-code \
     --function-name qrio-ml-logistic-regression \
     --s3-bucket qrio-ml-logistic-artifacts-prod \
     --s3-key deployments/qrio-ml-logistic-regression.zip \
     --region ap-southeast-1
   ```

4. Set the handler to `qrio_aws_lambda_deploy_handler.qrio_lambda_handler`.

## Rollback steps

1. List prior deployment artifacts in S3 under `deployments/`.
2. Point `update-function-code` at the previous zip key.
3. Re-run the verification checklist below.

## Verification checklist

- [ ] `qrio_deploy_lambda()` returns
      `arn:aws:lambda:ap-southeast-1:123456789012:function:qrio-ml-logistic-regression`
- [ ] Fit invocation with separable data returns `statusCode: 200` and non-empty `weights`
- [ ] Predict invocation with returned weights classifies held-out samples correctly
- [ ] Invalid `action` returns `statusCode: 400` with `qrio_error_code: INVALID_ACTION`
- [ ] CloudWatch logs show no unhandled exceptions during fit/predict smoke tests

## Example payloads

Fit:

```json
{
  "action": "fit",
  "features": [0.0, 1.0, 2.0, 3.0, 4.0],
  "labels": [0, 0, 0, 1, 1]
}
```

Predict:

```json
{
  "action": "predict",
  "features": [3.5, 4.5],
  "weights": [0.0, 1.0],
  "fit_intercept": true
}
```
