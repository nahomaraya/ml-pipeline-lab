# -*- coding: utf-8 -*-
"""
Main entrypoint to use the project like a library or CLI.

Examples:
  python main.py --csv accidents.csv --model dt --visualize
  python main.py --csv accidents.csv --model rf
  python main.py --csv accidents.csv --model lin --income 33000
  python main.py --csv accidents.csv --model logit
"""

import argparse
from typing import Optional

from data_preprocessing import preprocess_data
from linear_regression import train_linear_regression
from logistic_regression_model import train_logistic_regression
from decision_tree_model import train_decision_tree
from random_forest_model import train_random_forest


def run_pipeline(csv_path: str,
                 model_name: str,
                 visualize: bool = False,
                 income: Optional[float] = None,
                 predict_feature_name: Optional[str] = None,
                 predict_feature_value: Optional[float] = None):
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(csv_path, verbose=True)

    if model_name == "lin":
        pred_feature = predict_feature_name or ""
        try:
            pred_value = float(predict_feature_value) if predict_feature_value is not None else 0.0
        except (ValueError, TypeError):
            pred_value = 0.0

        model, pred = train_linear_regression(
            X_train, X_val, y_train,
            predict_feature_name=pred_feature,
            predict_feature_value=pred_value
        )
        print(f"Predicted value for {pred_feature}={pred_value}: {pred:.4f}")
        return model

    if model_name == "logit":
        model, result = train_logistic_regression(X_train, X_val, y_train, y_val)
        return model
    if model_name == "dt":
        model = train_decision_tree(X_train, X_val, y_train, y_val, visualize=visualize)
        return model
    if model_name == "rf":
        model = train_random_forest(X_train, X_val, y_train, y_val, visualize_tree=visualize)
        return model

    raise ValueError("Unknown model. Choose from: lin, logit, dt, rf")


def main():
    parser = argparse.ArgumentParser(description="ML pipeline")
    parser.add_argument("--csv", dest="csv_path", default="accidents.csv", help="Path to CSV dataset")
    parser.add_argument("--model", dest="model", required=True, choices=["lin", "logit", "dt", "rf"], help="Model to run")
    parser.add_argument("--visualize", action="store_true", help="Enable model visualization where supported")
    parser.add_argument("--predict-feature", dest="predict_feature_name", default=None, help="Feature to predict for linear regression")
    parser.add_argument("--predict-value", type=float, dest="predict_feature_value", default=None, help="Value of the feature to predict for linear regression")
    args = parser.parse_args()

    run_pipeline(args.csv_path, args.model, visualize=args.visualize, predict_feature_name=args.predict_feature_name, predict_feature_value=args.predict_feature_value)


if __name__ == "__main__":
    main()


