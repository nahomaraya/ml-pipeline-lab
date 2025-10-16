
import pandas as pd
import statsmodels.api as sm

__all__ = [
    "train_linear_regression",
]


def train_linear_regression(X_train: pd.DataFrame,
                            X_val: pd.DataFrame,
                            y_train,
                            income_value: float = 33000.0):
    """
    Train OLS linear regression and compute prediction at specified income value.

    Returns:
        model, prediction_at_income
    """
    X_train_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_const).fit()

    # Build representative input using mean and modes
    mean_values = X_train.mean(numeric_only=True)
    # Safe mode computation for all columns
    mode_values = X_train.mode(dropna=True).iloc[0]
    input_data = mean_values.copy()
    if predict_feature_name and predict_feature_name in X_train.columns and predict_feature_value is not None:
        input_data[predict_feature_name] = predict_feature_value
    # Set categorical-like one-hot groups to mode if present
    for col in X_train.columns:
        if col in mode_values.index and col not in input_data.index:
            input_data[col] = mode_values[col]

    input_df = pd.DataFrame([input_data], columns=X_train.columns)
    input_df = sm.add_constant(input_df, has_constant="add")
    prediction = float(model.predict(input_df)[0])
    return model, prediction