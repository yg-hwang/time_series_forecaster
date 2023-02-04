import pandas as pd


class BaseForecaster:
    def __init__(self):
        super().__init__()

    def fit(self, target: str, exogenous: list = None, **fit_params):
        return

    def predict(self, df_future: pd.DataFrame, steps: int, freq: str, **params):
        return
