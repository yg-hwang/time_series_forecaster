import pandas as pd

from ts_arima import ARIMA
from ts_prophet import PROPHET


class TimeSeriesForecaster(ARIMA, PROPHET):
    def __init__(self):
        super().__init__()
        self.algo = None
        self.forecaster = None
        self.df_fcst = None

    def _create_forecaster(self):
        if self.algo == "arima":
            return ARIMA(df=self.df)
        elif self.algo == "prophet":
            return PROPHET(df=self.df)
        else:
            raise ValueError("Invalid `algo`. Options: ['arima', 'prophet']")

    def train(self, df: pd.DataFrame, target: str, algo: str, **params):
        self.algo = algo
        self.df = df
        self.forecaster = self._create_forecaster()
        self.forecaster.fit(target=target, **params)

    def forecast(self, df_future: pd.DataFrame, steps: int, **params) -> None:
        self.df_fcst = self.forecaster.predict(
            df_future=df_future, steps=steps, **params
        )
