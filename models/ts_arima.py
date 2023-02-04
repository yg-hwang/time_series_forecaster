import pandas as pd
from pmdarima.arima import auto_arima
from base import BaseForecaster
from utils import add_date
from ts_preprocess import generate_future_dataframe


class ARIMA(BaseForecaster):
    def __init__(self, df: pd.DataFrame = None):
        super().__init__()
        self.df = df
        self.model = None
        self.exogenous = None

    def fit(self, target: str, **params) -> None:
        y = self.df[target]
        self.exogenous = [column for column in self.df.columns if column != target]
        if self.exogenous:
            X = self.df[self.exogenous]
        else:
            X = None
        self.model = auto_arima(y=y, X=X, **params)
        print(self.model.summary())
        self.model.fit(y=y, X=X)

    def predict(
        self, df_future: pd.DataFrame, steps: int, freq: str, **params
    ) -> pd.DataFrame:
        if self.exogenous:
            X = df_future[self.exogenous].iloc[:steps, :]
            pred, pi = self.model.predict(
                n_periods=steps, X=X, return_conf_int=True, **params
            )
        else:
            pred, pi = self.model.predict(
                n_periods=steps, return_conf_int=True, **params
            )
        df_fcst = pd.DataFrame(
            pi, index=df_future.index, columns=["yhat_lower", "yhat_upper"]
        )
        future_date = generate_future_dataframe(
            cutoff=add_date(date=self.df.index.max(), delta=1, freq=freq),
            freq=freq,
            steps=steps,
        )
        df_fcst["yhat"] = pd.Series(pred, index=future_date.index)
        return df_fcst.reset_index()[["date", "yhat", "yhat_lower", "yhat_upper"]]
