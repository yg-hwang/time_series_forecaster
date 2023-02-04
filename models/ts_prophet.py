import numpy as np
import pandas as pd
from prophet import Prophet

from .base import BaseForecaster
from .utils import get_lunar_date


class PROPHET(BaseForecaster):
    def __init__(self, df: pd.DataFrame = None):
        super().__init__()
        self.df = df
        self.holidays = None
        self.model = None
        self.exogenous = None

    def _create_holidays(self) -> None:
        lunar_new_year_list = [
            get_lunar_date(year, 1, 1) for year in np.arange(1990, 2050)
        ]
        lunar_chuseok_list = [
            get_lunar_date(year, 8, 15) for year in np.arange(1990, 2050)
        ]

        lunar_new_year = pd.DataFrame(
            {
                "holiday": "lunar_new_year",
                "ds": pd.to_datetime(lunar_new_year_list),
                "lower_window": -2,
                "upper_window": 2,
            }
        )
        lunar_chuseok = pd.DataFrame(
            {
                "holiday": "chuseok",
                "ds": pd.to_datetime(lunar_chuseok_list),
                "lower_window": -2,
                "upper_window": 2,
            }
        )
        self.holidays = pd.concat((lunar_new_year, lunar_chuseok))

    def fit(self, target: str, **params) -> None:
        self._create_holidays()
        if params:
            if "add_seasonality" in params:
                add_seasonality = params["add_seasonality"]
                _params = params.copy()
                _params.pop("add_seasonality")
                self.model = Prophet(**_params, holidays=self.holidays)
            else:
                self.model = Prophet(**params, holidays=self.holidays)
                add_seasonality = None
        else:
            self.model = Prophet(holidays=self.holidays)
            add_seasonality = None

        if add_seasonality:
            if type(add_seasonality) == list:
                for add in add_seasonality:
                    self.model.add_seasonality(**add)
            elif type(add_seasonality) == dict:
                self.model.add_seasonality(**add_seasonality)
            else:
                raise TypeError(
                    "TypeError: 'add_seasonality' is allowed dictionary or list type."
                )

        self.df = self.df.reset_index().rename(columns={"date": "ds", target: "y"})
        self.exogenous = self.df.drop(columns=["ds", "y"]).columns
        if len(self.exogenous) > 0:
            for ex in self.exogenous:
                self.model.add_regressor(ex)
        self.model.fit(self.df)

    def predict(
        self, df_future: pd.DataFrame, steps: int, freq: str, **params
    ) -> pd.DataFrame:
        if len(self.exogenous) > 0:
            X = (
                df_future[self.exogenous]
                .iloc[:steps, :]
                .reset_index()
                .rename(columns={"date": "ds"})
            )
        else:
            X = df_future.reset_index().iloc[:steps, :].rename(columns={"date": "ds"})
        df_fcst = self.model.predict(X)
        return df_fcst.rename(columns={"ds": "date"})[
            ["date", "yhat", "yhat_lower", "yhat_upper"]
        ]
