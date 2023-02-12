import itertools
import pandas as pd
from typing import Union
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


class ProphetGridSearch:
    def __init__(self):
        self.default_params = dict()
        self.hyperparams = dict()
        self.model_list = None

    def add_default_params(self, default_params: dict):
        """
        Prophet()에 고정으로 사용할 파라미터와 값을 dictionary로 생성한다.

        Example
        --------
        add_default_params({"yearly_seasonality": True, "weekly_seasonality": True, ...})
        """
        self.default_params.update(default_params)

    def add_hyperparams(self, params: dict):
        """
        Prophet()에 파라미터 조합을 만들기 위한 값을 dictionary로 생성한다.

        Example
        --------
        add_hyperparams({"yearly_seasonality": [True, False], "weekly_seasonality": [True, False], ...})
        """
        self.hyperparams.update(params)

    def create_hyperparams_space(self):
        """
        add_hyperparams()와 add_default_params()를 실행 후, 파라미터 조합을 list로 생성한다.

        Output Example
        --------------
        [{"growth": "linear", "yearly_seasonality": True}, ...]
        """
        prophet_hyperparams = [
            dict(zip(self.hyperparams.keys(), v))
            for v in itertools.product(*self.hyperparams.values())
        ]

        for idx, params in enumerate(prophet_hyperparams):
            params.update(self.default_params)
            prophet_hyperparams[idx] = params

        return prophet_hyperparams

    def build_models(
        self,
        df: pd.DataFrame,
        country_name: str = "KR",
        add_seasonality: Union[list, tuple] = None,
    ):
        """
        create_combinations_of_params()에서 생성된 파라미터 조합으로 Prophet()을 수행하고,
        파라미터 조합과 모델 객체를 list로 생성한다.

        Output Example
        --------------
        [{"growth": "linear",
         "yearly_seasonality": True,
         "add_seasonality": [{"name": "monthly", "period": 30.5, "fourier_order": 5}],
         "model": <prophet.forecaster.Prophet at 0x7f9fc9190090>}}, {...}, ...]
        """

        prophet_hyperparams = self.create_hyperparams_space()
        self.model_list = list()

        for param in prophet_hyperparams:
            m = Prophet(**param)
            m.add_country_holidays(country_name=country_name)

            if add_seasonality is None:
                m.fit(df)
                param["add_seasonality"] = add_seasonality
                param["model"] = m
                self.model_list.append(param)

            else:
                for add_seasonality_param in add_seasonality:
                    m.add_seasonality(**add_seasonality_param)
                m.fit(df)
                param_a = (
                    param.copy()
                )  # 아래에서 add_seasonality가 없는 버전으로도 fit을 하기 위해 param을 사용하지 않고 따로 생성
                param_a["add_seasonality"] = add_seasonality
                param_a["model"] = m
                self.model_list.append(param_a)

                # add_seasonality가 없는 경우도 추가
                m = Prophet(**param)
                m.add_country_holidays(country_name=country_name)
                m.fit(df)
                param["add_seasonality"] = None
                param["model"] = m
                self.model_list.append(param)

    @staticmethod
    def calculate_error(
        model: Prophet,
        initial: int,
        horizon: int,
        period: int,
        cutoffs: list = None,
        parallel: str = None,
        rolling_window: Union[int, float] = 1,
    ) -> dict:
        """
        Prophet()을 수행하여 만들어진 모델을 기준으로 cross validation을 수행하고,
        CV를 통해 fitting된 결과로부터 오차가 계산된 pd.DataFrame 반환한다.
        """
        df_cv = cross_validation(
            model=model,
            initial=f"{initial} days",
            horizon=f"{horizon} days",
            period=f"{period} days",
            cutoffs=cutoffs,
            parallel=parallel,
        )
        return performance_metrics(df=df_cv, rolling_window=rolling_window)

    def run_cv(
        self,
        initial: int,
        horizon: int,
        period: int,
        cutoffs: list = None,
        parallel: str = None,
        rolling_window: Union[int, float] = 1,
    ):
        """
        build_models()을 통해 생성된 모델을 기준으로 Cross Validation을 수행한다.
        각 모델별 파라미터와 Metirc을 pd.DataFrame으로 반환한다.
        """

        df_perf_metrics_all = pd.DataFrame()

        for idx, param_dict in enumerate(self.model_list):
            try:
                df_perf_metrics = self.calculate_error(
                    model=param_dict["model"],
                    initial=initial,
                    horizon=horizon,
                    period=period,
                    cutoffs=cutoffs,
                    parallel=parallel,
                    rolling_window=rolling_window,
                )
            except Exception as err:
                print(err)
                continue

            df_perf_metrics["model_idx"] = idx
            param_dict.pop("model")
            param_dict.update(self.default_params)
            df_perf_metrics["params"] = str(param_dict)
            df_perf_metrics_all = df_perf_metrics_all.append(df_perf_metrics)
        print(df_perf_metrics_all.head())
        return df_perf_metrics_all.sort_values(
            "mape", ascending=True, ignore_index=True
        )
