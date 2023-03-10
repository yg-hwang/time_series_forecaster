import pytz
import pandas as pd
from functools import reduce
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler

from .utils import add_date

KST = pytz.timezone("Asia/Seoul")
TODAY = datetime.now(KST).strftime("%Y-%m-%d")


def generate_future_dataframe(
    cutoff: str = TODAY, freq: str = "D", weekmask: str = None, steps: int = 1
) -> pd.DataFrame:
    """
    cutoff로부터 예측 기간(steps) 만큼의 미래 날짜 dataframe을 반환한다.

    :param cutoff: 예측 시작일
    :param freq: 날짜 주기 (D: 일별, W: 매주 일요일, MS: 매월 초, M: 매월 말, Y: 매년 말)
    :param weekmask: 특정 요일만 선택 (e.g. "Mon Wed Fri")
    :param steps: 예측 기간 (horizon)

    Example
    --------
    >>> generate_future_dataframe(cutoff="2022-04-01", freq="D", steps=3)
        date
    0   2022-04-01
    1   2022-04-02
    2   2022-04-03

    >>> generate_future_dataframe(cutoff="2022-04-01", freq="C", steps=3, weekmask="Mon")
        date
    0   2022-04-04 (Mon)
    1   2022-04-11 (Mon)
    2   2022-04-18 (Mon)

    >>> generate_future_dataframe(cutoff="2022-04-01", freq="C", steps=3, weekmask="Mon Wed")
        date
    0   2022-04-04 (Mon)
    1   2022-04-06 (Wed)
    2   2022-04-11 (Mon)
    """
    cutoff = str(cutoff)
    if steps >= 0:
        if freq == "C":
            date_range = pd.bdate_range(
                start=datetime.strptime(cutoff, "%Y-%m-%d"),
                periods=steps,
                freq="C",
                weekmask=weekmask,
            )
            return pd.DataFrame({"date": date_range})
        else:
            date_range = pd.date_range(
                start=datetime.strptime(cutoff, "%Y-%m-%d"), periods=steps, freq=freq
            )
            return pd.DataFrame({"date": date_range})
    else:
        raise ValueError("'steps' must be at least zero.")


def transform_feature_scale(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    dataframe의 컬럼을 지정하면 Min-Max scaling을 적용한 결과를 반환한다.
    """
    scaler = MinMaxScaler()
    return pd.DataFrame(data=scaler.fit_transform(df[features]), columns=features)


def split_train_test_data(df: pd.DataFrame, cutoff: str, steps: int, freq: str):
    """
    cutoff를 기준으로 Train, Test dataframe을 분할하여 결과를 반환한다.
    """
    date_end = add_date(date=cutoff, delta=steps - 1, freq=freq)
    df_train = df[df["date"] < cutoff].reset_index(drop=True)
    df_test = df[df["date"].between(cutoff, date_end)].reset_index(drop=True)
    return df_train, df_test


def decompose_seasonality_trend(df: pd.DataFrame, target: str = None) -> pd.DataFrame:
    """
    예측변수 y를 Trend, Seansonality, Resid로 분해한 결과를 반환한다.

    Example
    --------
    >>> decompose_seasonality_trend(df=df)
                trend       season     resid
    2023-01-13  1234.123    123.123    12.12
    ...
    """
    if target:
        df = df.rename(columns={target: "y"})
    stl = STL(df, robust=True).fit()
    dfs = [stl.trend, stl.seasonal, stl.resid]
    df_stl = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how="inner"
        ),
        dfs,
    )
    return df_stl


def create_cutoffs(cutoff_start: str, cutoff_end: str, freq: str = "D") -> list:
    """
    dataset을 train, test으로 분리할 때 사용할 cutoff 리스트를 생성한다.

    Example
    --------
    create_cutoffs(cutoff_start="2021-10-01", cutoff_end="2021-12-01", freq="D")
    >>> ["2021-10-01", "2021-10-02", "2021-10-03", ... , "2021-11-30", "2021-12-01"]

    create_cutoffs(cutoff_start="2021-10-01", cutoff_end="2021-12-01", freq="W")
    >>> ["2021-10-03", "2021-10-10", "2021-10-17", ... , "2021-11-21", "2021-11-28"]

    create_cutoffs(cutoff_start="2021-10-01", cutoff_end="2021-12-01", freq="M")
    >>> ["2021-10-01", "2021-11-01", "2021-12-01"]
    """

    d0 = datetime.strptime(cutoff_start, "%Y-%m-%d")
    d1 = datetime.strptime(cutoff_end, "%Y-%m-%d")
    interval = (d1 - d0).days

    if freq == "D":
        cutoffs = [d0 + timedelta(days=x) for x in range(interval + 1)]
        cutoffs = sorted([datetime.strftime(dt, "%Y-%m-%d") for dt in cutoffs])
        return cutoffs

    elif freq == "W":
        cutoffs = [d0 + timedelta(days=x) for x in range(interval + 1)]
        cutoffs_sunday = [cutoff for cutoff in cutoffs if cutoff.weekday() == 6]
        cutoffs_sunday = sorted(
            [datetime.strftime(dt, "%Y-%m-%d") for dt in cutoffs_sunday]
        )
        return cutoffs_sunday

    elif freq == "M":
        cutoffs = [cutoff_start]
        if cutoff_start == cutoff_end:
            return cutoffs
        else:
            interval = int(interval / 30)
            for i in range(1, interval + 1):
                if i < interval:
                    cutoffs.append(add_date(date=cutoff_start, delta=i, freq="M"))
                else:
                    cutoff_end = cutoff_end[:-2] + "01"
                    cutoffs.append(cutoff_end)
            return cutoffs

    else:
        raise ValueError("'freq' should be one of 'D', 'W' and 'M'.")
