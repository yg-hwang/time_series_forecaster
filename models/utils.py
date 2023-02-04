import math
import pandas as pd
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from korean_lunar_calendar import KoreanLunarCalendar


def add_date(date: str, delta: int, freq: str = "D") -> str:
    """
    date를 기준으로 일, 주, 월 단위로 날짜를 더하여 반환한다.

    Example
    --------
    >>> add_date(date="2022-05-01", delta=3, freq="D")
    '2022-05-04'

    >>> add_date(date="2022-05-01", delta=3, freq="W")
    '2022-05-22'

    >>> add_date(date="2022-05-01", delta=3, freq="M")
    '2022-08-01'
    """
    if isinstance(date, str):
        base_date = datetime.strptime(date, "%Y-%m-%d")
    else:
        try:
            base_date = datetime.strptime(str(date)[:10], "%Y-%m-%d")
        except Exception as err:
            return err

    if freq == "D":
        return (base_date + timedelta(days=delta)).strftime("%Y-%m-%d")
    elif freq == "W":
        return (base_date + timedelta(weeks=delta)).strftime("%Y-%m-%d")
    elif (freq == "M") | (freq == "MS"):
        return (base_date + relativedelta(months=delta)).strftime("%Y-%m-%d")
    elif (freq == "Y") | (freq == "YS"):
        return (base_date + relativedelta(years=delta)).strftime("%Y-%m-%d")
    else:
        raise ValueError("Invalid 'freq'. Use one of 'D', 'W', 'M' and 'Y'")


def remove_day_of_week(
    df: pd.DataFrame, date_column: str = "date", day_of_week: int = 6
) -> pd.DataFrame:
    """
    dataframe의 날짜에서 특정 요일을 제거한다.
    """
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df[df[date_column].dt.dayofweek != day_of_week].reset_index(drop=True)
        return df
    else:
        raise ValueError(f"date_column '{date_column}' not found.")


def get_lunar_date(year: int, month: int, day: int) -> str:
    """
    음력을 양력으로 변환한다.

    Example
    --------
    >>> get_lunar_date(year=2021, month=1, day=5)
    '2021-02-16'
    """
    calendar = KoreanLunarCalendar()
    calendar.setLunarDate(year, month, day, isIntercalation=False)
    return calendar.SolarIsoFormat()


def get_date_from_isocalendar(year: int, week: int, dow: int = 1) -> str:
    """
    연도, 주차, 요일로부터 날짜를 반환한다.

    Example
    --------
    >>> get_date_from_isocalendar(year=2022, week=18, dow=5) # 2022년 18주차의 금요일(5) 불러오기
    '2022-05-06'
    """
    year = int(year)
    week = int(week)
    dow = int(dow)
    dt = date.fromisocalendar(year, week, dow)
    return str(dt)


def generate_cyclic_coordinates(
    cyclic_num_of_points: int, plot: bool = False
) -> pd.DataFrame:
    """
    1차원으로 표현하기 어려운 순환 정보를 반영하는 2차원 변수를 생성한다.
    e.g., 1월과 12월은 거리가 가깝지만 6월과는 거리가 먼 정보를 담을 수 있는 변수 생성
    :param cyclic_num_of_points: integer, 등간격을 가진 순환 좌표의 개수
    :return: pandas.DataFrame, 등간격을 가진 순환 좌표축 (coordinate1, coordinate2)
    """
    unit_dist = 2 * math.pi / cyclic_num_of_points
    label = range(cyclic_num_of_points)
    coordinate1 = [math.sin(unit_dist * itr) for itr in label]
    coordinate2 = [math.cos(unit_dist * itr) for itr in label]

    if plot:
        # visualize coordinates
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(coordinate1, coordinate2)
        for itr in label:
            ax.annotate(itr + 1, xy=(coordinate1[itr], coordinate2[itr]))

    return pd.DataFrame(
        {
            "label": range(1, cyclic_num_of_points + 1),
            "coordinate1": coordinate1,
            "coordinate2": coordinate2,
        }
    )


def remove_outlier(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    IQR 1.5 기준으로 이상치를 Saturating한 결과를 반환한다.
    """
    df1 = df.copy()
    df = df._get_numeric_data()
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    for col in columns:
        for i in range(0, len(df[col])):
            if df[col][i] < lower_bound[col]:
                df[col][i] = lower_bound[col]

            if df[col][i] > upper_bound[col]:
                df[col][i] = upper_bound[col]
    for col in columns:
        df1[col] = df[col]
    return df1


def create_floor_and_cap(df: pd.DataFrame):
    """
    Prophet(growth="logistic")를 수행할 때 필요한 floor와 cap 컬럼을 생성한다. Prophet(growth="linear")일 때는 무관하다.
    """
    Q1 = df["y"].quantile(0.25)
    Q3 = df["y"].quantile(0.75)
    IQR = Q3 - Q1

    if Q1 - IQR * 1.5 > 0:
        floor = Q1 - IQR * 1.5
    else:
        floor = 0
    cap = Q3 + IQR * 1.5
    return floor, cap
