"""Основные этапы анализа выручки из домашней работы"""

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots              import plot_acf
from statsmodels.tsa.seasonal                   import STL
from statsmodels.tsa.stattools                  import adfuller, kpss
from statsmodels.nonparametric.smoothers_lowess import lowess


class SalesAnalysis:
    """
    Очищает операции одного выбранного товара и 
    строит ряд выручки с выбранной частотой
    """

    def __init__(self, data: pd.DataFrame, stock_code: str, frequency: str = "D"):
        """Принимает таблицу Online Retail, код товара и частоту D, W-SUN или MS"""
        if frequency not in ("D", "W-SUN", "MS"):
            raise ValueError("Частота должна быть D, W-SUN или MS")

        self.frequency  = frequency
        self.stock_code = str(stock_code)

        product = data.loc[
            data["StockCode"].astype("string") == self.stock_code
        ].copy()

        self.sales = self.clean_sales(product)
        if self.sales.empty:
            raise ValueError("После очистки не осталось продаж выбранного товара")

        self.aggregated = self.sales.resample(frequency, on="InvoiceDate").agg({
            "Quantity":  "sum",
            "Revenue":   "sum",
            "InvoiceNo": "nunique",
        })

        # Как и в исследовании, отсутствие записей считаем отсутствием продаж
        if frequency != "D":
            # Полноту периода определяем по границам всего журнала продаж
            dates = pd.to_datetime(data["InvoiceDate"])
            first, last = dates.min().normalize(), dates.max().normalize()
            period_freq = "W-SUN" if frequency == "W-SUN" else "M"
            periods = self.aggregated.index.to_period(period_freq)
            complete = (periods.start_time >= first) & (periods.end_time.normalize() <= last)
            self.aggregated = self.aggregated.loc[complete].copy()

        if self.aggregated.empty:
            raise ValueError("Для выбранной частоты не осталось полных периодов")
        self.revenue = self.aggregated["Revenue"]


    @staticmethod
    def clean_sales(data: pd.DataFrame) -> pd.DataFrame:
        """Оставляет платные продажи без отмен и дублей, с порогами из шага 2"""
        cancelled = (
            data["InvoiceNo"].astype("string")
                .str.upper()
                .str.startswith("C", na=False)
        )
        sales = data.loc[
            data["Description"].notna()
            & data["Quantity"].between(0, 20_000, inclusive="right")
            & data["UnitPrice"].between(0, 10_000, inclusive="right")
            & ~cancelled
        ].drop_duplicates().copy()

        sales["InvoiceDate"] = pd.to_datetime(sales["InvoiceDate"])
        sales["InvoiceNo"]   = sales["InvoiceNo"].astype("string")
        sales["StockCode"]   = sales["StockCode"].astype("string")
        sales["Revenue"]     = sales["Quantity"] * sales["UnitPrice"]
        return sales


    def describe(self) -> pd.Series:
        """Возвращает описательные статистики и долю периодов без выручки"""
        revenue = self.revenue
        stats = revenue.describe()

        stats["variance"]   = revenue.var()
        stats["range"]      = revenue.max() - revenue.min()
        stats["IQR"]        = revenue.quantile(0.75) - revenue.quantile(0.25)
        stats["CV"]         = revenue.std() / revenue.mean() if revenue.mean() != 0 else np.nan
        stats["zero_share"] = revenue.eq(0).mean()

        return stats

    
    def decompose_stl(self, period: int = 7) -> pd.DataFrame:
        """Разлагает выручку на тренд, сезонность и остаток методом STL"""
        result = STL(self.revenue, period=period, robust=True).fit()
        return pd.DataFrame({
            "Revenue": self.revenue,
            "T":       result.trend,
            "S":       result.seasonal,
            "R":       result.resid,
        })


    def decompose_lowess(self, frac: float) -> pd.DataFrame:
        """Оценивает тренд и остаток без выделения сезонной компоненты"""
        trend = lowess(
            endog=self.revenue,
            exog=range(len(self.revenue)),
            frac=frac,
            return_sorted=False,
        )
        return pd.DataFrame({
            "Revenue": self.revenue,
            "T":       trend,
            "R":       self.revenue - trend,
        })


    def difference(self, lag: int = 1, log: bool = False) -> pd.Series:
        """Вычисляет разности заданного лага, при необходимости после log1p"""
        if lag < 1:
            raise ValueError("Лаг должен быть положительным")

        revenue = np.log1p(self.revenue) if log else self.revenue
        return revenue.diff(lag).dropna()


    def stationarity(self, series: pd.Series | None = None) -> pd.DataFrame:
        """Проверяет ряд тестами ADF и KPSS относительно постоянного уровня"""
        revenue = self.revenue if series is None else series
        adf_stat, adf_p, *_ = adfuller(
            revenue, regression="c", autolag="AIC", result_object=False,
        )
        kpss_stat, kpss_p, *_ = kpss(
            revenue, regression="c", nlags="auto", result_object=False,
        )

        return pd.DataFrame({
            "statistic": [adf_stat, kpss_stat],
            "p_value":   [adf_p, kpss_p],
            "reject_H0": [adf_p < 0.05, kpss_p < 0.05],
        }, index=pd.Index(["ADF", "KPSS"], name="Тест"))


    def plot_diagnostics(self, series: pd.Series | None = None, window: int = 28):
        """Строит ряд, скользящие статистики и ACF до или после преобразования"""
        revenue = self.revenue if series is None else series
        rolling = revenue.rolling(window)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        axes[0].plot(revenue, label="Ряд")
        axes[0].plot(rolling.mean(), color="red", label="Rolling mean")
        axes[0].set_ylabel("Значение")
        axes[0].set_xlabel("Дата")
        axes[0].legend()

        axes[1].plot(rolling.std(), color="red", label="Rolling std")
        axes[1].set_ylabel("Стандартное отклонение")
        axes[1].set_xlabel("Дата")
        axes[1].legend()

        plot_acf(revenue, lags=min(30, len(revenue) - 1), zero=False, ax=axes[2])
        axes[2].set(title="Автокорреляция", xlabel="Лаг", ylabel="ACF")

        for ax in axes:
            ax.grid(alpha=0.3)

        fig.suptitle(f"Выручка: {self.frequency}, окно {window}")
        fig.tight_layout()
        plt.show()
