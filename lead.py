# lead.py
# Логарифмический S&P 500 + набор ведущих макро-индикаторов (FRED)
# Период: с 2016-01-01 по сегодня. Интерактивный график откроется в браузере
# и сохранится в lead_macro_spx.html.

import os
from datetime import datetime
import time
import math
import pandas as pd
import numpy as np

import yfinance as yf
from pandas_datareader import data as pdr

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# --- Настройки периода ---
START = "2016-01-01"
END = datetime.today().strftime("%Y-%m-%d")

# --- Рендерер Plotly: в браузер (избавляет от nbformat) ---
try:
    pio.renderers.default = "browser"  # можно поставить "vscode" если запускаешь в VSCode
except Exception:
    pass

# --- Набор индикаторов: код FRED + трансформация + частота ---
# transform: 'level' | 'yoy' | 'diff' | 'ma4_invert' | 'invert'
# freq: 'D' (daily), 'W' (weekly), 'M' (monthly)
INDICATORS = {
    "Building Permits (YoY)":       {"code": "PERMIT",        "freq": "M", "transform": "yoy"},
    "Housing Starts (YoY)":         {"code": "HOUST",         "freq": "M", "transform": "yoy"},
    "ISM New Orders (level)":       {"code": "NAPMNO",        "freq": "M", "transform": "level"},
    "Initial Claims 4wMA (inv)":    {"code": "ICSA",          "freq": "W", "transform": "ma4_invert"},
    "Yield Curve 10Y-3M (level)":   {"code": "T10Y3M",        "freq": "D", "transform": "level"},
    "Adj. NFCI (inv)":              {"code": "ANFCI",         "freq": "W", "transform": "invert"},
    "UMich Sentiment (YoY)":        {"code": "UMCSENT",       "freq": "M", "transform": "yoy"},
    "Industrial Production (YoY)":  {"code": "INDPRO",        "freq": "M", "transform": "yoy"},
    "Avg Weekly Hours (level)":     {"code": "AWHMAN",        "freq": "M", "transform": "level"},
    "Durable Goods Orders (YoY)":   {"code": "DGORDER",       "freq": "M", "transform": "yoy"},
    "HY OAS (inv)":                 {"code": "BAMLH0A0HYM2",  "freq": "D", "transform": "invert"},
    "ADS Business Conditions":      {"code": "ADS",           "freq": "D", "transform": "level"},
}

# --- Утилиты трансформаций ---
def to_yoy(series: pd.Series, freq: str) -> pd.Series:
    """Год-к-году с учётом частоты ряда."""
    s = series.copy()
    if freq == "M":
        return s.pct_change(12) * 100.0
    elif freq == "W":
        return s.pct_change(52) * 100.0
    else:
        # для дневной частоты используем 252 торговых дня
        return s.pct_change(252) * 100.0

def ma4_invert(series: pd.Series) -> pd.Series:
    """4-недельная скользящая средняя + инверсия (хуже = выше исходный ряд)."""
    return -series.rolling(4, min_periods=1).mean()

def standardize_z(series: pd.Series) -> pd.Series:
    """Z-score по наблюдаемому периоду (устойчиво к NaN)."""
    s = series.astype(float)
    valid = s.dropna()
    if valid.empty:
        return s * 0.0
    std = valid.std(ddof=0)
    if std == 0 or math.isnan(std):
        return (s - valid.mean()) * 0.0
    return (s - valid.mean()) / std

# --- Загрузка S&P 500 с защитой от пустых данных ---
def download_spx(start: str, end: str, retries: int = 3, pause_sec: float = 1.5) -> pd.DataFrame:
    last_err = None
    for _ in range(retries):
        try:
            df = yf.download("^GSPC", start=start, end=end, auto_adjust=True)
            if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
                out = df.rename(columns={"Close": "SPX"})[["SPX"]].copy()
                out.index = pd.to_datetime(out.index)
                # приводим к бизнес-дням и заполняем
                out = out.asfreq("B").ffill()
                # логарифм — для графика используем логшкалу, так что лог-колонка не обязательна,
                # но пусть будет для последующих вычислений при желании.
                out["SPX_LOG"] = np.log(out["SPX"])
                return out
        except Exception as e:
            last_err = e
        time.sleep(pause_sec)
    raise RuntimeError(f"Не удалось загрузить S&P 500 (^GSPC). Последняя ошибка: {last_err}")

# --- Загрузка FRED ---
def fetch_fred(code: str) -> pd.Series:
    s = pdr.DataReader(code, "fred", START, END).iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return s.astype(float)

def prepare_macro_series(name: str, meta: dict) -> pd.Series:
    code, freq, transform = meta["code"], meta["freq"], meta["transform"]
    s = fetch_fred(code)

    # 1) Нормализуем частоту для корректной YoY
    if freq == "M":
        s = s.resample("M").last()
    elif freq == "W":
        # FRED weekly часто по четвергам, приведём к "W-FRI" для стабильности
        s = s.resample("W-FRI").last()
    else:
        s = s.asfreq("D")

    # 2) Применяем трансформацию
    if transform == "yoy":
        s = to_yoy(s, freq)
    elif transform == "diff":
        s = s.diff()
    elif transform == "ma4_invert":
        s = ma4_invert(s)
    elif transform == "invert":
        s = -s
    # 'level' — без изменений

    # 3) Растягиваем до бизнес-дней и вперёд-заполняем
    s = s.resample("B").ffill()

    # 4) Z-оценка, чтобы все линии были сопоставимы по масштабу
    s = standardize_z(s)

    s.name = name
    return s

# --- Главная логика ---
def main():
    # S&P 500
    spx = download_spx(START, END)

    # Макро
    frames = {}
    for name, meta in INDICATORS.items():
        try:
            frames[name] = prepare_macro_series(name, meta)
        except Exception as e:
            print(f"[WARN] Не удалось загрузить {name} ({meta['code']}): {e}")

    if not frames:
        raise RuntimeError("Не удалось загрузить ни одного макро-индикатора с FRED.")

    macro = pd.DataFrame(frames)
    # Синхронизируем индексы
    macro = macro.loc[macro.index.intersection(spx.index)]
    macro = macro.loc[START:END].ffill()

    # --- График ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # S&P 500 (логарифмическая шкала по левой оси)
    fig.add_trace(
        go.Scatter(
            x=spx.index, y=spx["SPX"],
            name="S&P 500 (лог шкала)",
            mode="lines",
            line=dict(width=2),
            hovertemplate="Дата: %{x|%Y-%m-%d}<br>SPX: %{y:.2f}<extra></extra>"
        ),
        secondary_y=False
    )

    # Макро (Z-score) по правой оси
    default_show = {
        "ISM New Orders (level)",
        "Yield Curve 10Y-3M (level)",
        "Initial Claims 4wMA (inv)",
        "Adj. NFCI (inv)",
        "Building Permits (YoY)",
    }

    for col in macro.columns:
        visible = True if col in default_show else "legendonly"
        fig.add_trace(
            go.Scatter(
                x=macro.index, y=macro[col],
                name=col,
                mode="lines",
                visible=visible,
                hovertemplate="Дата: %{x|%Y-%m-%d}<br>Z: %{y:.2f}<extra></extra>"
            ),
            secondary_y=True
        )

    fig.update_layout(
        title=f"Логарифмический S&P 500 и ведущие макро-индикаторы (стандартизованные), {START} — {END}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=60, r=60, t=70, b=50),
    )

    fig.update_yaxes(title_text="S&P 500 (лог шкала)", type="log", secondary_y=False)
    fig.update_yaxes(title_text="Макро (Z-оценка)", secondary_y=True)
    fig.update_xaxes(title_text="Дата")

    # Показ и сохранение
    fig.show()
    out_file = "lead_macro_spx.html"
    fig.write_html(out_file, auto_open=False)
    print(f"[OK] График сохранён: {out_file}")

if __name__ == "__main__":
    main()
