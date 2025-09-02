# -*- coding: utf-8 -*-
"""
S&P 500 vs FOMC target rate (midpoint) + две таблицы в HTML с цветовой индикацией,
БЕЗ использования pandas Styler / jinja2.

1) Таблица решений (дата, действие, уровни до/после, Δ, SPX +1d/+5d) с раскраской:
   - Δ (bp): зелёный > 0, красный < 0
   - SPX реакции: зелёный > 0, красный < 0
2) Таблица закономерностей (по типу решения: средние реакции и доля положительных):
   - Avg +Nd, %: зелёный > 0, красный < 0
   - Pos +Nd, %: зелёный ≥ 50%, красный < 50%

Зависимости:
  pip install pandas pandas_datareader plotly numpy
"""

import os
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from textwrap import dedent

# -----------------------------
# Конфиг периода/вывода
# -----------------------------
START = "2015-10-01"   # Измени на "2015-10-01" для старта с октября 2015
END   = None           # None = до сегодня, либо "YYYY-MM-DD"
FORWARD_DAYS = (1, 5)  # горизонты реакции
OUTDIR = "out"
HTML_FILE = "fomc_spx_with_table.html"


# -----------------------------
# Загрузка данных
# -----------------------------
def load_spx(start=START, end=END):
    spx = pdr.DataReader("SP500", "fred", start, end)
    spx = spx.rename(columns={"SP500": "SPX"}).dropna()
    return spx

def load_ffr_midpoint(start=START, end=END):
    up = pdr.DataReader("DFEDTARU", "fred", start, end)  # upper bound
    lo = pdr.DataReader("DFEDTARL", "fred", start, end)  # lower bound
    df = up.join(lo, how="outer")
    df = df.rename(columns={"DFEDTARU": "upper", "DFEDTARL": "lower"})
    df["mid"] = (df["upper"] + df["lower"]) / 2.0
    df = df.ffill()  # протягиваем уровень по календарным дням
    return df[["upper", "lower", "mid"]]


# -----------------------------
# Выделение дат решений и реакции
# -----------------------------
def build_decisions(ffr_mid):
    """
    Находим даты, когда midpoint изменился (до/после/дельта/тип действия).
    """
    mid = ffr_mid["mid"].copy()
    before = mid.shift(1)
    change = mid.diff()
    idx = change[change != 0].index  # даты, где впервые виден новый уровень
    out = []
    for dt in idx:
        after = float(mid.loc[dt])
        bef = float(before.loc[dt])
        if np.isnan(bef):
            continue
        delta = after - bef
        action = "hike" if delta > 0 else "cut" if delta < 0 else "unch"
        out.append({"date": dt, "action": action, "before": bef, "after": after, "delta": delta})
    decisions = pd.DataFrame(out).sort_values("date").reset_index(drop=True)
    return decisions

def next_trading_day(prices, dt):
    """Первый торговый день в prices с индексом >= dt."""
    idx = prices.index.searchsorted(pd.Timestamp(dt))
    if idx >= len(prices.index):
        return None
    return prices.index[idx]

def compute_reactions(prices, decisions, horizon_days=FORWARD_DAYS):
    """
    Для каждой даты решения считаем доходность S&P 500 на +H дней (по ближайшему следующему торговому дню).
    """
    prices = prices.copy()
    out = []
    for _, row in decisions.iterrows():
        d = row["date"]
        t0 = next_trading_day(prices, d)
        if t0 is None:
            continue
        p0 = float(prices.loc[t0, "SPX"])
        rec = {"date": d, "action": row["action"], "before": row["before"], "after": row["after"], "delta": row["delta"]}
        for h in horizon_days:
            # ориентируемся по календарным дням, а не количеству торговых
            tH_pos = prices.index.searchsorted(t0 + pd.Timedelta(days=h))
            if tH_pos >= len(prices.index):
                rec[f"spx_plus_{h}d_pct"] = np.nan
            else:
                tH = prices.index[tH_pos]
                pH = float(prices.loc[tH, "SPX"])
                rec[f"spx_plus_{h}d_pct"] = (pH / p0 - 1.0) * 100.0
        out.append(rec)
    return pd.DataFrame(out).sort_values("date").reset_index(drop=True)


# -----------------------------
# Анализ закономерностей
# -----------------------------
def analyze_patterns(decisions_rxn, horizons=FORWARD_DAYS):
    """
    По каждому типу решения (hike/cut/unch):
      - количество случаев
      - средняя реакция S&P 500 на каждый горизонт
      - доля положительных исходов (% > 0) на каждый горизонт
    """
    actions = ["hike", "cut", "unch"]
    rows = []
    for action in actions:
        sub = decisions_rxn[decisions_rxn["action"] == action]
        row = {"Action": action, "Count": int(len(sub))}
        for h in horizons:
            col = f"spx_plus_{h}d_pct"
            row[f"Avg +{h}d, %"] = float(sub[col].mean()) if len(sub) else np.nan
            row[f"Pos +{h}d, %"] = float((sub[col] > 0).mean() * 100.0) if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------
# Визуализация Plotly
# -----------------------------
def make_figure(prices, ffr_mid, decisions):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        row_heights=[0.70, 0.30],
        subplot_titles=("S&P 500 (FRED)", "FFR Target Midpoint (%)")
    )

    # Верх: S&P 500
    fig.add_trace(
        go.Scatter(x=prices.index, y=prices["SPX"], name="S&P 500", mode="lines"),
        row=1, col=1
    )

    # Вертикальные линии дат решений
    color_map = {"hike": "green", "cut": "red", "unch": "blue"}
    for _, r in decisions.iterrows():
        fig.add_vline(
            x=r["date"], line=dict(dash="dot", width=2, color=color_map.get(r["action"], "gray")),
            row=1, col=1
        )

    # Низ: уровень midpoint
    fig.add_trace(
        go.Scatter(x=ffr_mid.index, y=ffr_mid["mid"], mode="lines+markers", name="FFR midpoint"),
        row=2, col=1
    )

    fig.update_layout(
        title="S&P 500 и решения ФРС (цветные линии) + уровень FFR midpoint",
        legend=dict(orientation="v", x=1.02, y=1.0),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Index", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    return fig


# -----------------------------
# Рендер HTML-таблиц с раскраской (без Styler)
# -----------------------------
def fmt_num(x, digits=2):
    return f"{x:.{digits}f}" if pd.notna(x) else ""

def td(text, cls=None, align_right=True):
    cls_attr = f' class="{cls}"' if cls else ""
    style = ' style="text-align:right;padding:8px 10px;border-bottom:1px solid #eee;"' if align_right \
            else ' style="text-align:left;padding:8px 10px;border-bottom:1px solid #eee;"'
    return f"<td{cls_attr}{style}>{text}</td>"

def th(text, align_right=True):
    style = ' style="text-align:right;padding:8px 10px;border-bottom:1px solid #eee;"' if align_right \
            else ' style="text-align:left;padding:8px 10px;border-bottom:1px solid #eee;"'
    return f"<th{style}>{text}</th>"

def table_decisions_html(t1: pd.DataFrame, horizons):
    # Заголовки
    headers = ["Decision date", "Action", "Midpoint before, %", "Midpoint after, %", "Δ (bp)"] + \
              [f"S&P 500 +{h}d, %" for h in horizons]
    thead = "<tr>" + th(headers[0], align_right=False) + th(headers[1], align_right=False) + \
            "".join(th(h) for h in headers[2:]) + "</tr>"

    # Строки
    rows_html = []
    for _, r in t1.iterrows():
        # классы для цветов
        delta_cls = "pos" if r["Δ (bp)"] > 0 else "neg" if r["Δ (bp)"] < 0 else None
        rxn_tds = []
        for h in horizons:
            val = r[f"S&P 500 +{h}d, %"]
            cls = "pos" if pd.notna(val) and val > 0 else "neg" if pd.notna(val) and val < 0 else None
            rxn_tds.append(td(fmt_num(val, 2), cls=cls, align_right=True))

        row = (
            "<tr>"
            + td(r["Decision date"], align_right=False)
            + td(r["Action"], align_right=False)
            + td(fmt_num(r["Midpoint before, %"], 2))
            + td(fmt_num(r["Midpoint after, %"], 2))
            + td(f"{r['Δ (bp)']:.0f}", cls=delta_cls)
            + "".join(rxn_tds)
            + "</tr>"
        )
        rows_html.append(row)

    table = f"""
    <table style="border-collapse:collapse;width:100%;margin:8px 0 24px;">
      <thead>{thead}</thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
    """
    return table

def table_patterns_html(t2: pd.DataFrame, horizons):
    headers = ["Action", "Count"] + sum([[f"Avg +{h}d, %", f"Pos +{h}d, %"] for h in horizons], [])
    thead = "<tr>" + th(headers[0], align_right=False) + th(headers[1]) + \
            "".join(th(h) for h in headers[2:]) + "</tr>"

    rows_html = []
    for _, r in t2.iterrows():
        tds = [
            td(r["Action"], align_right=False),
            td(f"{int(r['Count']):d}")
        ]
        for h in horizons:
            avg = r[f"Avg +{h}d, %"]
            pos = r[f"Pos +{h}d, %"]
            avg_cls = "pos" if pd.notna(avg) and avg > 0 else "neg" if pd.notna(avg) and avg < 0 else None
            pos_cls = "pos" if pd.notna(pos) and pos >= 50.0 else "neg" if pd.notna(pos) else None
            tds.append(td(fmt_num(avg, 2), cls=avg_cls))
            tds.append(td(fmt_num(pos, 2), cls=pos_cls))
        row = "<tr>" + "".join(tds) + "</tr>"
        rows_html.append(row)

    table = f"""
    <table style="border-collapse:collapse;width:100%;margin:8px 0 24px;">
      <thead>{thead}</thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
    """
    return table


# -----------------------------
# Экспорт HTML (график + 2 таблицы)
# -----------------------------
def export_html(fig, decisions_with_rxn, patterns_df, out_path, horizons=FORWARD_DAYS):
    # --------- Таблица 1: по каждому решению (в числовом виде) ---------
    t1 = decisions_with_rxn.copy()
    t1["Decision date"] = t1["date"].dt.strftime("%Y-%m-%d")
    t1["Action"] = t1["action"]
    t1["Midpoint before, %"] = t1["before"]
    t1["Midpoint after, %"]  = t1["after"]
    t1["Δ (bp)"] = (t1["delta"] * 100)  # число (для окраски)
    for h in horizons:
        t1[f"S&P 500 +{h}d, %"] = t1[f"spx_plus_{h}d_pct"]
    cols = ["Decision date", "Action", "Midpoint before, %", "Midpoint after, %", "Δ (bp)"] + \
           [f"S&P 500 +{h}d, %" for h in horizons]
    t1 = t1[cols]

    # --------- Таблица 2: сводка закономерностей ---------
    t2 = patterns_df.copy()
    t2 = t2[["Action", "Count"] + sum([[f"Avg +{h}d, %", f"Pos +{h}d, %"] for h in horizons], [])]

    # --------- Сборка HTML ---------
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    base_style = dedent("""
        <style>
        body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 0;}
        .container { max-width: 1200px; margin: 24px auto; padding: 0 16px; }
        h2 { margin: 24px 0 8px; }
        .note { color: #666; font-size: 13px; margin-bottom: 10px; }
        .pos { color: #179c52; }  /* зелёный */
        .neg { color: #d64545; }  /* красный */
        </style>
    """)

    table1_html = table_decisions_html(t1, horizons)
    table2_html = table_patterns_html(t2, horizons)

    html = f"""
    <!doctype html>
    <html lang="ru">
    <meta charset="utf-8">
    <title>S&P 500 vs FOMC (midpoint) + reactions + patterns</title>
    {base_style}
    <body>
      <div class="container">
        {fig_html}

        <h2>Таблица решений FOMC и реакции рынка</h2>
        <div class="note">
          Action: hike — повышение, cut — понижение, unch — без изменения. Δ (bp) — изменение таргета в базисных пунктах.
        </div>
        {table1_html}

        <h2>Закономерности по типам решений</h2>
        <div class="note">
          Avg +Nd, % — средняя доходность S&P 500 через N дней; Pos +Nd, % — доля положительных случаев (зелёный ≥ 50%).
        </div>
        {table2_html}
      </div>
    </body>
    </html>
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("1) Загружаем S&P 500 (SP500) с FRED...")
    prices = load_spx()
    print(f"   {len(prices)} строк")

    print("2) Загружаем верх/низ таргета ФРС и строим midpoint...")
    ffr_mid = load_ffr_midpoint()
    print(f"   {len(ffr_mid)} дат, последняя midpoint={ffr_mid['mid'].iloc[-1]:.2f}%")

    print("3) Находим даты изменений таргета (решения)...")
    decisions = build_decisions(ffr_mid)
    print(f"   {len(decisions)} решений в выборке")

    print(f"4) Считаем реакции рынка {FORWARD_DAYS} дней...")
    decisions_rxn = compute_reactions(prices, decisions, horizon_days=FORWARD_DAYS)

    print("5) Анализ закономерностей...")
    patterns = analyze_patterns(decisions_rxn, horizons=FORWARD_DAYS)
    print(patterns.to_string(index=False))

    print("6) Рисуем график и сохраняем HTML с двумя таблицами...")
    fig = make_figure(prices, ffr_mid, decisions)
    out_path = os.path.join(OUTDIR, HTML_FILE)
    export_html(fig, decisions_rxn, patterns, out_path, horizons=FORWARD_DAYS)
    print(f"Готово: {out_path}")


if __name__ == "__main__":
    main()
