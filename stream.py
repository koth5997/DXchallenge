# stream.py
import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="부산항 대시보드", layout="wide")

DATA_FILES = {
    "container": "busan_container_2020_2024.csv",
    "vessel_all": "busan_vessel_2020_2024.csv",
    "shiptype_total": "busan_vessel_shiptype_monthly_total_2020_2024.csv",
    "shiptype_byclass": "busan_vessel_shiptype_byclass_2020_2024.csv",
    "tonnage_byclass": "busan_vessel_tonnage_byclass_2020_2024.csv",
}

# ---------- Helpers ----------
@st.cache_data
def load_csv(path: str, parse_date_col="date"):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if parse_date_col in df.columns:
        df[parse_date_col] = pd.to_datetime(df[parse_date_col], errors="coerce")
        df = df.dropna(subset=[parse_date_col]).sort_values(parse_date_col)
    return df

def year_range_slider(df, date_col="date", key="yr_default"):
    if df is None or df.empty or date_col not in df.columns:
        return df
    years = sorted(df[date_col].dropna().dt.year.unique().tolist())
    if not years:
        return df
    lo, hi = int(min(years)), int(max(years))
    y1, y2 = st.sidebar.slider(
        "연도 범위", min_value=lo, max_value=hi, value=(lo, hi), key=key
    )
    return df[df[date_col].dt.year.between(y1, y2)].copy()

def moving_avg(df, value_col, window=3):
    df = df.sort_values("date").copy()
    df[f"{value_col}_MA{window}"] = df[value_col].rolling(window, min_periods=1).mean()
    return df

def pick_metric(df, candidates, default=None, label="지표 선택", key=None):
    avail = [c for c in candidates if c in df.columns]
    if not avail:
        return None
    default = default if (default in avail) else avail[0]
    return st.selectbox(label, avail, index=avail.index(default), key=key)

# ---------- Load data ----------
container = load_csv(DATA_FILES["container"])
vessel_all = load_csv(DATA_FILES["vessel_all"])
shiptype_total = load_csv(DATA_FILES["shiptype_total"])
shiptype_byclass = load_csv(DATA_FILES["shiptype_byclass"])
tonnage_byclass = load_csv(DATA_FILES["tonnage_byclass"])

st.title("🚢 부산항 월별 실적 대시보드 (2020–2024)")
st.caption("컨테이너/선박(선종·톤급) CSV를 이용해 인터랙티브 탐색")

# 사이드바 공통 옵션
st.sidebar.header("공통 옵션")
show_ma = st.sidebar.checkbox("3개월 이동평균 표시", value=True, key="global_ma")

# ---------- Tabs ----------
tabs = st.tabs([
    "📦 컨테이너",
    "🚢 선박(전체)",
    "🛳️ 선종별 (by class)",
    "⚖️ 톤급별 (by class)",
    "📈 선 전체 월합계"
])

# ========== Tab 1: 컨테이너 ==========
with tabs[0]:
    st.subheader("📦 컨테이너 월별 실적")
    if container is None or container.empty:
        st.warning("컨테이너 CSV가 없습니다.")
    else:
        df = year_range_slider(container, key="yr_container")
        metric = pick_metric(
            df,
            candidates=["합계","총계","수입","수출","수입환적","수출환적","TEU","Full","Empty"],
            default=("합계" if "합계" in df.columns else None),
            label="컨테이너 지표",
            key="metric_container"
        )
        if metric is None:
            st.info("표시할 수 있는 컨테이너 지표가 없습니다.")
        else:
            if show_ma:
                df = moving_avg(df, metric, window=3)
                y_cols = [metric, f"{metric}_MA3"]
            else:
                y_cols = [metric]

            col1, col2 = st.columns([2,1])
            with col1:
                fig = px.line(df, x="date", y=y_cols, markers=True, title=f"월별 {metric}")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                yearly = df.groupby(df["date"].dt.year)[metric].sum().reset_index(name=metric)
                fig2 = px.bar(yearly, x="date", y=metric, title=f"연도별 {metric} 합계")
                st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(df.reset_index(drop=True))

# ========== Tab 2: 선박(전체) ==========
with tabs[1]:
    st.subheader("🚢 선박 입출항 (월별 전체)")
    if vessel_all is None or vessel_all.empty:
        st.warning("선박 전체 CSV가 없습니다.")
    else:
        df = year_range_slider(vessel_all, key="yr_vessel_all")
        metric = pick_metric(
            df, candidates=["척수","GT","입항","출항","내항","외항","합계","총계"],
            default=("척수" if "척수" in df.columns else None),
            label="선박 지표",
            key="metric_vessel_all"
        )
        if metric is None:
            st.info("표시할 수 있는 선박 지표가 없습니다.")
        else:
            if show_ma:
                df = moving_avg(df, metric, window=3)
                y_cols = [metric, f"{metric}_MA3"]
            else:
                y_cols = [metric]
            fig = px.line(df, x="date", y=y_cols, markers=True, title=f"월별 {metric}")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.reset_index(drop=True))

# ========== Tab 3: 선종별(by class) ==========
with tabs[2]:
    st.subheader("🛳️ 선별 월별 실적")
    if shiptype_byclass is None or shiptype_byclass.empty:
        st.warning("선별(by class) CSV가 없습니다.")
    else:
        df = year_range_slider(shiptype_byclass, key="yr_shiptype_byclass")
        ship_list = sorted(df["선"].dropna().unique().tolist())
        left, right = st.columns([1,3])
        with left:
            pick = st.multiselect("선 선택 (복수 선택 가능)", ship_list, default=ship_list[:5], key="pick_shiptype")
            metric = pick_metric(df, candidates=["척수","GT"], default="척수", label="지표", key="metric_shiptype")
        view = df[df["선"].isin(pick)] if pick else df
        if metric and not view.empty:
            if show_ma:
                view = view.sort_values(["선","date"]).copy()
                view[f"{metric}_MA3"] = view.groupby("선")[metric].transform(lambda s: s.rolling(3, min_periods=1).mean())
                y_cols = [metric, f"{metric}_MA3"]
            else:
                y_cols = [metric]
            with right:
                fig = px.line(view, x="date", y=y_cols, color="선", markers=True,
                              title=f"선별 월별 {metric}")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("**선 Top-N (총합 기준)**")
        k = st.slider("Top N", 3, 20, 10, key="shiptype_topn_k")
        topn = (df.groupby("선")[metric].sum().sort_values(ascending=False).head(k).reset_index())
        col1, col2 = st.columns([2,1])
        with col1:
            figb = px.bar(topn, x="선", y=metric, title=f"선 Top-{k} 총합 {metric}")
            st.plotly_chart(figb, use_container_width=True)
        with col2:
            st.dataframe(topn)
        st.markdown("**원본 테이블**")
        st.dataframe(df.reset_index(drop=True))

# ========== Tab 4: 톤급별(by class) ==========
with tabs[3]:
    st.subheader("⚖️ 톤급별 월별 실적")
    if tonnage_byclass is None or tonnage_byclass.empty:
        st.warning("톤급별(by class) CSV가 없습니다.")
    else:
        df = year_range_slider(tonnage_byclass, key="yr_tonnage_byclass")
        tlist = sorted(df["톤급"].dropna().unique().tolist())
        left, right = st.columns([1,3])
        with left:
            pick = st.multiselect("톤급 선택 (복수 선택 가능)", tlist, default=tlist[:8], key="pick_tonnage")
            metric = pick_metric(df, candidates=["척수","GT"], default="척수", label="지표", key="metric_tonnage")
            stack = st.checkbox("스택 영역 그래프", value=True, key="tonnage_stack")
        view = df[df["톤급"].isin(pick)] if pick else df
        if metric and not view.empty:
            pivot = view.pivot_table(index="date", columns="톤급", values=metric, aggfunc="sum").fillna(0).sort_index()
            if stack:
                fig = px.area(pivot, x=pivot.index, y=pivot.columns, title=f"톤급별 월별 {metric}")
            else:
                fig = px.line(pivot, x=pivot.index, y=pivot.columns, title=f"톤급별 월별 {metric}", markers=True)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**톤급 Top-N (총합 기준)**")
        k = st.slider("Top N", 3, 20, 10, key="tonnage_topn_k")
        topn = (df.groupby("톤급")[metric].sum().sort_values(ascending=False).head(k).reset_index())
        c1, c2 = st.columns([2,1])
        with c1:
            figb = px.bar(topn, x="톤급", y=metric, title=f"톤급 Top-{k} 총합 {metric}")
            st.plotly_chart(figb, use_container_width=True)
        with c2:
            st.dataframe(topn)
        st.markdown("**원본 테이블**")
        st.dataframe(df.reset_index(drop=True))

# ========== Tab 5: 선종 전체 월합계 ==========
with tabs[4]:
    st.subheader("📈 선 전체 월합계 (척수/GT)")
    if shiptype_total is None or shiptype_total.empty:
        st.warning("선전체 합계 CSV가 없습니다.")
    else:
        df = year_range_slider(shiptype_total, key="yr_shiptype_total")
        metric = pick_metric(df, candidates=["척수","GT","합계","총계"], default=("척수" if "척수" in df.columns else None), key="metric_shiptype_total")
        if metric is None:
            st.info("표시할 수 있는 지표가 없습니다.")
        else:
            if show_ma:
                df = moving_avg(df, metric, window=3)
                y_cols = [metric, f"{metric}_MA3"]
            else:
                y_cols = [metric]
            fig = px.line(df, x="date", y=y_cols, markers=True, title=f"선 전체 월합계 - {metric}")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.reset_index(drop=True))

st.caption("ⓒ DX 챌린지 · 부산항 데이터 전처리/시각화 데모")
