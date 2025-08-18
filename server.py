# 부산항 환적 물동량 FastAPI 예측 서버
from fastapi import FastAPI, HTTPException,  UploadFile, File, Form
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import io
from pathlib import Path
import itertools
warnings.filterwarnings('ignore')

# FastAPI 앱 생성
app = FastAPI(
    title="부산항 환적 물동량 예측 API",
    description="실시간 경제지표 기반 부산항 환적 물동량 예측 시스템",
    version="2.0.0"
)

# CORS 설정 (React 프론트엔드와 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic 모델 정의
class EconomicIndicators(BaseModel):
    china_gdp: float = Field(default=5.2, ge=-2, le=8, description="중국 GDP 성장률 (%)")
    us_gdp: float = Field(default=2.5, ge=-1, le=5, description="미국 GDP 성장률 (%)")
    japan_gdp: float = Field(default=0.9, ge=-2, le=3, description="일본 GDP 성장률 (%)")
    korea_gdp: float = Field(default=3.1, ge=0, le=6, description="한국 GDP 성장률 (%)")
    oil_price: float = Field(default=82.0, ge=60, le=120, description="WTI 유가 ($/barrel)")
    exchange_rate: float = Field(default=1300, ge=1200, le=1500, description="원/달러 환율")
    container_rate: float = Field(default=2200, ge=1800, le=2800, description="컨테이너 운임지수")
    port_connectivity: float = Field(default=119.5, ge=100, le=140, description="부산항 연결성지수")
    global_trade: float = Field(default=4.2, ge=0, le=8, description="글로벌 무역 성장률 (%)")
    supply_chain: float = Field(default=85.0, ge=70, le=100, description="공급망 안정성 지수")

class MonthlyPrediction(BaseModel):
    date: str
    year: int
    month: int
    baseline: int
    predicted: int
    economic_impact: float

class PredictionResponse(BaseModel):
    total_impact: float
    multiplier: float
    yearly_totals: Dict[int, int]
    monthly_predictions: List[MonthlyPrediction]
    impacts_breakdown: Dict[str, Dict[str, float]]
    growth_rates: Dict[str, float]

class SensitivityAnalysis(BaseModel):
    indicator: str
    elasticity: float
    impact_1_percent: float
    current_impact: float
    importance_rank: int

# 부산항 예측 클래스
class BusanPortFastAPIPredictor:
    def __init__(self):
        # 실제 부산항 환적 물동량 데이터 (TEU/월)
        self.real_data = {
            '2020-01': 917547, '2020-02': 797984, '2020-03': 958877, '2020-04': 885187,
            '2020-05': 888330, '2020-06': 905925, '2020-07': 898638, '2020-08': 930599,
            '2020-09': 900790, '2020-10': 1016524, '2020-11': 1050490, '2020-12': 994587,
            
            '2021-01': 961966, '2021-02': 873801, '2021-03': 992440, '2021-04': 1020318,
            '2021-05': 1039881, '2021-06': 1017075, '2021-07': 1043942, '2021-08': 1004788,
            '2021-09': 975080, '2021-10': 1040169, '2021-11': 964985, '2021-12': 948103,
            
            '2022-01': 1035819, '2022-02': 896682, '2022-03': 914648, '2022-04': 954063,
            '2022-05': 982990, '2022-06': 977862, '2022-07': 1048967, '2022-08': 978307,
            '2022-09': 786209, '2022-10': 930273, '2022-11': 890293, '2022-12': 866490,
            
            '2023-01': 977113, '2023-02': 864822, '2023-03': 1045465, '2023-04': 1037675,
            '2023-05': 1000757, '2023-06': 949320, '2023-07': 979636, '2023-08': 981181,
            '2023-09': 995711, '2023-10': 978576, '2023-11': 1009360, '2023-12': 967880,
            
            '2024-01': 1028004, '2024-02': 1003140, '2024-03': 1048466, '2024-04': 1034960,
            '2024-05': 1112463, '2024-06': 1085343, '2024-07': 1106055, '2024-08': 1081672,
            '2024-09': 989506, '2024-10': 1081974, '2024-11': 1059765, '2024-12': 1106990
        }
        
        # 논문 기반 경제지표별 탄력성 계수
        self.elasticity_coefficients = {
            'china_gdp': 0.45,      # 중국 GDP 1% 증가 → 환적량 0.45% 증가
            'us_gdp': 0.30,         # 미국 GDP 1% 증가 → 환적량 0.30% 증가
            'japan_gdp': 0.15,      # 일본 GDP 1% 증가 → 환적량 0.15% 증가
            'korea_gdp': 0.20,      # 한국 GDP 1% 증가 → 환적량 0.20% 증가
            'oil_price': -0.12,     # 유가 1% 증가 → 환적량 0.12% 감소
            'exchange_rate': -0.08, # 원/달러 1% 증가 → 환적량 0.08% 감소
            'container_rate': -0.15, # 컨테이너 운임 1% 증가 → 환적량 0.15% 감소
            'port_connectivity': 0.25, # 항만연결성 1% 증가 → 환적량 0.25% 증가
            'global_trade': 0.40,   # 글로벌 무역량 1% 증가 → 환적량 0.40% 증가
            'supply_chain': 0.18    # 공급망 안정성 1% 증가 → 환적량 0.18% 증가
        }
        
        # 기준값 (2024년 평균 기준)
        self.baseline_indicators = {
            'china_gdp': 5.2, 'us_gdp': 2.5, 'japan_gdp': 0.9, 'korea_gdp': 3.1,
            'oil_price': 82.0, 'exchange_rate': 1300, 'container_rate': 2200,
            'port_connectivity': 119.5, 'global_trade': 4.2, 'supply_chain': 85.0
        }
        
        # 계절성 패턴
        self.seasonal_pattern = {
            1: 0.97, 2: 0.91, 3: 1.03, 4: 1.02, 5: 1.04, 6: 1.00,
            7: 1.03, 8: 1.01, 9: 0.96, 10: 1.02, 11: 0.99, 12: 0.99
        }
        
        # 베이스라인 계산
        self.baseline_monthly = sum([v for k, v in self.real_data.items() if k.startswith('2024')]) / 12
        self.annual_growth_rate = 0.035  # 3.5% 기본 성장률
    
        
    def calculate_economic_impact(self, indicators: EconomicIndicators):
        """경제지표 기반 영향도 계산"""
        total_impact = 0.0
        impacts = {}
        
        for indicator, current_value in indicators.dict().items():
            if indicator in self.elasticity_coefficients:
                baseline_value = self.baseline_indicators[indicator]
                elasticity = self.elasticity_coefficients[indicator]
                
                # 변화율 계산
                if indicator in ['china_gdp', 'us_gdp', 'japan_gdp', 'korea_gdp', 'global_trade']:
                    change_percent = current_value - baseline_value  # %p 변화
                else:
                    change_percent = (current_value - baseline_value) / baseline_value * 100  # % 변화
                
                # 탄력성 적용
                impact = (change_percent / 100) * elasticity
                total_impact += impact
                
                impacts[indicator] = {
                    'change_percent': round(change_percent, 2),
                    'impact': round(impact, 4),
                    'elasticity': elasticity,
                    'baseline': baseline_value,
                    'current': current_value
                }
        
        multiplier = 1 + total_impact
        
        return {
            'total_impact': round(total_impact, 4),
            'multiplier': round(multiplier, 4),
            'impacts': impacts
        }
    
    def predict_with_indicators(self, indicators: EconomicIndicators, forecast_months: int = 24):
        """
        SARIMA로 베이스라인 시계열을 학습/예측한 뒤,
        탄력성 승수(multiplier)를 곱해 시나리오 반영.
        (exog는 사용하지 않음. exog를 쓰려면 /forecast_exog_path 사용)
        """
        # 1) 경제적 영향(승수) 계산
        impact_result = self.calculate_economic_impact(indicators)
        multiplier = impact_result['multiplier']

        # 2) 과거 y 시계열 준비
        df_hist = pd.DataFrame(
            [{"date": pd.to_datetime(k), "ts_teu": v} for k, v in self.real_data.items()]
        ).sort_values("date")
        ts = (df_hist.set_index("date")["ts_teu"]
                    .asfreq("MS")
                    .interpolate("time"))

        # 3) SARIMA 차수 탐색 & 적합 (log-변환 안정화)
        y_tr = np.log(np.maximum(ts, 1.0))
        order, seas = auto_sarimax_order(y_tr, seasonal_periods=12)

        model = SARIMAX(y_tr, order=order, seasonal_order=seas,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)

        # 4) 예측
        pred = res.get_forecast(steps=forecast_months)
        mu = pred.predicted_mean
        mu = np.exp(mu)  # 로그 역변환
        fut_index = mu.index

        # 5) 응답 구성 (베이스라인 vs 시나리오)
        predictions = []
        for d, base_val in zip(fut_index, mu.values):
            year = int(d.year)
            month = int(d.month)
            date_str = d.strftime("%Y-%m")

            baseline_val = float(base_val)
            final_val = baseline_val * multiplier

            predictions.append(MonthlyPrediction(
                date=date_str,
                year=year,
                month=month,
                baseline=int(round(baseline_val)),
                predicted=int(round(final_val)),
                economic_impact=round(multiplier, 4)
            ))

        # 6) 연도별 합계/성장률
        yearly_totals = {}
        years_in_forecast = sorted(set(p.year for p in predictions))
        for y in years_in_forecast:
            yearly_totals[y] = sum(p.predicted for p in predictions if p.year == y)

        actual_2024 = sum(v for k, v in self.real_data.items() if k.startswith('2024'))
        growth_rates = {}
        if 2025 in yearly_totals:
            growth_rates['2025_vs_2024'] = round((yearly_totals[2025] - actual_2024) / actual_2024 * 100, 2)
        if 2026 in yearly_totals:
            growth_rates['2026_vs_2024'] = round((yearly_totals[2026] - actual_2024) / actual_2024 * 100, 2)
        if 2025 in yearly_totals and 2026 in yearly_totals:
            growth_rates['2026_vs_2025'] = round((yearly_totals[2026] - yearly_totals[2025]) / yearly_totals[2025] * 100, 2)

        return PredictionResponse(
            total_impact=impact_result['total_impact'],
            multiplier=impact_result['multiplier'],
            yearly_totals=yearly_totals,
            monthly_predictions=predictions,
            impacts_breakdown=impact_result['impacts'],
            growth_rates=growth_rates
        )


# --- SARIMAX용 자동 차수 탐색 (AIC 최소) ---
def auto_sarimax_order(y, seasonal_periods=12):
    p_list, d_list, q_list = [0,1], [1], [0,1]
    P_list, D_list, Q_list = [0,1], [1], [0,1]
    best=None; best_aic=float("inf")
    for (p,d,q,P,D,Q) in itertools.product(p_list,d_list,q_list,P_list,D_list,Q_list):
        try:
            m=SARIMAX(y, order=(p,d,q), seasonal_order=(P,D,Q,seasonal_periods),
                      enforce_stationarity=False, enforce_invertibility=False)
            r=m.fit(disp=False)
            if r.aic < best_aic:
                best_aic=r.aic; best=((p,d,q),(P,D,Q,seasonal_periods))
        except Exception:
            pass
    return best or ((1,1,1),(0,1,1,seasonal_periods))

# --- 거시 CSV -> 월별 exog 프레임 (과거/미래) ---
def prepare_macro_df(file_bytes: bytes, filename: str, sheet_name: str | int | None = None) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        # Excel
        import pandas as pd
        # sheet_name이 숫자 문자열이면 int로 변환
        if isinstance(sheet_name, str) and sheet_name.isdigit():
            sheet_name = int(sheet_name)
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name or 0, engine="openpyxl")
    elif ext == ".csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext} (csv/xlsx/xls 만 허용)")

    # ===== 아래는 기존 전처리 로직 유지 =====
    date_col = None
    for c in ["date","월","Month","month","기간"]:
        if c in df.columns:
            date_col = c; break
    if date_col is None:
        raise ValueError("표에 'date' 또는 '월/Month' 컬럼이 필요합니다.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].copy()
    df["date"] = df[date_col].dt.to_period("M").dt.to_timestamp()  # 월초

    keep = ["date"] + [c for c in df.columns if c != date_col]
    df = df[keep]

    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return (df.groupby("date", as_index=False)
              .mean(numeric_only=True)
              .sort_values("date"))


def build_exog_from_csv(macro_df: pd.DataFrame,
                        hist_index: pd.DatetimeIndex,
                        fut_index: pd.DatetimeIndex,
                        features: list,
                        use_yoy: bool,
                        include_lag1: bool):
    X = macro_df.set_index("date").sort_index()
    features = [c for c in features if c in X.columns]
    if not features:
        return None, None
    X = X[features].copy()
    if use_yoy:
        X = X.pct_change(12) * 100.0  # YoY %
    ex_hist = X.reindex(hist_index).ffill()
    ex_fut  = X.reindex(fut_index ).ffill()
    if include_lag1:
        for c in list(ex_hist.columns):
            ex_hist[f"{c}_l1"] = ex_hist[c].shift(1)
            ex_fut[f"{c}_l1"]  = ex_fut[c].shift(1)
        ex_hist = ex_hist.ffill()
        ex_fut  = ex_fut.ffill()
    ex_hist = ex_hist.fillna(0.0)
    ex_fut  = ex_fut.fillna(0.0)
    return ex_hist, ex_fut

# --- 프론트에서 준 미래 월별 경로(JSON)로 exog 미래 부분 오버라이드 ---
def exog_future_from_json(fut_index: pd.DatetimeIndex,
                          features: list,
                          future_exog_json: str,
                          include_lag1: bool) -> pd.DataFrame:
    data = json.loads(future_exog_json) if future_exog_json else {}
    rows=[]
    for d in fut_index:
        ym = d.strftime("%Y-%m")
        row = {k: float(data.get(ym, {}).get(k, "nan")) for k in features}
        row["date"]=d
        rows.append(row)
    df = (pd.DataFrame(rows)
            .set_index("date")
            .astype(float)
            .sort_index()
            .ffill().bfill()
            .fillna(0.0))
    if include_lag1:
        for c in list(df.columns):
            df[f"{c}_l1"] = df[c].shift(1)
        df = df.ffill().fillna(0.0)
    return df

# 전역 예측기 인스턴스
predictor = BusanPortFastAPIPredictor()

# API 엔드포인트들
@app.get("/")
async def root():
    """API 루트 - 서버 상태 확인"""
    return {
        "message": "부산항 환적 물동량 예측 API",
        "version": "2.0.0",
        "status": "운영 중",
        "endpoints": {
            "predict": "/predict",
            "sensitivity": "/sensitivity-analysis",
            "scenarios": "/scenarios",
            "historical": "/historical-data",
            "docs": "/docs"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_transshipment(indicators: EconomicIndicators):
    """
    실시간 경제지표 기반 환적 물동량 예측
    
    - **china_gdp**: 중국 GDP 성장률 (%)
    - **us_gdp**: 미국 GDP 성장률 (%)
    - **oil_price**: WTI 유가 ($/barrel)
    - **exchange_rate**: 원/달러 환율
    - 기타 경제지표들...
    
    Returns: 2025-2026년 월별 예측 결과
    """
    try:
        result = predictor.predict_with_indicators(indicators, 24)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실행 오류: {str(e)}")

@app.get("/sensitivity-analysis")
async def sensitivity_analysis():
    """민감도 분석 - 각 경제지표 1% 변화시 영향도"""
    try:
        baseline = EconomicIndicators()
        sensitivity_results = []
        
        for indicator, elasticity in predictor.elasticity_coefficients.items():
            # 1% 변화시 영향 계산
            impact_1_percent = abs(elasticity)
            
            sensitivity_results.append(SensitivityAnalysis(
                indicator=indicator,
                elasticity=elasticity,
                impact_1_percent = abs(elasticity),   
                #impact_1_percent=round(impact_1_percent * 100, 2),  # 백분율로 변환
                current_impact=0.0,  # 기준값이므로 0
                importance_rank=0
            ))
        
        # 중요도순 정렬
        sensitivity_results.sort(key=lambda x: x.impact_1_percent, reverse=True)
        for i, result in enumerate(sensitivity_results):
            result.importance_rank = i + 1
        
        return {
            "message": "경제지표별 민감도 분석 결과",
            "methodology": "각 지표 1% 변화시 환적량에 미치는 영향",
            "sensitivity_analysis": sensitivity_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"민감도 분석 오류: {str(e)}")

@app.get("/scenarios")
async def predefined_scenarios():
    """사전 정의된 시나리오별 예측"""
    try:
        scenarios = {
            "baseline": EconomicIndicators(),
            "optimistic": EconomicIndicators(
                china_gdp=6.0, us_gdp=3.0, japan_gdp=1.5, korea_gdp=4.0,
                oil_price=75, exchange_rate=1250, container_rate=2000,
                port_connectivity=125, global_trade=5.5, supply_chain=92
            ),
            "pessimistic": EconomicIndicators(
                china_gdp=3.8, us_gdp=1.5, japan_gdp=0.2, korea_gdp=2.0,
                oil_price=95, exchange_rate=1400, container_rate=2500,
                port_connectivity=115, global_trade=2.5, supply_chain=78
            ),
            "recovery": EconomicIndicators(
                china_gdp=5.5, us_gdp=2.8, japan_gdp=1.2, korea_gdp=3.5,
                oil_price=80, exchange_rate=1280, container_rate=2100,
                port_connectivity=122, global_trade=4.8, supply_chain=88
            )
        }
        
        scenario_results = {}
        for scenario_name, scenario_indicators in scenarios.items():
            prediction = predictor.predict_with_indicators(scenario_indicators, 24)
            scenario_results[scenario_name] = {
                "name": scenario_name,
                "indicators": scenario_indicators.dict(),
                "yearly_totals": prediction.yearly_totals,
                "total_impact": prediction.total_impact,
                "growth_2026": prediction.growth_rates["2026_vs_2024"]
            }
        
        return {
            "message": "사전 정의된 시나리오별 예측 결과",
            "scenarios": scenario_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시나리오 분석 오류: {str(e)}")

@app.get("/historical-data")
async def get_historical_data():
    """과거 실적 데이터 반환"""
    try:
        historical_data = []
        for date, value in predictor.real_data.items():
            year, month = date.split('-')
            historical_data.append({
                "date": date,
                "year": int(year),
                "month": int(month),
                "transshipment": value
            })
        
        # 연도별 통계
        yearly_stats = {}
        for year in range(2020, 2025):
            yearly_total = sum([item["transshipment"] for item in historical_data if item["year"] == year])
            yearly_stats[year] = yearly_total
        
        return {
            "message": "부산항 환적 물동량 과거 실적",
            "data_period": "2020.01 ~ 2024.12",
            "total_months": len(historical_data),
            "monthly_data": historical_data,
            "yearly_statistics": yearly_stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 조회 오류: {str(e)}")

@app.get("/baseline-indicators")
async def get_baseline_indicators():
    """기준 경제지표 값들 반환"""
    return {
        "message": "2024년 기준 경제지표 값들",
        "baseline_indicators": predictor.baseline_indicators,
        "elasticity_coefficients": predictor.elasticity_coefficients,
        "methodology": "논문 기반 탄력성 계수 적용"
    }
@app.post("/forecast_exog_path")
async def forecast_exog_path(
    horizon: int = Form(12),
    log: bool = Form(True),
    macro_files: List[UploadFile] | None = File(None),
    macro_csv: UploadFile | None = File(None),   # ← 이름은 그대로 두고, 엑셀도 허용
    features_json: str = Form("[]"),
    use_yoy: bool = Form(True),
    include_lag1: bool = Form(True),
    future_exog: str = Form(""),
    sheet_name: str | None = Form(None)          # ← 추가: 엑셀 시트 선택(없으면 1번째 시트)
):
    try:
        # 1) 내장된 과거 환적 실적 시계열
        df_hist = pd.DataFrame([
            {"date": pd.to_datetime(k), "ts_teu": v}
            for k, v in predictor.real_data.items()
        ]).sort_values("date")
        ts = (df_hist.set_index("date")["ts_teu"]
                      .asfreq("MS")
                      .interpolate("time"))
        last = ts.index.max()
        fut_index = pd.date_range(last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

        # 2) exog 구성
                # 2) exog 구성
        features = json.loads(features_json) if features_json else []
        ex_hist = ex_fut = None
        have_exog = False

        if macro_csv is not None:
            try:
                # 파일은 한 번만 읽습니다
                file_bytes = await macro_csv.read()
                macro_df = prepare_macro_df(
                    file_bytes,
                    macro_csv.filename or "uploaded",
                    sheet_name=sheet_name
                )

                ex_hist_csv, ex_fut_csv = build_exog_from_csv(
                    macro_df, ts.index, fut_index,
                    features=features, use_yoy=use_yoy, include_lag1=include_lag1
                )

                if ex_hist_csv is not None and ex_fut_csv is not None and ex_hist_csv.shape[1] > 0:
                    have_exog = True
                    ex_hist, ex_fut = ex_hist_csv, ex_fut_csv

                    # 프론트에서 월별 미래 경로가 오면 오버라이드
                    if future_exog:
                        ex_fut_override = exog_future_from_json(
                            fut_index, features, future_exog, include_lag1=include_lag1
                        )
                        # 공통 열만 안전하게 덮어쓰기
                        for col in ex_fut_override.columns:
                            if col in ex_fut.columns:
                                ex_fut[col] = ex_fut_override[col]
            except Exception as e:
                have_exog = False  # 파싱 문제 시 exog 없이 폴백
                # 필요하면 로그
                # print(f"macro parse error: {e}")


        # 3) SARIMA(X) 적합/예측
        y_tr = np.log(np.maximum(ts, 1.0)) if log else ts
        order, seas = auto_sarimax_order(y_tr, seasonal_periods=12)

        if have_exog:
            model = SARIMAX(y_tr, exog=ex_hist, order=order, seasonal_order=seas,
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            pred = res.get_forecast(steps=horizon, exog=ex_fut)
            mu = pred.predicted_mean
            if log: mu = np.exp(mu)
            scenario = [{"month": d.strftime("%Y-%m"), "value": float(mu.loc[d])} for d in fut_index]
            return {
                "mode": "sarimax_exog",
                "order": order, "seasonal_order": seas,
                "scenario": scenario
            }
        else:
            # exog가 없으면 SARIMA 폴백
            model = SARIMAX(y_tr, order=order, seasonal_order=seas,
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            pred = res.get_forecast(steps=horizon)
            mu = pred.predicted_mean
            if log: mu = np.exp(mu)
            baseline = [{"month": d.strftime("%Y-%m"), "value": float(mu.loc[d])} for d in fut_index]
            return {
                "mode": "sarima_fallback",
                "order": order, "seasonal_order": seas,
                "baseline": baseline,
                "message": "macro_csv가 없어 exog 미사용(SARIMA 폴백)."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/forecast_exog_path 오류: {e}")
# 서버 실행
if __name__ == "__main__":
    import uvicorn
    print("부산항 환적 물동량 예측 API 서버 시작")
    print("FastAPI 자동 문서: http://localhost:8000/docs")
    print("실시간 예측: http://localhost:8000/predict")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# 실행 명령어:
# pip install fastapi uvicorn pandas numpy pydantic
# uvicorn main:app --reload --host 0.0.0.0 --port 8000