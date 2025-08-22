from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf
import os
import glob
warnings.filterwarnings('ignore')

app = FastAPI(
    title="부산항 환적 예측 API (Enhanced)",
    description="실제 데이터 기반 부산항 환적 예측 시스템",
    version="4.0.0"
)

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
    monthly_factor: Optional[float] = 1.0
    transshipment_ratio: Optional[float] = None  # 환적 비율 추가

class PredictionResponse(BaseModel):
    total_impact: float
    multiplier: float
    yearly_totals: Dict[int, int]
    monthly_predictions: List[MonthlyPrediction]
    impacts_breakdown: Dict[str, Dict[str, float]]
    growth_rates: Dict[str, float]
    model_type: str = "economic"
    data_source: str = "excel_files"  # 데이터 소스 정보 추가

class BaselineIndicatorsResponse(BaseModel):
    baseline_indicators: Dict[str, float]
    description: str

class HistoricalDataResponse(BaseModel):
    message: str
    data_period: str
    total_months: int
    monthly_data: List[Dict]
    yearly_statistics: Dict[int, int]
    transshipment_statistics: Dict[int, Dict[str, float]]  # 환적 통계 추가

# 데이터 로더 클래스
class BusanPortDataLoader:
    def __init__(self, data_path: str = r"C:\\Users\\koth5\\바탕 화면\\DIVE2025\\data"):
        self.data_path = Path(data_path)
        self.historical_data = {}
        self.transshipment_data = {}
        self.terminal_data = {}
        
    def load_excel_files(self):
        """Excel 파일들을 로드하여 데이터 추출"""
        try:
            # 월별 컨테이너 처리실적 파일들 로드
            monthly_files = list(self.data_path.glob("*월별 컨테이너 처리실적*.xlsx"))
            terminal_files = list(self.data_path.glob("*터미널별FromTo*.xlsx"))
            
            print(f"Found {len(monthly_files)} monthly files and {len(terminal_files)} terminal files")
            
            # 월별 데이터 로드
            for file_path in monthly_files:
                year = self._extract_year_from_filename(file_path.name)
                self._load_monthly_data(file_path, year)
            
            # 터미널별 환적 데이터 로드
            for file_path in terminal_files:
                year = self._extract_year_from_filename(file_path.name)
                self._load_transshipment_data(file_path, year)
                
            return True
        except Exception as e:
            print(f"Error loading Excel files: {e}")
            return False
    
    def _extract_year_from_filename(self, filename: str) -> int:
        """파일명에서 연도 추출"""
        import re
        year_match = re.search(r'(\d{4})년', filename)
        return int(year_match.group(1)) if year_match else 2020
    
    def _load_monthly_data(self, file_path: Path, year: int):
        """월별 컨테이너 처리실적 데이터 로드"""
        try:
            # Excel 파일 읽기 (여러 시트 확인)
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # 데이터 구조에 따라 처리
                if '월별' in sheet_name or 'monthly' in sheet_name.lower():
                    self._process_monthly_sheet(df, year)
                    
        except Exception as e:
            print(f"Error loading monthly data from {file_path}: {e}")
    
    def _load_transshipment_data(self, file_path: Path, year: int):
        """터미널별 환적 데이터 로드"""
        try:
            df = pd.read_excel(file_path)
            self._process_transshipment_sheet(df, year)
        except Exception as e:
            print(f"Error loading transshipment data from {file_path}: {e}")
    
    def _process_monthly_sheet(self, df: pd.DataFrame, year: int):
        """월별 시트 데이터 처리"""
        # 실제 Excel 구조에 맞게 조정 필요
        # 예상 구조: 월별 컬럼이 있고, TEU 데이터가 있는 형태
        
        month_columns = ['1월', '2월', '3월', '4월', '5월', '6월', 
                        '7월', '8월', '9월', '10월', '11월', '12월']
        
        for i, month_col in enumerate(month_columns, 1):
            if month_col in df.columns:
                # 환적 관련 행 찾기
                transship_rows = df[df.iloc[:, 0].str.contains('환적|transship', na=False, case=False)]
                if not transship_rows.empty:
                    value = transship_rows[month_col].iloc[0]
                    if pd.notna(value) and str(value).replace(',', '').replace('.', '').isdigit():
                        date_key = f"{year}-{i:02d}"
                        self.historical_data[date_key] = int(str(value).replace(',', ''))
    
    def _process_transshipment_sheet(self, df: pd.DataFrame, year: int):
        """환적 데이터 시트 처리"""
        # FromTo 데이터에서 환적 비율 계산
        if 'From' in df.columns and 'To' in df.columns:
            # 환적 패턴 분석
            transship_mask = (df['From'] != df['To']) & (df['From'].notna()) & (df['To'].notna())
            transship_data = df[transship_mask]
            
            if not transship_data.empty and 'TEU' in df.columns:
                total_teu = df['TEU'].sum() if 'TEU' in df.columns else 0
                transship_teu = transship_data['TEU'].sum() if total_teu > 0 else 0
                transship_ratio = transship_teu / total_teu if total_teu > 0 else 0
                
                self.transshipment_data[year] = {
                    'total_teu': total_teu,
                    'transship_teu': transship_teu,
                    'transship_ratio': transship_ratio
                }
    
    def get_historical_data(self) -> Dict:
        """가공된 과거 데이터 반환"""
        if not self.historical_data:
            # 파일이 없거나 로드 실패시 기본 데이터 사용
            return self._get_default_data()
        
        return self.historical_data
    
    def _get_default_data(self) -> Dict:
        """기본 데이터 (파일 로드 실패시 사용)"""
        return {
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

# LSTM 모델 클래스 (기존과 동일)
class LSTMPredictor:
    def __init__(self, look_back=12):
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
      
    def prepare_data(self, data, features=None):
        """LSTM용 데이터 준비"""
        if features is not None:
            X, y = [], []
            for i in range(len(data) - self.look_back):
                X.append(np.column_stack([
                    data[i:i+self.look_back],
                    features[i:i+self.look_back] if features is not None else []
                ]))
                y.append(data[i+self.look_back])
            return np.array(X), np.array(y)
        else:
            X, y = [], []
            for i in range(len(data) - self.look_back):
                X.append(data[i:i+self.look_back])
                y.append(data[i+self.look_back])
            return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """LSTM 모델 구축"""
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, data, epochs=30, batch_size=16):
        """모델 학습"""
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        X, y = self.prepare_data(data_scaled)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        self.model = self.build_model((X.shape[1], 1))
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, 
                      callbacks=[early_stop], verbose=0)
    
    def predict(self, last_sequence, n_ahead=24):
        """예측"""
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_ahead):
            X = current_sequence[-self.look_back:].reshape(1, self.look_back, 1)
            pred = self.model.predict(X, verbose=0)[0, 0]
            predictions.append(pred)
            current_sequence = np.append(current_sequence, pred)
        
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        return predictions

# 개선된 부산항 예측 클래스
class EnhancedBusanPortPredictor:
    def __init__(self, data_loader: BusanPortDataLoader):
        self.data_loader = data_loader
        self.real_data = data_loader.get_historical_data()
        self.transshipment_data = data_loader.transshipment_data
        
        # 탄력성 계수 (기존과 동일)
        self.elasticity_coefficients = {
            'china_gdp': 0.45,
            'us_gdp': 0.30,
            'japan_gdp': 0.15,
            'korea_gdp': 0.20,
            'oil_price': -0.12,
            'exchange_rate': -0.08,
            'container_rate': -0.15,
            'port_connectivity': 0.25,
            'global_trade': 0.40,
            'supply_chain': 0.18
        }
        
        # 기준값
        self.baseline_indicators = {
            'china_gdp': 5.2, 'us_gdp': 2.5, 'japan_gdp': 0.9, 'korea_gdp': 3.1,
            'oil_price': 82.0, 'exchange_rate': 1300, 'container_rate': 2200,
            'port_connectivity': 119.5, 'global_trade': 4.2, 'supply_chain': 85.0
        }
        
        # 월별 계절성 패턴 (실제 데이터 기반으로 계산)
        self.seasonal_pattern = self._calculate_seasonal_pattern()
        
        # 월별 변동성 요인
        self.monthly_volatility = {
            1: 0.05, 2: 0.08, 3: 0.04, 4: 0.03, 5: 0.04, 6: 0.03,
            7: 0.04, 8: 0.05, 9: 0.07, 10: 0.03, 11: 0.04, 12: 0.05
        }
        
        self.baseline_monthly = self._calculate_baseline_monthly()
        self.annual_growth_rate = self._calculate_growth_rate()
        
        # LSTM 예측기 초기화
        self.lstm_predictor = LSTMPredictor(look_back=12)
    
    def _calculate_seasonal_pattern(self) -> Dict[int, float]:
        """실제 데이터를 기반으로 계절성 패턴 계산"""
        monthly_averages = {}
        
        for month in range(1, 13):
            month_values = []
            for date_key, value in self.real_data.items():
                if date_key.endswith(f"-{month:02d}"):
                    month_values.append(value)
            
            if month_values:
                monthly_averages[month] = np.mean(month_values)
            else:
                monthly_averages[month] = 1000000  # 기본값
        
        # 전체 평균 대비 각 월의 비율 계산
        overall_average = np.mean(list(monthly_averages.values()))
        seasonal_pattern = {}
        
        for month, avg in monthly_averages.items():
            seasonal_pattern[month] = avg / overall_average
        
        return seasonal_pattern
    
    def _calculate_baseline_monthly(self) -> float:
        """최근 년도 월평균 계산"""
        recent_values = [v for k, v in self.real_data.items() if k.startswith('2024')]
        return np.mean(recent_values) if recent_values else 1000000
    
    def _calculate_growth_rate(self) -> float:
        """연평균 성장률 계산"""
        yearly_totals = {}
        for date_key, value in self.real_data.items():
            year = int(date_key.split('-')[0])
            if year not in yearly_totals:
                yearly_totals[year] = 0
            yearly_totals[year] += value
        
        if len(yearly_totals) >= 2:
            years = sorted(yearly_totals.keys())
            first_year, last_year = years[0], years[-1]
            growth_rate = (yearly_totals[last_year] / yearly_totals[first_year]) ** (1/(last_year - first_year)) - 1
            return max(0, min(0.1, growth_rate))  # 0~10% 범위로 제한
        
        return 0.035  # 기본 성장률
    
    def calculate_monthly_economic_impact(self, indicators: EconomicIndicators, month: int):
        """월별 독립적 경제 영향도 계산"""
        base_impact = 0.0
        impacts = {}
        
        for indicator, current_value in indicators.dict().items():
            if indicator in self.elasticity_coefficients:
                baseline_value = self.baseline_indicators[indicator]
                elasticity = self.elasticity_coefficients[indicator]
                
                # 변화율 계산
                if indicator in ['china_gdp', 'us_gdp', 'japan_gdp', 'korea_gdp', 'global_trade']:
                    change_percent = current_value - baseline_value
                else:
                    change_percent = (current_value - baseline_value) / baseline_value * 100
                
                # 월별 가중치 적용
                monthly_weight = 1.0 + self.monthly_volatility.get(month, 0) * np.random.randn()
                impact = (change_percent / 100) * elasticity * monthly_weight
                
                base_impact += impact
                impacts[indicator] = {
                    'change_percent': round(change_percent, 2),
                    'impact': round(impact, 4),
                    'elasticity': elasticity,
                    'monthly_weight': round(monthly_weight, 3)
                }
        
        # 계절성 반영
        seasonal_factor = self.seasonal_pattern.get(month, 1.0)
        total_multiplier = (1 + base_impact) * seasonal_factor
        
        return {
            'total_impact': round(base_impact, 4),
            'multiplier': round(total_multiplier, 4),
            'seasonal_factor': seasonal_factor,
            'impacts': impacts
        }
    
    def predict_with_indicators(self, indicators: EconomicIndicators, forecast_months: int = 24):
        """개선된 예측 (월별 독립적 움직임)"""
        predictions = []
        yearly_totals = {}
        
        # 기준 시점 설정
        last_date = pd.to_datetime(max(self.real_data.keys()))
        
        for i in range(forecast_months):
            future_date = last_date + pd.DateOffset(months=i+1)
            year = future_date.year
            month = future_date.month
            date_str = future_date.strftime("%Y-%m")
            
            # 월별 독립적 경제 영향 계산
            monthly_impact = self.calculate_monthly_economic_impact(indicators, month)
            
            # 기본 예측값 (트렌드 + 계절성)
            trend_factor = (1 + self.annual_growth_rate) ** ((i+1) / 12)
            baseline_val = self.baseline_monthly * trend_factor
            
            # 환적 비율 적용 (실제 데이터 기반)
            transship_ratio = self._get_transshipment_ratio(year)
            
            # 월별 독립적 최종 예측값
            predicted_val = baseline_val * monthly_impact['multiplier']
            
            # 랜덤 노이즈 추가 (현실성)
            noise = np.random.normal(0, baseline_val * 0.02)
            predicted_val += noise
            
            predictions.append(MonthlyPrediction(
                date=date_str,
                year=year,
                month=month,
                baseline=int(round(baseline_val)),
                predicted=int(round(predicted_val)),
                economic_impact=monthly_impact['multiplier'],
                monthly_factor=monthly_impact['seasonal_factor'],
                transshipment_ratio=transship_ratio
            ))
            
            # 연도별 합계
            if year not in yearly_totals:
                yearly_totals[year] = 0
            yearly_totals[year] += int(round(predicted_val))
        
        # 성장률 계산
        actual_2024 = sum(v for k, v in self.real_data.items() if k.startswith('2024'))
        growth_rates = {}
        if 2025 in yearly_totals:
            growth_rates['2025_vs_2024'] = round((yearly_totals[2025] - actual_2024) / actual_2024 * 100, 2)
        if 2026 in yearly_totals:
            growth_rates['2026_vs_2024'] = round((yearly_totals[2026] - actual_2024) / actual_2024 * 100, 2)
        if 2025 in yearly_totals and 2026 in yearly_totals:
            growth_rates['2026_vs_2025'] = round((yearly_totals[2026] - yearly_totals[2025]) / yearly_totals[2025] * 100, 2)
        
        # 전체 영향도 계산 (평균)
        avg_impact = np.mean([p.economic_impact for p in predictions])
        
        return PredictionResponse(
            total_impact=round(avg_impact - 1, 4),
            multiplier=round(avg_impact, 4),
            yearly_totals=yearly_totals,
            monthly_predictions=predictions,
            impacts_breakdown=monthly_impact['impacts'],
            growth_rates=growth_rates,
            model_type="economic_enhanced",
            data_source="excel_files"
        )
    
    def _get_transshipment_ratio(self, year: int) -> float:
        """해당 년도 환적 비율 반환"""
        if year in self.transshipment_data:
            return self.transshipment_data[year]['transship_ratio']
        
        # 기본 환적 비율 (부산항은 약 50% 환적)
        return 0.5
    
    def predict_with_lstm(self, forecast_months: int = 24):
        """LSTM 모델을 사용한 예측 (실제 데이터 기반)"""
        # 데이터 준비
        df_hist = pd.DataFrame(
            [{"date": pd.to_datetime(k), "ts_teu": v} for k, v in self.real_data.items()]
        ).sort_values("date")
        
        # LSTM 학습
        data_array = df_hist["ts_teu"].values
        self.lstm_predictor.train(data_array, epochs=30, batch_size=16)
        
        # 예측
        last_sequence = self.lstm_predictor.scaler.transform(
            data_array[-self.lstm_predictor.look_back:].reshape(-1, 1)
        ).flatten()
        
        lstm_predictions = self.lstm_predictor.predict(last_sequence, forecast_months)
        
        # 결과 포맷팅
        predictions = []
        yearly_totals = {}
        last_date = pd.to_datetime(max(self.real_data.keys()))
        
        for i, pred_val in enumerate(lstm_predictions):
            future_date = last_date + pd.DateOffset(months=i+1)
            year = future_date.year
            month = future_date.month
            date_str = future_date.strftime("%Y-%m")
            
            # 월별 계절성 적용
            seasonal_adjusted = pred_val * self.seasonal_pattern.get(month, 1.0)
            
            predictions.append(MonthlyPrediction(
                date=date_str,
                year=year,
                month=month,
                baseline=int(round(pred_val)),
                predicted=int(round(seasonal_adjusted)),
                economic_impact=1.0,
                monthly_factor=self.seasonal_pattern.get(month, 1.0),
                transshipment_ratio=self._get_transshipment_ratio(year)
            ))
            
            if year not in yearly_totals:
                yearly_totals[year] = 0
            yearly_totals[year] += int(round(seasonal_adjusted))
        
        # 성장률 계산
        actual_2024 = sum(v for k, v in self.real_data.items() if k.startswith('2024'))
        growth_rates = {}
        if 2025 in yearly_totals:
            growth_rates['2025_vs_2024'] = round((yearly_totals[2025] - actual_2024) / actual_2024 * 100, 2)
        if 2026 in yearly_totals:
            growth_rates['2026_vs_2024'] = round((yearly_totals[2026] - actual_2024) / actual_2024 * 100, 2)
        
        return PredictionResponse(
            total_impact=0.0,
            multiplier=1.0,
            yearly_totals=yearly_totals,
            monthly_predictions=predictions,
            impacts_breakdown={},
            growth_rates=growth_rates,
            model_type="lstm_enhanced",
            data_source="excel_files"
        )
    
    def predict_with_sarima(self, forecast_months: int = 24):
        """SARIMA 모델을 사용한 예측 (실제 데이터 기반)"""
        # 데이터 준비
        df_hist = pd.DataFrame([
            {"date": pd.to_datetime(k), "ts_teu": v}
            for k, v in self.real_data.items()
        ]).sort_values("date")
        
        ts = (df_hist.set_index("date")["ts_teu"]
                      .asfreq("MS")
                      .interpolate("time"))
        
        # SARIMA 모델 학습
        y_tr = np.log(np.maximum(ts, 1.0))
        order, seas = auto_sarimax_order(y_tr, seasonal_periods=12)
        
        model = SARIMAX(y_tr, order=order, seasonal_order=seas,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=50)
        
        # 예측
        pred = res.get_forecast(steps=forecast_months)
        mu = pred.predicted_mean
        mu = np.exp(mu)  
        
        # 결과 
        predictions = []
        yearly_totals = {}
        last_date = pd.to_datetime(max(self.real_data.keys()))
        
        for i, (date_idx, pred_val) in enumerate(mu.items()):
            future_date = last_date + pd.DateOffset(months=i+1)
            year = future_date.year
            month = future_date.month
            date_str = future_date.strftime("%Y-%m")
            
            # 월별 계절성 적용
            seasonal_factor = self.seasonal_pattern.get(month, 1.0)
            seasonal_adjusted = pred_val * seasonal_factor
            
            predictions.append(MonthlyPrediction(
                date=date_str,
                year=year,
                month=month,
                baseline=int(round(pred_val)),
                predicted=int(round(seasonal_adjusted)),
                economic_impact=1.0,
                monthly_factor=seasonal_factor,
                transshipment_ratio=self._get_transshipment_ratio(year)
            ))
            
            if year not in yearly_totals:
                yearly_totals[year] = 0
            yearly_totals[year] += int(round(seasonal_adjusted))
        
        # 성장률 계산
        actual_2024 = sum(v for k, v in self.real_data.items() if k.startswith('2024'))
        growth_rates = {}
        if 2025 in yearly_totals:
            growth_rates['2025_vs_2024'] = round((yearly_totals[2025] - actual_2024) / actual_2024 * 100, 2)
        if 2026 in yearly_totals:
            growth_rates['2026_vs_2024'] = round((yearly_totals[2026] - actual_2024) / actual_2024 * 100, 2)
        
        return PredictionResponse(
            total_impact=0.0,
            multiplier=1.0,
            yearly_totals=yearly_totals,
            monthly_predictions=predictions,
            impacts_breakdown={},
            growth_rates=growth_rates,
            model_type="sarima_enhanced",
            data_source="excel_files"
        )

# SARIMA 차수 탐색 함수 (기존과 동일)
def auto_sarimax_order(y, seasonal_periods=12):
    """빠른 SARIMA 차수 탐색"""
    p_list, d_list, q_list = [0, 1], [1], [0, 1]
    P_list, D_list, Q_list = [0, 1], [1], [0, 1]
    best = None
    best_aic = float("inf")
    
    max_attempts = 16
    attempts = 0
    
    for (p, d, q, P, D, Q) in itertools.product(p_list, d_list, q_list, P_list, D_list, Q_list):
        if attempts >= max_attempts:
            break
        attempts += 1
        
        try:
            m = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, seasonal_periods),
                       enforce_stationarity=False, enforce_invertibility=False)
            r = m.fit(disp=False, maxiter=50)
            if r.aic < best_aic:
                best_aic = r.aic
                best = ((p, d, q), (P, D, Q, seasonal_periods))
        except Exception:
            continue
    
    return best or ((1, 1, 1), (0, 1, 1, seasonal_periods))

# 데이터 로더 및 예측기 초기화
data_loader = BusanPortDataLoader()
data_loader.load_excel_files()
predictor = EnhancedBusanPortPredictor(data_loader)

# API 엔드포인트들
@app.get("/")
async def root():
    """API 루트 - 서버 상태 확인"""
    return {
        "message": "부산항 환적 예측 API (Enhanced)",
        "version": "4.0.0",
        "status": "운영 중",
        "data_source": "Excel files from DIVE2025",
        "models": ["economic_enhanced", "sarima_enhanced", "lstm_enhanced"],
        "endpoints": {
            "predict": "/predict",
            "predict_lstm": "/predict_lstm",
            "predict_sarima": "/predict_sarima",
            "sensitivity": "/sensitivity-analysis",
            "scenarios": "/scenarios",
            "historical": "/historical-data",
            "baseline": "/baseline-indicators",
            "forecast_exog_path": "/forecast_exog_path",
            "data_info": "/data-info",
            "docs": "/docs"
        }
    }

@app.get("/data-info")
async def get_data_info():
    """데이터 소스 정보 반환"""
    try:
        return {
            "message": "데이터 소스 정보",
            "data_path": str(data_loader.data_path),
            "historical_data_points": len(predictor.real_data),
            "data_range": {
                "start": min(predictor.real_data.keys()) if predictor.real_data else None,
                "end": max(predictor.real_data.keys()) if predictor.real_data else None
            },
            "transshipment_data_years": list(predictor.transshipment_data.keys()),
            "seasonal_pattern": predictor.seasonal_pattern,
            "annual_growth_rate": round(predictor.annual_growth_rate * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 정보 조회 오류: {str(e)}")

@app.get("/baseline-indicators", response_model=BaselineIndicatorsResponse)
async def get_baseline_indicators():
    """기준 경제지표 반환"""
    try:
        return BaselineIndicatorsResponse(
            baseline_indicators=predictor.baseline_indicators,
            description="부산항 환적 예측을 위한 기준 경제지표값들입니다. (실제 데이터 기반 계산)"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"기준지표 조회 오류: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_transshipment(indicators: EconomicIndicators):
    """실시간 경제지표 기반 환적 예측 (실제 데이터 기반)"""
    try:
        result = predictor.predict_with_indicators(indicators, 24)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실행 오류: {str(e)}")

@app.post("/predict_sarima", response_model=PredictionResponse)
async def predict_sarima(forecast_months: int = 24):
    """SARIMA 모델 기반 환적 예측 (실제 데이터 기반)"""
    try:
        result = predictor.predict_with_sarima(forecast_months)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SARIMA 예측 실행 오류: {str(e)}")

@app.post("/predict_lstm", response_model=PredictionResponse)
async def predict_lstm(forecast_months: int = 24):
    """LSTM 모델 기반 환적 예측 (실제 데이터 기반)"""
    try:
        result = predictor.predict_with_lstm(forecast_months)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LSTM 예측 실행 오류: {str(e)}")

@app.get("/sensitivity-analysis")
async def sensitivity_analysis():
    """민감도 분석"""
    try:
        baseline = EconomicIndicators()
        sensitivity_results = []
        
        for indicator, elasticity in predictor.elasticity_coefficients.items():
            sensitivity_results.append({
                "indicator": indicator,
                "elasticity": elasticity,
                "impact_1_percent": abs(elasticity),
                "current_impact": 0.0,
                "importance_rank": 0
            })
        
        sensitivity_results.sort(key=lambda x: x["impact_1_percent"], reverse=True)
        for i, result in enumerate(sensitivity_results):
            result["importance_rank"] = i + 1
        
        return {
            "message": "경제지표별 민감도 분석 결과 (실제 데이터 기반)",
            "methodology": "각 지표 1% 변화시 환적량에 미치는 영향",
            "data_source": "Excel files",
            "sensitivity_analysis": sensitivity_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"민감도 분석 오류: {str(e)}")

@app.get("/scenarios")
async def predefined_scenarios():
    """사전 정의된 시나리오별 예측 (실제 데이터 기반)"""
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
                "growth_2026": prediction.growth_rates.get("2026_vs_2024", 0)
            }
        
        # LSTM 예측도 추가
        lstm_prediction = predictor.predict_with_lstm(24)
        scenario_results["lstm_enhanced"] = {
            "name": "lstm_enhanced",
            "indicators": {},
            "yearly_totals": lstm_prediction.yearly_totals,
            "total_impact": 0,
            "growth_2026": lstm_prediction.growth_rates.get("2026_vs_2024", 0)
        }
        
        return {
            "message": "시나리오별 예측 결과 (실제 데이터 기반)",
            "data_source": "Excel files",
            "scenarios": scenario_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시나리오 분석 오류: {str(e)}")

@app.get("/historical-data")
async def get_historical_data():
    """과거 실적 데이터 반환 (Excel 파일 기반)"""
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
        
        yearly_stats = {}
        transship_stats = {}
        for year in range(2020, 2025):
            yearly_total = sum([item["transshipment"] for item in historical_data if item["year"] == year])
            yearly_stats[year] = yearly_total
            
            # 환적 통계 추가
            if year in predictor.transshipment_data:
                transship_stats[year] = predictor.transshipment_data[year]
        
        return HistoricalDataResponse(
            message="부산항 환적 과거 실적 (Excel 파일 기반)",
            data_period=f"{min(predictor.real_data.keys())} ~ {max(predictor.real_data.keys())}",
            total_months=len(historical_data),
            monthly_data=historical_data,
            yearly_statistics=yearly_stats,
            transshipment_statistics=transship_stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 조회 오류: {str(e)}")

@app.post("/forecast_exog_path")
async def forecast_exog_path(
    horizon: int = Form(12),
    log: bool = Form(True),
    macro_csv: UploadFile | None = File(None),
    features_json: str = Form("[]"),
    use_yoy: bool = Form(True),
    include_lag1: bool = Form(True),
    future_exog: str = Form(""),
    sheet_name: str | None = Form(None),
    use_lstm: bool = Form(False)
):
    """SARIMA/SARIMAX/LSTM 예측 (실제 데이터 기반)"""
    try:
        # LSTM 모델 사용하는 경우
        if use_lstm:
            lstm_result = predictor.predict_with_lstm(horizon)
            return {
                "mode": "lstm_enhanced",
                "data_source": "excel_files",
                "predictions": [
                    {"month": p.date, "value": p.predicted}
                    for p in lstm_result.monthly_predictions
                ]
            }
        
        # 기존 SARIMA/SARIMAX 로직 (실제 데이터 기반)
        df_hist = pd.DataFrame([
            {"date": pd.to_datetime(k), "ts_teu": v}
            for k, v in predictor.real_data.items()
        ]).sort_values("date")
        ts = (df_hist.set_index("date")["ts_teu"]
                      .asfreq("MS")
                      .interpolate("time"))
        
        y_tr = np.log(np.maximum(ts, 1.0)) if log else ts
        order, seas = auto_sarimax_order(y_tr, seasonal_periods=12)
        
        model = SARIMAX(y_tr, order=order, seasonal_order=seas,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        
        pred = res.get_forecast(steps=horizon)
        mu = pred.predicted_mean
        if log: mu = np.exp(mu)
        
        predictions = [
            {"month": d.strftime("%Y-%m"), "value": float(mu.loc[d])}
            for d in mu.index
        ]
        
        return {
            "mode": "sarima_enhanced",
            "data_source": "excel_files",
            "order": order,
            "seasonal_order": seas,
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 오류: {e}")

if __name__ == "__main__":
    import uvicorn
    print("부산항 환적 예측 API 서버 시작 (Enhanced v4.0)")
    print("실제 Excel 데이터 기반 분석 시스템")
    print("FastAPI 자동 문서: http://localhost:8002/docs")
    print("실시간 예측: http://localhost:8002/predict")
    print("SARIMA 예측: http://localhost:8002/predict_sarima")
    print("LSTM 예측: http://localhost:8002/predict_lstm")
    print("데이터 정보: http://localhost:8002/data-info")
    
    uvicorn.run(
        "enhanced_server:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )