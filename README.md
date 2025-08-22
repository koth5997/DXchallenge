# DIVE 2025 해커톤 - 부산항 데이터 처리와 미래 예측
이 레포는 **DIVE 2025 해커톤**에서 진행한  **부산항 물동량·입출항 데이터 처리** 과제를 위한 Python 스크립트입니다.

** !!!파일 수정으로 인해 enhanced_sever.py 와 App2를 실행시켜주세요 포트번호 8002 **
**데이터 파일은 규정상 올릴수 없습니다**
---

# 부산항 환적 물동량 예측 시스템 (FastAPI × React)

부산항 월별 환적 물동량(TEU)을 **실시간 경제지표**와 **시계열 모델**로 예측하는 대시보드입니다.

* 프론트엔드: React + Recharts (탭별 예측/민감도/시나리오/과거실적/EXOG 경로)
* 백엔드: FastAPI + statsmodels(SARIMA/SARIMAX)
* 입력: GDP·유가·환율·운임지수·항만연결성·글로벌무역·공급망지수 등
* 출력: 월별 예측, 연간 합계, 성장률(2025/2026), 탄력성 기반 영향도

---

## 주요 기능

* **물동량 예측(시나리오)**: 사용자가 입력한 경제지표 → \*\*탄력성 승수(multiplier)\*\*를 계산 → **SARIMA** 베이스라인 예측값에 곱하여 반영
* **민감도(탄력성) 분석**: 각 지표 **1% 변화 시 환적량 변화율**을 탄력성으로 표/차트 제공
* **시나리오 비교**: 기준/낙관/비관/회복 시나리오 미리 제공(연간 합계·성장률 비교)
* **과거 실적**: 2020.01–2024.12 월별 실적 및 연도별 합계 시각화
* **EXOG 경로 예측**: CSV/XLSX 업로드 + 월별 미래 경로(전년동월 대비 % 등) 지정 → \*\*SARIMAX(exog)\*\*로 예측 (미제공 시 SARIMA 폴백)

---

##  아키텍처 개요

```
React(App.js) ──(HTTP/JSON)──▶ FastAPI(server.py)
  ├─ /predict                ──▶ SARIMA 기반 베이스라인 × 탄력성 승수
  ├─ /sensitivity-analysis   ──▶ 지표별 탄력성 테이블/차트 데이터
  ├─ /scenarios              ──▶ 기준·낙관·비관·회복 예측 결과
  ├─ /historical-data        ──▶ 2020–2024 실적 및 연간 통계
  └─ /forecast_exog_path     ──▶ (선택) CSV/XLSX + exog 로 SARIMAX 예측
```

* **CORS** 허용: [http://localhost:3000](http://localhost:3000) (개발용)
* **자동 문서**: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

---

## 수치 계산 방식

### 1) 탄력성(Elasticity) 테이블

각 경제지표에 대해 "지표 1% 변화 → 환적량 변동률"을 정의한 탄력성 계수(예: `china_gdp=0.45`, `oil_price=-0.12` 등)를 사용합니다.

```text
예) 중국 GDP 1%p ↑ → 환적량 0.45% ↑
    유가 1% ↑ → 환적량 0.12% ↓
```

> 탄력성은 모델 내부 딕셔너리로 관리되며, 분석/문헌의 가정값을 사용했습니다.

### 2) 변화율 계산 규칙 (change\_percent)

지표 성격에 따라 **두 가지 방식**을 적용합니다.

* **% 지표(성장률류)**: `china_gdp, us_gdp, japan_gdp, korea_gdp, global_trade`

  * `change_percent = (현재값 − 기준값)`
  * 단위: **퍼센트 포인트(pp)**

* **지수/레벨 지표(수준류)**: `oil_price, exchange_rate, container_rate, port_connectivity, supply_chain`

  * `change_percent = (현재값 − 기준값) / 기준값 × 100`
  * 단위: **% 변화**

### 3) 총 영향도와 승수(multiplier)

각 지표별 영향도를 합산해 총 영향도를 만들고, 예측치에 곱할 **승수**를 산출합니다.

* 지표별 영향도:

  * `impact_i = (change_percent_i / 100) × elasticity_i`
* 총 영향도:

  * `total_impact = Σ impact_i`
* **승수(multiplier)**:

  * `multiplier = 1 + total_impact`

> 예: 총 영향도 `+0.032` → 승수 `1.032`(= +3.2%)

### 4) 베이스라인 예측 (SARIMA)

* 2020.01–2024.12 **월별 TEU 실적**으로 시계열을 구성
* 로그 변환 후 **SARIMA 파라미터를 AIC 최소**로 자동 탐색
* 향후 `N`개월 예측치를 산출 → \*\*승수(multiplier)\*\*를 곱해 최종값 생성

### 5) 성장률 계산

* `2025_vs_2024` = `(2025연간예측 − 2024연간실적) / 2024연간실적 × 100(%)`
* `2026_vs_2024`, `2026_vs_2025`도 동일 규칙 적용

### 6) EXOG 경로 + SARIMAX (선택)

CSV/XLSX의 월별 거시지표를 \*\*전년동월 대비(%)\*\*로 변환(옵션), `lag1` 추가(옵션) 후
히스토리/미래 exog 행렬을 만들고 \*\*SARIMAX(exog)\*\*로 예측합니다.
프론트에서 전달한 **월별 미래 경로**(JSON)가 있으면 해당 월부터 exog를 덮어씁니다.

* 파일 컬럼 중 `date/월/Month/기간` 등 날짜 열을 자동 인식 → 월초 기준으로 정렬/집계
* exog 컬럼은 `features`(JSON) 목록과 **교집합만** 사용
* exog가 없거나 파싱 실패 시 **SARIMA 폴백**

---

## 내장 데이터 & 기준값

* **월별 실적(TEU)**: 2020.01–2024.12이 코드에 내장되어 있음
* **기준값(2024년 평균 등)**: GDP·유가·환율·운임·연결성·무역·공급망 지표의 기준치
* **계절성 패턴**(참고용): 1–12월 계절계수(베이스라인 논리 보조)

> 데이터 파일 없이도 동작하며, 필요 시 서버 코드의 딕셔너리를 최신 수치로 갱신하세요.

---

## 기술 스택

* **Frontend**: React, Recharts, lucide-react, Tailwind 스타일 유사 클래스(유틸리티 기반)
* **Backend**: FastAPI, pydantic, statsmodels(SARIMA/SARIMAX), pandas, numpy, openpyxl(엑셀)

---

## 빠른 시작

### 1) 백엔드 (Python 3.10+ 권장)

```bash
pip install -r requirements.txt
uvicorn enhanced_server:app --reload --host 0.0.0.0 --port 8002
```

* 문서: [http://localhost:8002/docs](http://localhost:8002/docs)

### 2) 프론트엔드

1. React 앱을 생성하거나 기존 프로젝트에 `App.js`를 추가합니다.
2. 필요한 라이브러리를 설치합니다.

```bash
npm install recharts lucide-react
# (CRA 기준) 개발 서버 실행
npm start
```

> `App.js` 상단의 `API_BASE_URL`이 `http://localhost:8000`인지 확인하세요.

---

## API 요약

### `POST /predict`

* Body: 경제지표(JSON)
* Return: 월별 예측(24개월), 연간 합계, 성장률, 승수, 지표별 영향도

### `GET /sensitivity-analysis`

* Return: 지표별 탄력성(1% 변화 영향) 테이블 + 정렬 순위

### `GET /scenarios`

* Return: 기준/낙관/비관/회복 각 시나리오의 연간 합계·총 영향·성장률

### `GET /historical-data`

* Return: 2020–2024 월별 실적 + 연도별 합계

### `POST /forecast_exog_path`

* Form-data: `macro_csv`(csv/xlsx), `features_json`(예: `["shippingRate","oilPrice","connectivity"]`), `use_yoy`(bool), `include_lag1`(bool), `future_exog`(월별 JSON), `sheet_name`(옵션)
* Return: `mode=sarimax_exog`일 경우 `scenario`(월별 값), 없으면 `sarima_fallback`

---

## 예시 요청

```bash
# 1) 단순 예측
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "china_gdp": 5.6,
        "us_gdp": 2.7,
        "japan_gdp": 1.0,
        "korea_gdp": 3.3,
        "oil_price": 80,
        "exchange_rate": 1280,
        "container_rate": 2100,
        "port_connectivity": 122,
        "global_trade": 4.6,
        "supply_chain": 90
      }'

# 2) EXOG 경로 예측 (csv/xlsx + features + 옵션)
#   - Postman/브라우저에서 form-data로 전송 권장
```

---

## 프론트 탭 구성

* **물동량 예측**: 경제지표 폼, 결과 요약(총 영향·승수·연간 합계·성장률), 월별 라인차트
* **민감도 분석**: 1% 변화 영향 바차트 + 테이블(순위·탄력성·영향도)
* **시나리오 분석**: 기준/낙관/비관/회복 카드(총 영향·연간합계·성장률)
* **과거 실적**: 월별 실적 라인, 연도별 바차트
* **EXOG 경로**: CSV/XLSX 업로드, 전년동월 대비·지연 포함 옵션, 월별 표 편집, SARIMAX 예측 라인차트

---

## 환경/단위 가이드

* GDP·글로벌무역: **%**(성장률), 입력값은 **퍼센트(예: 3.1)**
* 유가(USD/bbl), 환율(KRW/USD), 운임지수/연결성/공급망: **레벨/지수** 값
* 시작월(YYYY-MM), 예측기간(horizon: 개월)

---

## FAQ / 트러블슈팅

* **CORS 오류**: 프론트/백엔드 포트 확인(3000↔8000), 서버 CORS 허용 확인
* **엑셀 파싱 실패**: `date/월/Month/기간` 중 하나의 날짜열 존재 필요, 첫 시트를 기본으로 사용 (원하면 `sheet_name` 지정)
* **예측이 평평함**: exog 미제공 시 SARIMA 폴백. exog 제공 + `use_yoy/lag1` 옵션을 조정하거나 future\_exog로 월별 경로를 명시하세요.
* **탄력성 조정**: 서버 코드의 탄력성 딕셔너리를 직접 수정 가능(팀 가정 반영)

---

## 라이선스

MIT (예시). 프로젝트 정책에 맞게 수정하세요.

---



