import React, { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar
} from 'recharts';
import { TrendingUp, Ship, Globe, BarChart3, Settings, Info } from 'lucide-react';

// FastAPI 서버 주소
const API_BASE_URL = 'http://localhost:8000';

const BusanPortForecast = () => {
  const [indicators, setIndicators] = useState({
    china_gdp: 5.2,
    us_gdp: 2.5,
    japan_gdp: 0.9,
    korea_gdp: 3.1,
    oil_price: 82.0,
    exchange_rate: 1300,
    container_rate: 2200,
    port_connectivity: 119.5,
    global_trade: 4.2,
    supply_chain: 85.0
  });

  const [prediction, setPrediction] = useState(null);
  const [sensitivity, setSensitivity] = useState(null);
  const [scenarios, setScenarios] = useState(null);
  const [historical, setHistorical] = useState(null);
  const [baseline, setBaseline] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('forecast'); // 'forecast' | 'sensitivity' | 'scenarios' | 'historical' | 'exog'
  
  // ===== EXOG 탭 상태 =====
  const [macroFiles, setMacroFiles] = useState([]); // ✅ 여러 파일 업로드 상태
  const [horizon, setHorizon] = useState(12);
  const [useYoY, setUseYoY] = useState(true);
  const [lag1, setLag1] = useState(true);
  const [startYM, setStartYM] = useState('2025-01');
  const [features, setFeatures] = useState(['shippingRate', 'oilPrice', 'connectivity']);
  const [sheetName, setSheetName] = useState('');

  // 월 리스트 만들기
  const monthsOf = (ym, h) => {
    const [y, m] = ym.split('-').map(Number);
    const out = [];
    let yy = y, mm = m;
    for (let i = 0; i < h; i++) {
      out.push(`${yy}-${String(mm).padStart(2, '0')}`);
      mm++;
      if (mm > 12) { mm = 1; yy++; }
    }
    return out;
  };
  const months = useMemo(() => monthsOf(startYM, horizon), [startYM, horizon]);

  // 월별 외생 경로 (기본 0)
  const [exogPath, setExogPath] = useState(() => {
    const obj = {};
    monthsOf('2025-01', 12).forEach(m => obj[m] = { shippingRate: 0, oilPrice: 0, connectivity: 0 });
    return obj;
  });
  useEffect(() => {
    setExogPath(prev => {
      const next = {};
      months.forEach(m => {
        next[m] = prev[m] ?? { shippingRate: 0, oilPrice: 0, connectivity: 0 };
      });
      return next;
    });
  }, [months]);

  const onChangeExogCell = (ym, key, val) => {
    setExogPath(prev => ({ ...prev, [ym]: { ...prev[ym], [key]: Number(val) || 0 } }));
  };

  const [exogSeries, setExogSeries] = useState([]);

  // 안전한 숫자 포맷팅/체크
  const safeToFixed = (value, decimals = 2) => {
    if (value === null || value === undefined || isNaN(value)) return '0.00';
    return Number(value).toFixed(decimals);
  };
  const safeNumber = (value, defaultValue = 0) => {
    if (value === null || value === undefined || isNaN(value)) return defaultValue;
    return Number(value);
  };

  useEffect(() => {
    fetchHistoricalData();
    fetchSensitivityAnalysis();
    fetchScenarios();
    fetchBaseline();
  }, []);

  const fetchPrediction = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(indicators)
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();

      const validatedData = {
        total_impact: safeNumber(data.total_impact, 0),
        multiplier: safeNumber(data.multiplier, 1),
        yearly_totals: data.yearly_totals || { 2025: 0, 2026: 0 },
        monthly_predictions: data.monthly_predictions || [],
        impacts_breakdown: data.impacts_breakdown || {},
        growth_rates: data.growth_rates || {
          '2025_vs_2024': 0,
          '2026_vs_2024': 0,
          '2026_vs_2025': 0
        }
      };
      setPrediction(validatedData);
    } catch (err) {
      console.error('Prediction error:', err);
      setError(`예측 오류: ${err.message}`);
    }
    setLoading(false);
  };

  const fetchSensitivityAnalysis = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/sensitivity-analysis`);
      if (response.ok) setSensitivity(await response.json());
    } catch (error) {
      console.error('Sensitivity analysis error:', error);
    }
  };

  const fetchScenarios = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/scenarios`);
      if (response.ok) setScenarios(await response.json());
    } catch (error) {
      console.error('Scenarios error:', error);
    }
  };

  const fetchHistoricalData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/historical-data`);
      if (response.ok) setHistorical(await response.json());
    } catch (error) {
      console.error('Historical data error:', error);
    }
  };

  const fetchBaseline = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/baseline-indicators`);
      if (response.ok) setBaseline(await response.json());
    } catch (error) {
      console.error('Baseline data error:', error);
    }
  };

  const handleIndicatorChange = (key, value) => {
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      setIndicators(prev => ({ ...prev, [key]: numValue }));
    }
  };

  const resetToBaseline = () => {
    if (baseline && baseline.baseline_indicators) {
      setIndicators(baseline.baseline_indicators);
    }
  };

  const formatNumber = (num) => new Intl.NumberFormat('ko-KR').format(safeNumber(num, 0));

  const indicatorLabels = {
    china_gdp: '중국 GDP 성장률 (%)',
    us_gdp: '미국 GDP 성장률 (%)',
    japan_gdp: '일본 GDP 성장률 (%)',
    korea_gdp: '한국 GDP 성장률 (%)',
    oil_price: 'WTI 유가 ($/barrel)',
    exchange_rate: '원/달러 환율',
    container_rate: '컨테이너 운임지수',
    port_connectivity: '부산항 연결성지수',
    global_trade: '글로벌 무역 성장률 (%)',
    supply_chain: '공급망 안정성 지수'
  };

  const TabButton = ({ id, label, icon: Icon, active, onClick }) => (
    <button
      onClick={() => onClick(id)}
      className={`flex items-center px-4 py-2 rounded-lg font-medium transition-colors ${
        active ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
      }`}
    >
      <Icon className="w-4 h-4 mr-2" />
      {label}
    </button>
  );

  // ====== 탭 1: 물동량 예측 ======
  const renderForecastTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-800">경제지표 입력</h3>
          <button
            onClick={resetToBaseline}
            className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
          >
            기준값 재설정
          </button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(indicators).map(([key, value]) => (
            <div key={key} className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                {indicatorLabels[key]}
              </label>
              <input
                type="number"
                step="0.1"
                value={value}
                onChange={(e) => handleIndicatorChange(key, e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          ))}
        </div>
        <button
          onClick={fetchPrediction}
          disabled={loading}
          className="mt-4 w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? '예측 중...' : '물동량 예측하기'}
        </button>
        
        {error && (
          <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}
      </div>

      {prediction && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">예측 결과 요약</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">총 경제적 영향:</span>
                <span className={`font-semibold ${safeNumber(prediction.total_impact) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {safeToFixed(safeNumber(prediction.total_impact) * 100, 2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">승수 효과:</span>
                <span className="font-semibold">{safeToFixed(safeNumber(prediction.multiplier, 1), 4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">2025년 예상 물동량:</span>
                <span className="font-semibold text-blue-600">
                  {formatNumber(safeNumber(prediction.yearly_totals?.[2025], 0))} TEU
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">2026년 예상 물동량:</span>
                <span className="font-semibold text-blue-600">
                  {formatNumber(safeNumber(prediction.yearly_totals?.[2026], 0))} TEU
                </span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">성장률 분석</h3>
            <div className="space-y-3">
              {prediction.growth_rates && Object.entries(prediction.growth_rates).map(([key, value]) => (
                <div key={key} className="flex justify-between">
                  <span className="text-gray-600">{key.replace('_', ' vs ')}:</span>
                  <span className={`font-semibold ${safeNumber(value) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {safeNumber(value) > 0 ? '+' : ''}{safeToFixed(safeNumber(value), 2)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {prediction && prediction.monthly_predictions && prediction.monthly_predictions.length > 0 && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">월별 예측 물동량</h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={prediction.monthly_predictions}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip formatter={(value) => [formatNumber(value), 'TEU']} />
              <Legend />
              <Line type="monotone" dataKey="baseline" stroke="#94a3b8" name="기준선" />
              <Line type="monotone" dataKey="predicted" stroke="#2563eb" name="예측값" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );

  // ====== 탭 2: 민감도 분석 ======
  const renderSensitivityTab = () => (
    <div className="space-y-6">
      {sensitivity && sensitivity.sensitivity_analysis && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">민감도 분석</h3>
          <p className="text-gray-600 mb-4">{sensitivity.methodology}</p>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={sensitivity.sensitivity_analysis}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="indicator" angle={-45} textAnchor="end" height={100} />
              <YAxis />
              <Tooltip formatter={(value) => [`${safeToFixed(safeNumber(value), 2)}%`, '1% 변화시 영향']} />
              <Bar dataKey="impact_1_percent" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-4 overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">순위</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">지표</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">탄력성</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">1% 변화시 영향</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {sensitivity.sensitivity_analysis.map((item, index) => (
                  <tr key={index}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{item.importance_rank}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{indicatorLabels[item.indicator] || item.indicator}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{safeToFixed(safeNumber(item.elasticity), 3)}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{safeToFixed(safeNumber(item.impact_1_percent), 2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );

  // ====== 탭 3: 시나리오 ======
  const renderScenariosTab = () => (
    <div className="space-y-6">
      {scenarios && scenarios.scenarios && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.entries(scenarios.scenarios).map(([scenarioName, scenarioData]) => (
            <div key={scenarioName} className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 capitalize">
                {scenarioName === 'baseline' ? '기준 시나리오' :
                 scenarioName === 'optimistic' ? '낙관적 시나리오' :
                 scenarioName === 'pessimistic' ? '비관적 시나리오' :
                 scenarioName === 'recovery' ? '회복 시나리오' : scenarioName}
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">총 경제적 영향:</span>
                  <span className={`font-semibold ${safeNumber(scenarioData.total_impact) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {safeToFixed(safeNumber(scenarioData.total_impact) * 100, 2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">2025년 예상:</span>
                  <span className="font-semibold">
                    {formatNumber(safeNumber(scenarioData.yearly_totals?.[2025], 0))} TEU
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">2026년 예상:</span>
                  <span className="font-semibold">
                    {formatNumber(safeNumber(scenarioData.yearly_totals?.[2026], 0))} TEU
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">2026년 성장률:</span>
                  <span className={`font-semibold ${safeNumber(scenarioData.growth_2026) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {safeNumber(scenarioData.growth_2026) > 0 ? '+' : ''}{safeToFixed(safeNumber(scenarioData.growth_2026), 2)}%
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  // ====== 탭 4: 과거 실적 ======
  const renderHistoricalTab = () => (
    <div className="space-y-6">
      {historical && (
        <>
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">과거 실적 데이터</h3>
            <p className="text-gray-600 mb-4">데이터 기간: {historical.data_period}</p>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={historical.monthly_data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip formatter={(value) => [formatNumber(value), 'TEU']} />
                <Line type="monotone" dataKey="transshipment" stroke="#2563eb" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">연도별 통계</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={Object.entries(historical.yearly_statistics || {}).map(([year, value]) => ({ year, value }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip formatter={(value) => [formatNumber(value), 'TEU']} />
                <Bar dataKey="value" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );

  // ====== 탭 5: EXOG 경로 예측 ======
  const runExogForecast = async () => {
    try {
      const form = new FormData();

      // ✅ 현재 서버가 macro_csv 단일 파일만 받는 경우: 첫 번째 파일만 전송
      if (macroFiles.length > 0) {
        form.append('macro_csv', macroFiles[0]);
      }
      // 🔁 만약 서버가 다중 파일을 받도록 (List[UploadFile] macro_files)로 바뀌면 아래로 교체
      // (macroFiles || []).forEach(f => form.append('macro_files', f));

      form.append('horizon', String(horizon));
      form.append('log', 'true');
      form.append('features_json', JSON.stringify(features));
      form.append('use_yoy', String(useYoY));
      form.append('include_lag1', String(lag1));
      form.append('future_exog', JSON.stringify(exogPath));
      if (sheetName) form.append('sheet_name', sheetName);

      const res = await fetch(`${API_BASE_URL}/forecast_exog_path`, {
        method: 'POST',
        body: form
      });
      const data = await res.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      const arr = (data.scenario || data.baseline || []).map(d => ({
        month: d.month,
        value: Math.round(safeNumber(d.value, 0))
      }));
      setExogSeries(arr);
    } catch (e) {
      console.error(e);
      alert('EXOG 예측 호출 실패');
    }
  };
  
  const renderExogTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">월별 외생 경로 입력</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* 좌: 설정 */}
          <div>
            <div className="mb-2">csv, xlsx (선택)</div>
            <input
              type="file"
              multiple
              accept=".csv,.xlsx,.xls,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel"
              onChange={e => setMacroFiles(Array.from(e.target.files || []))}
            />
            {macroFiles?.length > 0 && (
              <ul className="mt-2 text-sm text-gray-600 list-disc ml-5">
                {macroFiles.map(f => <li key={f.name}>{f.name}</li>)}
              </ul>
            )}
            <div className="mt-2 text-sm text-gray-500">
              CSV/XLSX를 여러 개 선택할 수 있습니다(서로 다른 지표 파일을 합칩니다).
            </div>
                   
            <div className="mt-2">
              <label className="block mb-1">시트명/번호(선택)</label>
              <input
                className="border px-2 py-1"
                value={sheetName}
                onChange={e => setSheetName(e.target.value)}
                placeholder="예: 0 또는 Sheet1"
              />
              <div className="text-xs text-gray-500">
                비우면 첫 번째 시트를 사용합니다. 숫자 입력 시 인덱스로 처리(0,1,2…)
              </div>
            </div>
            <div className="mt-4">
              <label className="mr-2">시작(YYYY-MM)</label>
              <input className="border px-2 py-1" value={startYM} onChange={e => setStartYM(e.target.value)} />
            </div>
            <div className="mt-2">
              <label className="mr-2">horizon(개월)</label>
              <input type="number" className="border px-2 py-1" value={horizon} onChange={e => setHorizon(Number(e.target.value) || 12)} />
            </div>
            <div className="mt-2">
              <label className="mr-2"><input type="checkbox" checked={useYoY} onChange={e => setUseYoY(e.target.checked)} /> 전년동월 대비(%)</label>
            </div>
            <div className="mt-1">
              <label className="mr-2"><input type="checkbox" checked={lag1} onChange={e => setLag1(e.target.checked)} /> 1개월 지연 포함</label>
            </div>
            <div className="mt-2">
              <label className="block mb-1">features(JSON)</label>
              <input
                className="w-full border px-2 py-1"
                value={JSON.stringify(features)}
                onChange={e => {
                  try { setFeatures(JSON.parse(e.target.value || '[]')); }
                  catch { /* 무시 */ }
                }}
              />
              <div className="text-xs text-gray-500 mt-1">CSV 컬럼명과 동일하게 적으세요</div>
            </div>

            <button
              onClick={runExogForecast}
              className="mt-4 w-full px-6 py-3 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700"
            >
              EXOG 경로로 예측 실행
            </button>
          </div>

          {/* 우: 월별 테이블 */}
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr>
                  <th className="border px-2 py-1">월</th>
                  {features.map(f => <th key={f} className="border px-2 py-1">{f}</th>)}
                </tr>
              </thead>
              <tbody>
                {months.map(m => (
                  <tr key={m}>
                    <td className="border px-2 py-1">{m}</td>
                    {features.map(f => (
                      <td key={f} className="border px-2 py-1">
                        <input
                          className="w-24 border px-2 py-1"
                          value={safeNumber(exogPath[m]?.[f], 0)}
                          onChange={e => onChangeExogCell(m, f, e.target.value)}
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="text-xs text-gray-500 mt-1">예: 2025-08부터 shippingRate=10 등으로 입력하면 그 달부터만 반영됩니다.</div>
          </div>
        </div>
      </div>

      {/* 예측 결과 차트 */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">EXOG 기반 월별 예측</h3>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={exogSeries}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip formatter={(v) => [v?.toLocaleString(), '예측']} />
            <Legend />
            <Line type="monotone" dataKey="value" name="시나리오" stroke="#10b981" strokeWidth={3} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Ship className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">부산항 환적 물동량 예측 시스템</h1>
                <p className="text-gray-600">실시간 경제지표 기반 부산항 환적 물동량 예측</p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500">Version 2.0.0</div>
              <div className="text-sm text-blue-600">Powered by FastAPI & React</div>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-2 mb-6 overflow-x-auto">
          <TabButton id="forecast" label="물동량 예측" icon={TrendingUp} active={activeTab === 'forecast'} onClick={setActiveTab} />
          <TabButton id="sensitivity" label="민감도 분석" icon={BarChart3} active={activeTab === 'sensitivity'} onClick={setActiveTab} />
          <TabButton id="scenarios" label="시나리오 분석" icon={Globe} active={activeTab === 'scenarios'} onClick={setActiveTab} />
          <TabButton id="historical" label="과거 실적" icon={Info} active={activeTab === 'historical'} onClick={setActiveTab} />
          <TabButton id="exog" label="EXOG 경로 예측" icon={Settings} active={activeTab === 'exog'} onClick={setActiveTab} />
        </div>

        {/* Tab Content */}
        {activeTab === 'forecast' && renderForecastTab()}
        {activeTab === 'sensitivity' && renderSensitivityTab()}
        {activeTab === 'scenarios' && renderScenariosTab()}
        {activeTab === 'historical' && renderHistoricalTab()}
        {activeTab === 'exog' && renderExogTab()}
      </div>
    </div>
  );
};

export default BusanPortForecast;
