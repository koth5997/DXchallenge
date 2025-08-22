import React, { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell
} from 'recharts';
import { TrendingUp, Ship, Globe, BarChart3, Settings, Info, Brain, Database, Activity } from 'lucide-react';

// FastAPI 서버 주소
const API_BASE_URL = 'http://localhost:8002';

const EnhancedBusanPortForecast = () => {
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
  const [lstmPrediction, setLstmPrediction] = useState(null);
  const [sarimaPrediction, setSarimaPrediction] = useState(null);
  const [sensitivity, setSensitivity] = useState(null);
  const [scenarios, setScenarios] = useState(null);
  const [historical, setHistorical] = useState(null);
  const [baseline, setBaseline] = useState(null);
  const [dataInfo, setDataInfo] = useState(null); // 새로운 데이터 정보 상태
  const [loading, setLoading] = useState(false);
  const [lstmLoading, setLstmLoading] = useState(false);
  const [sarimaLoading, setSarimaLoading] = useState(false);
  const [exogLoading, setExogLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('forecast');
  
  // EXOG 탭 상태
  const [macroFiles, setMacroFiles] = useState([]);
  const [horizon, setHorizon] = useState(12);
  const [useYoY, setUseYoY] = useState(true);
  const [lag1, setLag1] = useState(true);
  const [startYM, setStartYM] = useState('2025-01');
  
  // Features 상태
  const [availableFeatures] = useState(['shippingRate', 'oilPrice', 'connectivity', 'gdp', 'exchange']);
  const [selectedFeatures, setSelectedFeatures] = useState(['shippingRate', 'oilPrice', 'connectivity']);
  
  const [sheetName, setSheetName] = useState('');
  const [useLSTM, setUseLSTM] = useState(false);
  
  // 모델별 예측 결과 저장
  const [sarimaPredictions, setSarimaPredictions] = useState([]);
  const [sarimaxPredictions, setSarimaxPredictions] = useState([]);
  const [lstmModelPredictions, setLstmModelPredictions] = useState([]);

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

  // 월별 외생 경로
  const [exogPath, setExogPath] = useState(() => {
    const obj = {};
    monthsOf('2025-01', 12).forEach(m => {
      obj[m] = { shippingRate: 0, oilPrice: 0, connectivity: 0, gdp: 0, exchange: 0 };
    });
    return obj;
  });
  
  useEffect(() => {
    setExogPath(prev => {
      const next = {};
      months.forEach(m => {
        next[m] = prev[m] ?? { shippingRate: 0, oilPrice: 0, connectivity: 0, gdp: 0, exchange: 0 };
      });
      return next;
    });
  }, [months]);

  const onChangeExogCell = (ym, key, val) => {
    setExogPath(prev => ({ ...prev, [ym]: { ...prev[ym], [key]: Number(val) || 0 } }));
  };

  // 안전한 숫자 포맷팅
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
    fetchDataInfo(); // 새로운 데이터 정보 가져오기
    // 기본 모델들 자동 실행
    fetchSARIMAPrediction();
  }, []);

  // 새로운 데이터 정보 fetch 함수
  const fetchDataInfo = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/data-info`);
      if (response.ok) {
        const data = await response.json();
        setDataInfo(data);
      }
    } catch (error) {
      console.error('Data info fetch error:', error);
    }
  };

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
      setPrediction(data);
    } catch (err) {
      console.error('Prediction error:', err);
      setError(`예측 오류: ${err.message}`);
    }
    setLoading(false);
  };

  const fetchLSTMPrediction = async () => {
    setLstmLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/predict_lstm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ forecast_months: 24 })
      });
      if (response.ok) {
        const data = await response.json();
        setLstmPrediction(data);
      }
    } catch (error) {
      console.error('LSTM prediction error:', error);
    }
    setLstmLoading(false);
  };

  const fetchSARIMAPrediction = async () => {
    setSarimaLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/predict_sarima`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ forecast_months: 24 })
      });
      if (response.ok) {
        const data = await response.json();
        setSarimaPrediction(data);
      }
    } catch (error) {
      console.error('SARIMA prediction error:', error);
    }
    setSarimaLoading(false);
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
      if (response.ok) {
        const data = await response.json();
        setBaseline(data);
      }
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

  const featureLabels = {
    shippingRate: '선적료',
    oilPrice: '유가',
    connectivity: '연결성',
    gdp: 'GDP',
    exchange: '환율'
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

  // 세 모델 비교 차트 데이터 준비 (메모이제이션)
  const comparisonData = useMemo(() => {
    const data = [];
    const maxLength = Math.max(
      prediction?.monthly_predictions?.length || 0,
      lstmPrediction?.monthly_predictions?.length || 0,
      sarimaPrediction?.monthly_predictions?.length || 0
    );
    
    if (maxLength === 0) return [];
    
    for (let i = 0; i < maxLength; i++) {
      const item = {
        date: prediction?.monthly_predictions?.[i]?.date || 
               lstmPrediction?.monthly_predictions?.[i]?.date || 
               sarimaPrediction?.monthly_predictions?.[i]?.date
      };
      
      if (prediction && prediction.monthly_predictions?.[i]) {
        item.economic = prediction.monthly_predictions[i].predicted;
      }
      
      if (lstmPrediction && lstmPrediction.monthly_predictions?.[i]) {
        item.lstm = lstmPrediction.monthly_predictions[i].predicted;
      }
      
      if (sarimaPrediction && sarimaPrediction.monthly_predictions?.[i]) {
        item.sarima = sarimaPrediction.monthly_predictions[i].predicted;
      }
      
      data.push(item);
    }
    
    return data;
  }, [prediction, lstmPrediction, sarimaPrediction]);

  // EXOG 차트 데이터 메모이제이션
  const exogChartData = useMemo(() => {
    const maxLength = Math.max(sarimaPredictions.length, sarimaxPredictions.length, lstmModelPredictions.length);
    if (maxLength === 0) return [];
    
    return [...Array(maxLength)].map((_, i) => ({
      month: sarimaPredictions[i]?.month || sarimaxPredictions[i]?.month || lstmModelPredictions[i]?.month,
      sarima: sarimaPredictions[i]?.value,
      sarimax: sarimaxPredictions[i]?.value,
      lstm: lstmModelPredictions[i]?.value
    }));
  }, [sarimaPredictions, sarimaxPredictions, lstmModelPredictions]);

  // 환적 비율 파이 차트 데이터
  const transshipmentPieData = useMemo(() => {
    if (!historical?.transshipment_statistics) return [];
    
    const latestYear = Math.max(...Object.keys(historical.transshipment_statistics).map(Number));
    const latestData = historical.transshipment_statistics[latestYear];
    
    if (!latestData) return [];
    
    return [
      { name: '환적', value: latestData.transship_teu || 0, color: '#0088FE' },
      { name: '수출입', value: (latestData.total_teu || 0) - (latestData.transship_teu || 0), color: '#00C49F' }
    ];
  }, [historical]);

  // 화적량 예측 탭
  const renderForecastTab = () => (
    <div className="space-y-6">
      {/* 데이터 소스 정보 패널 */}
      {dataInfo && (
        <div className="bg-gradient-to-r from-blue-50 to-green-50 rounded-lg shadow-lg p-6 border-l-4 border-blue-500">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-800 flex items-center">
              <Database className="w-5 h-5 mr-2 text-blue-600" />
              실제 데이터 기반 분석 시스템
            </h3>
            <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
              Enhanced v4.0
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-600">데이터 포인트:</span>
              <span className="font-semibold ml-2">{dataInfo.historical_data_points}개월</span>
            </div>
            <div>
              <span className="text-gray-600">데이터 기간:</span>
              <span className="font-semibold ml-2">{dataInfo.data_range?.start} ~ {dataInfo.data_range?.end}</span>
            </div>
            <div>
              <span className="text-gray-600">연평균 성장률:</span>
              <span className="font-semibold ml-2">{dataInfo.annual_growth_rate}%</span>
            </div>
          </div>
        </div>
      )}

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
        <div className="mt-4 flex space-x-2">
          <button
            onClick={fetchPrediction}
            disabled={loading}
            className="flex-1 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors flex items-center justify-center"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                경제모델 분석 중
              </>
            ) : (
              <>
                <Activity className="w-4 h-4 mr-2" />
                경제모델 예측 (Enhanced)
              </>
            )}
          </button>
          <button
            onClick={fetchSARIMAPrediction}
            disabled={sarimaLoading}
            className="flex-1 px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors flex items-center justify-center"
          >
            {sarimaLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                SARIMA 분석 중
              </>
            ) : (
              'SARIMA 예측 (Enhanced)'
            )}
          </button>
          <button
            onClick={fetchLSTMPrediction}
            disabled={lstmLoading}
            className="flex-1 px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 transition-colors flex items-center justify-center"
          >
            {lstmLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                LSTM 분석 중
              </>
            ) : (
              <>
                <Brain className="w-4 h-4 mr-2" />
                LSTM 예측 (Enhanced)
              </>
            )}
          </button>
        </div>
        
        {error && (
          <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}
        
        {/* 진행 상황 표시 */}
        {(loading || lstmLoading || sarimaLoading) && (
          <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded">
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-3"></div>
              <div className="text-blue-800">
                {loading && 'Excel 데이터 기반 경제모델 예측 중... (2-3초 소요)'}
                {sarimaLoading && 'Excel 데이터 기반 SARIMA 시계열 분석 중... (3-5초 소요)'}
                {lstmLoading && 'Excel 데이터 기반 LSTM 모델 학습 중... (5-10초 소요)'}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 환적 비율 시각화 */}
      {transshipmentPieData.length > 0 && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            부산항 환적 비율 (실제 데이터 기반)
          </h3>
          <div className="flex justify-center">
            <ResponsiveContainer width={300} height={200}>
              <PieChart>
                <Pie
                  data={transshipmentPieData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={(entry) => `${entry.name}: ${(entry.value / transshipmentPieData.reduce((a, b) => a + b.value, 0) * 100).toFixed(1)}%`}
                >
                  {transshipmentPieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [formatNumber(value), 'TEU']} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* 모델 비교 차트 */}
      {(prediction || lstmPrediction || sarimaPrediction) && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            AI 모델별 예측 비교 (실제 데이터 기반)
          </h3>
          <div className="mb-4 text-sm text-gray-600">
            ✨ Excel 파일에서 추출한 실제 부산항 환적 데이터를 기반으로 한 AI 모델 예측 결과
          </div>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip formatter={(value) => [formatNumber(value), 'TEU']} />
              <Legend />
              {prediction && (
                <Line 
                  type="monotone" 
                  dataKey="economic" 
                  stroke="#2563eb" 
                  name="경제모델 (Enhanced)" 
                  strokeWidth={3}
                />
              )}
              {sarimaPrediction && (
                <Line 
                  type="monotone" 
                  dataKey="sarima" 
                  stroke="#10b981" 
                  name="SARIMA (Enhanced)" 
                  strokeWidth={2}
                  strokeDasharray="3 3"
                />
              )}
              {lstmPrediction && (
                <Line 
                  type="monotone" 
                  dataKey="lstm" 
                  stroke="#9333ea" 
                  name="LSTM (Enhanced)" 
                  strokeWidth={2}
                  strokeDasharray="5 5"
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* 예측 결과 요약 */}
      {(prediction || sarimaPrediction || lstmPrediction) && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {prediction && (
            <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-blue-500">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">경제모델 예측 (Enhanced)</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">데이터 소스:</span>
                  <span className="font-semibold text-blue-600">Excel Files</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">모델 타입:</span>
                  <span className="font-semibold text-blue-600">{prediction.model_type}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">평균 경제 영향:</span>
                  <span className={`font-semibold ${safeNumber(prediction.total_impact) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {safeToFixed(safeNumber(prediction.total_impact) * 100, 2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">2025년 예상 환적량:</span>
                  <span className="font-semibold text-blue-600">
                    {formatNumber(safeNumber(prediction.yearly_totals?.[2025], 0))} TEU
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">2026년 예상 환적량:</span>
                  <span className="font-semibold text-blue-600">
                    {formatNumber(safeNumber(prediction.yearly_totals?.[2026], 0))} TEU
                  </span>
                </div>
              </div>
            </div>
          )}

          {sarimaPrediction && (
            <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-green-500">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">SARIMA 예측 (Enhanced)</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">데이터 소스:</span>
                  <span className="font-semibold text-green-600">Excel Files</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">모델 타입:</span>
                  <span className="font-semibold text-green-600">{sarimaPrediction.model_type}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">2025년 예상 환적량:</span>
                  <span className="font-semibold text-green-600">
                    {formatNumber(safeNumber(sarimaPrediction.yearly_totals?.[2025], 0))} TEU
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">2026년 예상 환적량:</span>
                  <span className="font-semibold text-green-600">
                    {formatNumber(safeNumber(sarimaPrediction.yearly_totals?.[2026], 0))} TEU
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">성장률 (2026 vs 2024):</span>
                  <span className={`font-semibold ${safeNumber(sarimaPrediction.growth_rates?.['2026_vs_2024']) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {safeNumber(sarimaPrediction.growth_rates?.['2026_vs_2024']) > 0 ? '+' : ''}{safeToFixed(safeNumber(sarimaPrediction.growth_rates?.['2026_vs_2024']), 2)}%
                  </span>
                </div>
              </div>
            </div>
          )}

          {lstmPrediction && (
            <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-purple-500">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">LSTM 예측 (Enhanced)</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">데이터 소스:</span>
                  <span className="font-semibold text-purple-600">Excel Files</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">모델 타입:</span>
                  <span className="font-semibold text-purple-600">{lstmPrediction.model_type}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">2025년 예상 환적량:</span>
                  <span className="font-semibold text-purple-600">
                    {formatNumber(safeNumber(lstmPrediction.yearly_totals?.[2025], 0))} TEU
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">2026년 예상 환적량:</span>
                  <span className="font-semibold text-purple-600">
                    {formatNumber(safeNumber(lstmPrediction.yearly_totals?.[2026], 0))} TEU
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">성장률 (2026 vs 2024):</span>
                  <span className={`font-semibold ${safeNumber(lstmPrediction.growth_rates?.['2026_vs_2024']) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {safeNumber(lstmPrediction.growth_rates?.['2026_vs_2024']) > 0 ? '+' : ''}{safeToFixed(safeNumber(lstmPrediction.growth_rates?.['2026_vs_2024']), 2)}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );

  // EXOG 경로 예측 탭 (기존과 유사하지만 Enhanced 표시 추가)
  const runExogForecast = async () => {
    setExogLoading(true);
    try {
      const form = new FormData();
      
      if (macroFiles.length > 0) {
        form.append('macro_csv', macroFiles[0]);
      }
      
      form.append('horizon', String(horizon));
      form.append('log', 'true');
      form.append('features_json', JSON.stringify(selectedFeatures));
      form.append('use_yoy', String(useYoY));
      form.append('include_lag1', String(lag1));
      form.append('future_exog', JSON.stringify(exogPath));
      form.append('use_lstm', String(useLSTM));
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
      
      // 모델 타입에 따라 결과 저장
      if (data.mode === 'lstm_enhanced') {
        setLstmModelPredictions(data.predictions || []);
      } else if (data.mode === 'sarima_enhanced') {
        setSarimaPredictions(data.predictions || []);
      } else if (data.mode === 'sarimax_enhanced') {
        setSarimaxPredictions(data.predictions || []);
      }
      
    } catch (e) {
      console.error(e);
      alert('예측 호출 실패');
    }
    setExogLoading(false);
  };

  const renderExogTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          시계열 모델 예측 (SARIMA / SARIMAX / LSTM Enhanced)
        </h3>
        <div className="mb-4 text-sm text-gray-600 bg-blue-50 p-3 rounded">
          실제 Excel 파일 데이터를 기반으로 한 고급 시계열 분석 모델입니다.
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* 좌: 설정 */}
          <div>
            <div className="mb-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={useLSTM}
                  onChange={e => setUseLSTM(e.target.checked)}
                  className="mr-2"
                />
                <Brain className="w-4 h-4 mr-1" />
                LSTM 모델 사용 (Enhanced)
              </label>
            </div>
            
            <div className="mb-2">csv, xlsx (선택)</div>
            <input
              type="file"
              multiple
              accept=".csv,.xlsx,.xls"
              onChange={e => setMacroFiles(Array.from(e.target.files || []))}
              disabled={useLSTM}
            />
            {macroFiles?.length > 0 && !useLSTM && (
              <ul className="mt-2 text-sm text-gray-600 list-disc ml-5">
                {macroFiles.map(f => <li key={f.name}>{f.name}</li>)}
              </ul>
            )}
            
            {!useLSTM && (
              <>
                <div className="mt-2">
                  <label className="block mb-1">시트명/번호(선택)</label>
                  <input
                    className="border px-2 py-1"
                    value={sheetName}
                    onChange={e => setSheetName(e.target.value)}
                    placeholder="예: 0 또는 Sheet1"
                  />
                </div>

                {/* Features 선택 체크박스 */}
                <div className="mt-4">
                  <label className="block mb-2 font-medium">사용할 특성 변수 선택:</label>
                  <div className="space-y-2">
                    {availableFeatures.map(feature => (
                      <label key={feature} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={selectedFeatures.includes(feature)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedFeatures(prev => [...prev, feature]);
                            } else {
                              setSelectedFeatures(prev => prev.filter(f => f !== feature));
                            }
                          }}
                          className="mr-2"
                        />
                        {featureLabels[feature] || feature}
                      </label>
                    ))}
                  </div>
                </div>
                
                <div className="mt-2">
                  <label className="mr-2">
                    <input type="checkbox" checked={useYoY} onChange={e => setUseYoY(e.target.checked)} />
                    전년동월 대비(%)
                  </label>
                </div>
                <div className="mt-1">
                  <label className="mr-2">
                    <input type="checkbox" checked={lag1} onChange={e => setLag1(e.target.checked)} />
                    1개월 지연 포함
                  </label>
                </div>
              </>
            )}
            
            <div className="mt-4">
              <label className="mr-2">시작(YYYY-MM)</label>
              <input className="border px-2 py-1" value={startYM} onChange={e => setStartYM(e.target.value)} />
            </div>
            <div className="mt-2">
              <label className="mr-2">horizon(개월)</label>
              <input type="number" className="border px-2 py-1" value={horizon} onChange={e => setHorizon(Number(e.target.value) || 12)} />
            </div>

            <button
              onClick={runExogForecast}
              disabled={exogLoading}
              className="mt-4 w-full px-6 py-3 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-50 flex items-center justify-center"
            >
              {exogLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  {useLSTM ? 'Excel 데이터 기반 LSTM 학습 중...' : 'Excel 데이터 기반 SARIMA 분석 중...'}
                </>
              ) : useLSTM ? (
                <>
                  <Brain className="w-4 h-4 mr-2" />
                  LSTM 예측 실행 (Enhanced)
                </>
              ) : (
                'SARIMA(X) 예측 실행 (Enhanced)'
              )}
            </button>
            
            {/* 진행 상황 표시 */}
            {exogLoading && (
              <div className="mt-4 p-4 bg-emerald-50 border border-emerald-200 rounded">
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-emerald-600 mr-3"></div>
                  <div className="text-emerald-800">
                    {useLSTM ? 'Excel 데이터 기반 LSTM 모델 학습 중... (5-10초 소요)' : 'Excel 데이터 기반 SARIMA 차수 탐색 중... (3-5초 소요)'}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* 우: 월별 테이블 (LSTM 사용시 비활성화) */}
          {!useLSTM && (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr>
                    <th className="border px-2 py-1">월</th>
                    {selectedFeatures.map(f => <th key={f} className="border px-2 py-1">{featureLabels[f] || f}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {months.map(m => (
                    <tr key={m}>
                      <td className="border px-2 py-1">{m}</td>
                      {selectedFeatures.map(f => (
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
            </div>
          )}
        </div>
      </div>

      {/* 예측 결과 차트 */}
      {(sarimaPredictions.length > 0 || sarimaxPredictions.length > 0 || lstmModelPredictions.length > 0) && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">시계열 모델 예측 결과 (Enhanced)</h3>
          <div className="mb-4 text-sm text-gray-600 bg-green-50 p-3 rounded">
            실제 Excel 데이터를 기반으로 한 시계열 모델 예측 결과입니다.
          </div>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={exogChartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip formatter={(v) => [v?.toLocaleString(), '예측']} />
              <Legend />
              {sarimaPredictions.length > 0 && (
                <Line type="monotone" dataKey="sarima" name="SARIMA (Enhanced)" stroke="#10b981" strokeWidth={2} />
              )}
              {sarimaxPredictions.length > 0 && (
                <Line type="monotone" dataKey="sarimax" name="SARIMAX (Enhanced)" stroke="#f59e0b" strokeWidth={2} />
              )}
              {lstmModelPredictions.length > 0 && (
                <Line type="monotone" dataKey="lstm" name="LSTM (Enhanced)" stroke="#9333ea" strokeWidth={3} />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );

  // 나머지 탭들 (민감도, 시나리오, 과거실적)은 기존과 동일하지만 Enhanced 표시 추가
  const renderSensitivityTab = () => (
    <div className="space-y-6">
      {sensitivity && sensitivity.sensitivity_analysis && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">민감도 분석 (Enhanced)</h3>
          <div className="mb-4 text-sm text-gray-600 bg-blue-50 p-3 rounded">
            실제 Excel 데이터를 기반으로 계산된 경제지표별 민감도 분석 결과입니다.
          </div>
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
        </div>
      )}
    </div>
  );

  const renderScenariosTab = () => (
    <div className="space-y-6">
      <div className="bg-blue-50 p-4 rounded-lg mb-6">
        <h4 className="font-semibold text-blue-800 mb-2">Enhanced 시나리오 분석</h4>
        <p className="text-blue-700 text-sm">실제 Excel 파일 데이터를 기반으로 한 시나리오별 예측 비교</p>
      </div>
      
      {scenarios && scenarios.scenarios && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.entries(scenarios.scenarios).map(([scenarioName, scenarioData]) => (
            <div key={scenarioName} className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 capitalize">
                {scenarioName === 'baseline' ? '기준 시나리오' :
                 scenarioName === 'optimistic' ? '낙관적 시나리오' :
                 scenarioName === 'pessimistic' ? '비관적 시나리오' :
                 scenarioName === 'lstm_enhanced' ? 'LSTM 예측 (Enhanced)' : scenarioName}
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

  const renderHistoricalTab = () => (
    <div className="space-y-6">
      <div className="bg-green-50 p-4 rounded-lg mb-6">
        <h4 className="font-semibold text-green-800 mb-2">실제 데이터 기반 과거 실적</h4>
        <p className="text-green-700 text-sm">Excel 파일에서 추출한 부산항 환적 실제 데이터</p>
      </div>
      
      {historical && (
        <>
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">과거 실적 데이터 (Enhanced)</h3>
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
            <h3 className="text-lg font-semibold text-gray-800 mb-4">연도별 통계 (Enhanced)</h3>
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

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Ship className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">부산항 환적량 예측 시스템 (Enhanced)</h1>
                <p className="text-gray-600">실제 Excel 데이터 기반 AI 다중 모델 예측 시스템</p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500">Enhanced v4.0</div>
              <div className="text-xs text-gray-400 mt-1">
                Excel 데이터 기반 고도화
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-2 mb-6 overflow-x-auto">
          <TabButton id="forecast" label="환적량 예측" icon={TrendingUp} active={activeTab === 'forecast'} onClick={setActiveTab} />
          <TabButton id="sensitivity" label="민감도 분석" icon={BarChart3} active={activeTab === 'sensitivity'} onClick={setActiveTab} />
          <TabButton id="scenarios" label="시나리오 분석" icon={Globe} active={activeTab === 'scenarios'} onClick={setActiveTab} />
          <TabButton id="historical" label="과거 실적" icon={Info} active={activeTab === 'historical'} onClick={setActiveTab} />
          <TabButton id="exog" label="시계열 모델" icon={Settings} active={activeTab === 'exog'} onClick={setActiveTab} />
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

export default EnhancedBusanPortForecast;