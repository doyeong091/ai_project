const API_BASE_URL = window.location.origin;

let forecastChart = null;
let predictTimeout = null;
let cachedOptions = null;

const CATEGORY_MAP = {
  price: { label: '현물 가격 지표', color: '#1A56DB' },
  market: { label: '글로벌 거시경제 지표', color: '#059669' },
  shock: { label: '지정학적 게이트키퍼 리스크 (Shock 촉발)', color: '#D42B2B' },
  conflict: { label: '지리적 분쟁 실물 지표 (ACLED)', color: '#7C3AED' }
};

// [스텝 1] 최상단 대시보드 영역에 3대 유종 디폴트 예측치를 상시 고정 출력
async function loadTopbarDefaultPredictions() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/predict/default`);
    if (!response.ok) return;

    const data = await response.json();

    const oils = ['Dubai', 'WTI', 'Brent'];
    const idMap = { Dubai: 'Dubai', WTI: 'Wti', Brent: 'Brent' };

    oils.forEach(oil => {
      const predData = data.predictions[oil];
      if (!predData) return;

      const htmlKey = idMap[oil];
      const priceEl = document.getElementById(`kpi${htmlKey}Price`);
      const deltaEl = document.getElementById(`kpi${htmlKey}Delta`);

      if (priceEl && deltaEl) {
        priceEl.textContent = '$' + predData.predicted_price_10d.toFixed(2);

        const pct = predData.predicted_return_pct;
        const isUp = pct >= 0;

        deltaEl.textContent = `${isUp ? '▲ +' : '▼ '}${pct.toFixed(2)}%`;
        deltaEl.className = `kpi-sub ${isUp ? 'tag-up' : 'tag-dn'}`;
      }
    });
  } catch (e) {
    console.error('상단 고정 대시보드 동기화 실패:', e);
  }
}

// [스텝 2] 선택 유종의 default 및 shock_aware 모델 피처 허용 목록
async function getSupportedFeaturesForOil(oil) {
  return null;

  try {
    const lowOil = oil.toLowerCase();

    const [resDef, resShock] = await Promise.all([
      fetch(`${API_BASE_URL}/api/model-features/${lowOil}/default`),
      fetch(`${API_BASE_URL}/api/model-features/${lowOil}/shock_aware`)
    ]);

    const defData = resDef.ok ? await resDef.json() : { features: [] };
    const shockData = resShock.ok ? await resShock.json() : { features: [] };

    const combinedFeatures = [
      ...(defData.features || []),
      ...(shockData.features || [])
    ];

    if (combinedFeatures.length === 0) {
      console.warn(`[경고] ${oil} 모델의 피처 리스트가 비어있습니다.`);
    }

    return new Set(combinedFeatures.map(f => f.toLowerCase()));
  } catch (e) {
    console.error(`[에러] ${oil} 피처 목록 로드 중 통신 실패:`, e);
    return null;
  }
}

// [스텝 3] 기획에 맞춰 슬라이더 목록을 실시간으로 다시 그리기
async function renderDynamicSliders() {
  const container = document.getElementById('simulatorGrid');
  if (!container || !cachedOptions) return;

  const oilType = document.getElementById('targetOilSelect').value;

  document.getElementById('kpiTargetName').textContent = oilType;
  document.getElementById('predictDynamicTitle').textContent = `${oilType} 시뮬레이션 예측가`;

  const supportedFeatures = await getSupportedFeaturesForOil(oilType);

  container.innerHTML = '';

  Object.keys(cachedOptions.categories).forEach(catKey => {
    const items = cachedOptions.categories[catKey];

    if (!items || items.length === 0) return;

    let groupHasVisibleItems = false;
    const groupBlock = document.createElement('div');

    items.forEach(item => {
      if (supportedFeatures && !supportedFeatures.has(item.key.toLowerCase())) {
        return;
      }

      groupHasVisibleItems = true;

      const defaultVal = item.default_value ?? 0;

      const formattedVal =
        item.key === 'crude_inventory'
          ? Math.round(defaultVal).toLocaleString()
          : typeof defaultVal === 'number'
            ? defaultVal.toFixed(2)
            : defaultVal;

      let min = 0;
      let max = 100;
      let step = 0.01;

      if (catKey === 'price') {
        min = Math.max(0, Math.floor(defaultVal - 40));
        max = Math.ceil(defaultVal + 40);
      } else if (catKey === 'market') {
        min = defaultVal === 0 ? 0 : Math.floor(defaultVal * 0.5);
        max = defaultVal === 0 ? 200 : Math.ceil(defaultVal * 1.5);

        if (item.key === 'crude_inventory') {
          step = 10;
        }
      } else if (catKey === 'shock' || catKey === 'conflict') {
        min = 0;
        max = Math.max(50, defaultVal * 3);
        step = 1;
      }

      const pct = ((defaultVal - min) / (max - min) * 100).toFixed(1);

      const row = document.createElement('div');
      row.style.marginBottom = '12px';

      row.innerHTML = `
        <div class="sim-header">
          <div class="sim-label" style="font-size:0.78rem;">${item.label}</div>
          <div class="sim-value" id="simVal_${item.key}" style="font-size:0.82rem;">
            ${formattedVal}${item.unit ? ' ' + item.unit : ''}
          </div>
        </div>

        <input
          type="range"
          class="sim-slider feature-input-item"
          id="slider_${item.key}"
          data-key="${item.key}"
          data-unit="${item.unit || ''}"
          min="${min}"
          max="${max}"
          step="${step}"
          value="${defaultVal}"
          style="background: linear-gradient(to right, #1A56DB ${pct}%, #E8ECF2 ${pct}%)"
          oninput="window.handleSliderDrag(this, ${min}, ${max})"
        >

        <div style="font-size:0.68rem; color:var(--text-muted); margin-top:2px;">
          ${item.description || ''}
        </div>
      `;

      groupBlock.appendChild(row);
    });

    if (groupHasVisibleItems) {
      const catMeta = CATEGORY_MAP[catKey] || { label: catKey, color: '#4A5568' };

      const sectionHeader = document.createElement('div');
      sectionHeader.innerHTML = `
        <div class="divider" style="margin: 1.5rem 0 0.75rem 0;"></div>
        <div style="font-size:0.75rem; font-weight:700; color:${catMeta.color}; margin-bottom:0.75rem; display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${catMeta.color}"></span>
          ${catMeta.label}
        </div>
      `;

      container.appendChild(sectionHeader);
      container.appendChild(groupBlock);
    }
  });
}

window.handleSliderDrag = function (slider, min, max) {
  const val = parseFloat(slider.value);
  const key = slider.getAttribute('data-key');
  const unit = slider.getAttribute('data-unit');

  const pct = ((val - min) / (max - min) * 100).toFixed(1);
  slider.style.background = `linear-gradient(to right, #1A56DB ${pct}%, #E8ECF2 ${pct}%)`;

  const displayVal =
    key === 'crude_inventory'
      ? Math.round(val).toLocaleString()
      : val.toFixed(2);

  document.getElementById(`simVal_${key}`).textContent =
    displayVal + (unit ? ' ' + unit : '');

  triggerDebouncedSimulation();
};

async function onTargetOilChange() {
  await renderDynamicSliders();
  triggerDebouncedSimulation();
}

function triggerDebouncedSimulation() {
  if (predictTimeout) clearTimeout(predictTimeout);

  const mainBlock = document.getElementById('predictMainBlock');

  if (mainBlock) {
    mainBlock.classList.add('loading-state');
  }

  predictTimeout = setTimeout(() => {
    updateSimulationPrediction();
  }, 150);
}

async function updateSimulationPrediction() {
  const oilType = document.getElementById('targetOilSelect').value;
  const inputs = document.querySelectorAll('.feature-input-item');

  const selectedFeatures = {};

  inputs.forEach(input => {
    selectedFeatures[input.getAttribute('data-key')] = parseFloat(input.value);
  });

  const payload = {
    oil_type: oilType,
    selected_features: selectedFeatures
  };

  try {
    const response = await fetch(`${API_BASE_URL}/api/predict/simulation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error('simulation api failed');
    }

    const result = await response.json();

    renderPredictionUI(result);

    // 중요:
    // 숫자 카드와 그래프가 같은 시뮬레이션 입력값을 사용하도록
    // path API에도 동일한 payload를 다시 전달한다.
    await updatePredictionPathChart(payload, result);
  } catch (e) {
    console.error(e);
  } finally {
    const mainBlock = document.getElementById('predictMainBlock');

    if (mainBlock) {
      mainBlock.classList.remove('loading-state');
    }
  }
}

function renderPredictionUI(res) {
  const predPrice = res.predicted_price_10d;
  const currentPrice = res.current_price;
  const returnPct = res.predicted_return_pct;
  const isUp = returnPct >= 0;

  document.getElementById('predictNumber').textContent = '$' + predPrice.toFixed(2);
  document.getElementById('predictNumber').className =
    'predict-number ' + (isUp ? 'up' : 'dn');

  document.getElementById('predictDelta').className =
    'predict-delta ' + (isUp ? 'up' : 'dn');

  document.getElementById('deltaArrow').textContent = isUp ? '▲' : '▼';

  const priceDiff = predPrice - currentPrice;

  document.getElementById('deltaText').textContent =
    `${isUp ? '+' : ''}$${priceDiff.toFixed(2)} (${isUp ? '+' : ''}${returnPct.toFixed(2)}%)`;

  document.getElementById('predictModelName').textContent =
    `${res.model_name} (${res.model_type})`;

  const modeTag = document.getElementById('serverModeTag');

  if (modeTag) {
    modeTag.textContent = `🟢 MODE: ${res.model_type.toUpperCase()}`;
    modeTag.style.background = res.model_type === 'shock_aware' ? '#FFF1F1' : '#ECFDF5';
    modeTag.style.color = res.model_type === 'shock_aware' ? '#D42B2B' : '#059669';
  }

  renderAppliedFactors(res.applied_selected_features);
}

async function updatePredictionPathChart(payload, simulationResult) {
  if (!forecastChart) return;

  try {
    const response = await fetch(`${API_BASE_URL}/api/predict/path`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error('path api failed');
    }

    const data = await response.json();
    const path = data.path || [];

    if (path.length === 0) {
      throw new Error('empty prediction path');
    }

    forecastChart.data.labels = path.map(point => point.label);
    forecastChart.data.datasets[0].data = path.map(point =>
      Number(point.predicted_price)
    );
    forecastChart.data.datasets[0].label = `${data.oil_type} ${data.model_type} path`;
    forecastChart.update();
  } catch (error) {
    console.warn('prediction path chart update failed, using linear fallback:', error);

    if (!simulationResult) return;

    const currentPrice = Number(simulationResult.current_price);
    const predPrice = Number(simulationResult.predicted_price_10d);

    if (!Number.isFinite(currentPrice) || !Number.isFinite(predPrice)) return;

    const stepSize = (predPrice - currentPrice) / 4;

    forecastChart.data.labels = [
      '현재',
      '2일 뒤',
      '5일 뒤',
      '8일 뒤',
      '10일 뒤(예측)'
    ];

    forecastChart.data.datasets[0].data = [
      currentPrice,
      currentPrice + stepSize,
      currentPrice + stepSize * 2,
      currentPrice + stepSize * 3,
      predPrice
    ];

    forecastChart.data.datasets[0].label = 'linear fallback path';
    forecastChart.update();
  }
}

function renderAppliedFactors(applied) {
  const container = document.getElementById('factorRows');
  if (!container) return;

  container.innerHTML = '';

  const appKeys = applied ? Object.keys(applied) : [];

  if (appKeys.length === 0) {
    container.innerHTML =
      '<div style="font-size:0.78rem; color:var(--text-muted); text-align:center; padding:1rem;">변동된 입력 시뮬레이션 지표가 없습니다.</div>';
    return;
  }

  const maxVal = Math.max(
    ...appKeys.map(key => Math.abs(parseFloat(applied[key]) || 0)),
    1
  );

  appKeys.forEach(key => {
    const val = applied[key];
    const numVal = parseFloat(val) || 0;

    const pct = maxVal > 0
      ? (Math.abs(numVal) / maxVal * 100).toFixed(1)
      : 0;

    const barColor = numVal > 0 ? '#1A56DB' : '#E8ECF2';

    const row = document.createElement('div');
    row.className = 'factor-row';

    row.style.display = 'grid';
    row.style.gridTemplateColumns = '240px 1fr 65px';
    row.style.alignItems = 'center';
    row.style.gap = '1rem';
    row.style.marginBottom = '8px';

    row.innerHTML = `
      <div
        class="factor-name"
        style="font-size:0.75rem; font-family:monospace; color:var(--text-primary); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;"
      >
        ${key}
      </div>

      <div
        class="factor-bar-wrap"
        style="height:6px; background:#F4F6FA; border-radius:3px; overflow:hidden;"
      >
        <div
          class="factor-bar"
          style="width:${pct}%; height:100%; background:${barColor} !important; border-radius:3px; transition:width 0.4s ease;"
        ></div>
      </div>

      <div
        class="factor-score"
        style="font-family:var(--font-num); font-size:0.78rem; font-weight:700; text-align:right; color:${numVal > 0 ? '#1A56DB' : 'var(--text-muted)'};"
      >
        ${typeof val === 'number' ? val.toFixed(2) : val}
      </div>
    `;

    container.appendChild(row);
  });
}

function initForecastChart() {
  const ctx = document.getElementById('forecastChart');
  if (!ctx) return;

  forecastChart = new Chart(ctx.getContext('2d'), {
    type: 'line',
    data: {
      labels: ['현재', '2일 뒤', '5일 뒤', '8일 뒤', '10일 뒤(예측)'],
      datasets: [
        {
          label: '시뮬레이션 트렌드',
          data: [0, 0, 0, 0, 0],
          borderColor: '#1A56DB',
          backgroundColor: 'rgba(26, 86, 219, 0.04)',
          borderWidth: 2.5,
          tension: 0.25,
          fill: true,
          pointBackgroundColor: '#1A56DB',
          pointRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        y: {
          grid: { color: '#E8ECF2' }
        },
        x: {
          grid: { display: false }
        }
      }
    }
  });
}

async function initApp() {
  initForecastChart();

  try {
    const response = await fetch(`${API_BASE_URL}/api/simulation-options`);

    if (response.ok) {
      cachedOptions = await response.json();
    }
  } catch (e) {
    console.error(e);
  }

  await loadTopbarDefaultPredictions();
  await renderDynamicSliders();
  triggerDebouncedSimulation();
}

window.addEventListener('DOMContentLoaded', initApp);