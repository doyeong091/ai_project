const API_BASE_URL = 'http://127.0.0.1:8000';

let forecastChart = null;
let predictTimeout = null;
let cachedOptions = null;
let cachedDefaultFeatureValues = null;
let cachedModelFeatureMap = {};

const CATEGORY_MAP = {
  price: { label: '현물 가격 지표', color: '#1A56DB' },
  market: { label: '글로벌 거시경제 지표', color: '#059669' },
  shock: { label: '지정학적 게이트키퍼 리스크 (Shock 촉발)', color: '#D42B2B' },
  conflict: { label: '지리적 분쟁 실물 지표 (ACLED)', color: '#7C3AED' },
  advanced: { label: '고급 모델 내부 Feature', color: '#111827' }
};

const OIL_ID_MAP = {
  Dubai: 'Dubai',
  WTI: 'Wti',
  Brent: 'Brent'
};

function isAdvancedMode() {
  const toggle = document.getElementById('advancedFeatureToggle');
  return !!(toggle && toggle.checked);
}

function getAdvancedModelType() {
  const select = document.getElementById('advancedModelTypeSelect');

  if (!select) {
    return 'default';
  }

  return select.value || 'default';
}

function normalizeOilName(oil) {
  return String(oil || '').trim().toLowerCase();
}

function formatNumber(value, key = '') {
  const num = Number(value);

  if (!Number.isFinite(num)) {
    return '0';
  }

  if (
    key === 'crude_inventory' ||
    key.includes('count') ||
    key.includes('fatalities')
  ) {
    return Math.round(num).toLocaleString();
  }

  return num.toFixed(2);
}

function isChangedFromDefault(key, value, step = 0.01) {
  if (!cachedDefaultFeatureValues) {
    return true;
  }

  const defaultValue = Number(cachedDefaultFeatureValues[key]);
  const currentValue = Number(value);
  const stepValue = Number(step);

  if (!Number.isFinite(defaultValue) || !Number.isFinite(currentValue)) {
    return true;
  }

  const tolerance = Number.isFinite(stepValue)
    ? Math.max(stepValue / 2, 1e-6)
    : 1e-6;

  return Math.abs(currentValue - defaultValue) > tolerance;
}

function getSliderRange(item, catKey) {
  const key = item.key;
  const defaultVal = Number(item.default_value ?? 0);

  let min = 0;
  let max = 100;
  let step = 0.01;

  if (catKey === 'price') {
    min = Math.max(0, Math.floor(defaultVal - 40));
    max = Math.ceil(defaultVal + 40);
    step = 0.01;
  } else if (catKey === 'market') {
    min = defaultVal === 0 ? 0 : Math.floor(defaultVal * 0.5);
    max = defaultVal === 0 ? 200 : Math.ceil(defaultVal * 1.5);
    step = key === 'crude_inventory' ? 10 : 0.01;
  } else if (catKey === 'shock' || catKey === 'conflict') {
    min = 0;
    max = Math.max(50, Math.ceil(defaultVal * 3));
    step = 1;
  } else {
    const absVal = Math.abs(defaultVal);

    if (absVal === 0) {
      min = -100;
      max = 100;
    } else {
      min = Math.floor(defaultVal - absVal * 2);
      max = Math.ceil(defaultVal + absVal * 2);

      if (min === max) {
        min -= 10;
        max += 10;
      }
    }

    step = key.includes('count') || key.includes('fatalities') ? 1 : 0.01;
  }

  if (min === max) {
    max = min + 1;
  }

  return { min, max, step };
}

function getOptionMetaMap() {
  const map = {};

  if (!cachedOptions || !cachedOptions.categories) {
    return map;
  }

  Object.values(cachedOptions.categories).forEach(items => {
    (items || []).forEach(item => {
      map[item.key] = item;
    });
  });

  return map;
}

async function loadTopbarDefaultPredictions() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/predict/default`);

    if (!response.ok) {
      return;
    }

    const data = await response.json();
    const oils = ['Dubai', 'WTI', 'Brent'];

    oils.forEach(oil => {
      const predData = data.predictions[oil];

      if (!predData) {
        return;
      }

      const htmlKey = OIL_ID_MAP[oil];
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

async function loadDefaultFeatureValues() {
  if (cachedDefaultFeatureValues) {
    return cachedDefaultFeatureValues;
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/default-feature-values`);

    if (!response.ok) {
      cachedDefaultFeatureValues = {};
      return cachedDefaultFeatureValues;
    }

    const data = await response.json();

    cachedDefaultFeatureValues = data.features || {};
    return cachedDefaultFeatureValues;
  } catch (e) {
    console.error('default feature values 로드 실패:', e);
    cachedDefaultFeatureValues = {};
    return cachedDefaultFeatureValues;
  }
}

async function getSupportedFeaturesForOil(oil, modelType = 'default') {
  const lowOil = normalizeOilName(oil);
  const cacheKey = `${lowOil}_${modelType}`;

  if (cachedModelFeatureMap[cacheKey]) {
    return cachedModelFeatureMap[cacheKey];
  }

  try {
    const response = await fetch(
      `${API_BASE_URL}/api/model-features/${lowOil}/${modelType}`
    );

    const data = response.ok ? await response.json() : { features: [] };
    const uniqueFeatures = Array.from(new Set(data.features || []));

    cachedModelFeatureMap[cacheKey] = uniqueFeatures;

    return uniqueFeatures;
  } catch (e) {
    console.error(`[에러] ${oil} ${modelType} 피처 목록 로드 중 통신 실패:`, e);
    cachedModelFeatureMap[cacheKey] = [];
    return [];
  }
}

function buildBasicOptionGroups() {
  if (!cachedOptions || !cachedOptions.categories) {
    return {};
  }

  return cachedOptions.categories;
}

async function buildAdvancedOptionGroups(oilType) {
  const defaults = await loadDefaultFeatureValues();
  const optionMetaMap = getOptionMetaMap();
  const modelType = getAdvancedModelType();
  const modelFeatures = await getSupportedFeaturesForOil(oilType, modelType);

  const items = modelFeatures.map(key => {
    const knownMeta = optionMetaMap[key];
    const defaultValue = defaults[key];

    if (knownMeta) {
      return {
        ...knownMeta,
        default_value: defaultValue ?? knownMeta.default_value ?? 0,
        category: knownMeta.category || 'advanced'
      };
    }

    return {
      key,
      label: key,
      description: `${modelType} 모델이 실제 사용하는 내부 feature입니다.`,
      default_value: defaultValue ?? 0,
      unit: null,
      category: 'advanced'
    };
  });

  return {
    advanced: items
  };
}

function renderSliderItem(item, catKey, groupBlock) {
  const defaultVal = Number(item.default_value ?? 0);
  const displayVal = formatNumber(defaultVal, item.key);
  const { min, max, step } = getSliderRange(item, catKey);
  const pct = ((defaultVal - min) / (max - min) * 100).toFixed(1);

  const row = document.createElement('div');
  row.style.marginBottom = '12px';

  row.innerHTML = `
    <div class="sim-header">
      <div class="sim-label" style="font-size:0.78rem;">${item.label}</div>
      <div class="sim-value" id="simVal_${item.key}" style="font-size:0.82rem;">
        ${displayVal}${item.unit ? ' ' + item.unit : ''}
      </div>
    </div>
    <input type="range" class="sim-slider feature-input-item"
           id="slider_${item.key}"
           data-key="${item.key}"
           data-unit="${item.unit || ''}"
           min="${min}"
           max="${max}"
           step="${step}"
           value="${defaultVal}"
           style="background: linear-gradient(to right, #1A56DB ${pct}%, #E8ECF2 ${pct}%)"
           oninput="window.handleSliderDrag(this, ${min}, ${max})">
    <div style="font-size:0.68rem; color:var(--text-muted); margin-top:2px;">
      ${item.description || ''}
    </div>
  `;

  groupBlock.appendChild(row);
}

async function renderDynamicSliders() {
  const container = document.getElementById('simulatorGrid');

  if (!container || !cachedOptions) {
    return;
  }

  const oilType = document.getElementById('targetOilSelect').value;
  const advancedMode = isAdvancedMode();
  const modelType = getAdvancedModelType();

  document.getElementById('kpiTargetName').textContent = oilType;
  document.getElementById('predictDynamicTitle').textContent =
    `${oilType} 시뮬레이션 예측가`;

  container.innerHTML = `
    <p style="font-size: 0.8rem; color: var(--text-muted); text-align: center;">
      ${advancedMode ? `${modelType} 모델 feature 로드 중...` : '대표 feature 로드 중...'}
    </p>
  `;

  const optionGroups = advancedMode
    ? await buildAdvancedOptionGroups(oilType)
    : buildBasicOptionGroups();

  container.innerHTML = '';

  Object.keys(optionGroups).forEach(catKey => {
    const items = optionGroups[catKey];

    if (!items || items.length === 0) {
      return;
    }

    const catMeta = CATEGORY_MAP[catKey] || {
      label: catKey,
      color: '#4A5568'
    };

    const sectionHeader = document.createElement('div');
    sectionHeader.innerHTML = `
      <div class="divider" style="margin: 1.5rem 0 0.75rem 0;"></div>
      <div style="font-size:0.75rem; font-weight:700; color:${catMeta.color}; margin-bottom:0.75rem; display:flex; align-items:center; gap:6px;">
        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${catMeta.color}"></span>
        ${catMeta.label}
        ${advancedMode ? `<span style="font-size:0.68rem; color:var(--text-muted); font-weight:400;">(${modelType}, ${items.length}개)</span>` : ''}
      </div>
    `;

    const groupBlock = document.createElement('div');

    items.forEach(item => {
      renderSliderItem(item, catKey, groupBlock);
    });

    container.appendChild(sectionHeader);
    container.appendChild(groupBlock);
  });
}

window.handleSliderDrag = function(slider, min, max) {
  const val = parseFloat(slider.value);
  const key = slider.getAttribute('data-key');
  const unit = slider.getAttribute('data-unit');

  const pct = ((val - min) / (max - min) * 100).toFixed(1);
  slider.style.background =
    `linear-gradient(to right, #1A56DB ${pct}%, #E8ECF2 ${pct}%)`;

  const displayVal = formatNumber(val, key);
  document.getElementById(`simVal_${key}`).textContent =
    displayVal + (unit ? ' ' + unit : '');

  triggerDebouncedSimulation();
};

async function onTargetOilChange() {
  await renderDynamicSliders();
  triggerDebouncedSimulation();
}

window.onTargetOilChange = onTargetOilChange;

async function onAdvancedToggleChange() {
  const box = document.getElementById('advancedModelTypeBox');

  if (box) {
    box.style.display = isAdvancedMode() ? 'block' : 'none';
  }

  await renderDynamicSliders();
  triggerDebouncedSimulation();
}

window.onAdvancedToggleChange = onAdvancedToggleChange;

async function onAdvancedModelTypeChange() {
  await renderDynamicSliders();
  triggerDebouncedSimulation();
}

window.onAdvancedModelTypeChange = onAdvancedModelTypeChange;

function triggerDebouncedSimulation() {
  if (predictTimeout) {
    clearTimeout(predictTimeout);
  }

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
    const key = input.getAttribute('data-key');
    const value = parseFloat(input.value);
    const step = parseFloat(input.getAttribute('step') || '0.01');

    if (isChangedFromDefault(key, value, step)) {
      selectedFeatures[key] = value;
    }
  });

  const payload = {
    oil_type: oilType,
    selected_features: selectedFeatures
  };

  if (isAdvancedMode()) {
    payload.model_type = getAdvancedModelType();
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/predict/simulation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`prediction failed: ${response.status}`);
    }

    const result = await response.json();

    renderPredictionUI(result);
    await updatePredictionPathChart(payload);
  } catch (e) {
    console.error(e);
  } finally {
    const mainBlock = document.getElementById('predictMainBlock');

    if (mainBlock) {
      mainBlock.classList.remove('loading-state');
    }
  }
}

async function updatePredictionPathChart(payload) {
  if (!forecastChart) {
    return;
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/predict/path`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`path prediction failed: ${response.status}`);
    }

    const data = await response.json();
    const path = data.path || [];

    if (path.length === 0) {
      return;
    }

    forecastChart.data.labels = path.map(point => point.label);
    forecastChart.data.datasets[0].data = path.map(point => point.predicted_price);
    forecastChart.update();
  } catch (e) {
    console.error('모델 기반 path 차트 업데이트 실패:', e);
  }
}

function renderPredictionUI(res) {
  const predPrice = res.predicted_price_10d;
  const currentPrice = res.current_price;
  const returnPct = res.predicted_return_pct;
  const isUp = returnPct >= 0;

  document.getElementById('predictNumber').textContent =
    '$' + predPrice.toFixed(2);

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
    modeTag.style.background =
      res.model_type === 'shock_aware' ? '#FFF1F1' : '#ECFDF5';
    modeTag.style.color =
      res.model_type === 'shock_aware' ? '#D42B2B' : '#059669';
  }

  renderAppliedFactors(
    res.applied_selected_features,
    res.ignored_selected_features
  );
}

function renderAppliedFactors(applied, ignored) {
  const container = document.getElementById('factorRows');

  if (!container) {
    return;
  }

  container.innerHTML = '';

  const appliedKeys = applied ? Object.keys(applied) : [];
  const ignoredKeys = ignored ? Object.keys(ignored) : [];

  if (appliedKeys.length === 0 && ignoredKeys.length === 0) {
    container.innerHTML =
      '<div style="font-size:0.78rem; color:var(--text-muted); text-align:center; padding:1rem;">변동된 입력 시뮬레이션 지표가 없습니다.</div>';
    return;
  }

  if (appliedKeys.length === 0) {
    container.innerHTML +=
      '<div style="font-size:0.75rem; color:var(--text-muted); padding:0.4rem 0;">현재 선택된 모델에 직접 반영된 입력값이 없습니다.</div>';
  }

  const maxVal = Math.max(
    ...appliedKeys.map(key => Math.abs(Number(applied[key]) || 0)),
    1
  );

  appliedKeys.forEach(key => {
    const val = applied[key];
    const numVal = Number(val) || 0;
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
      <div class="factor-name" style="font-size:0.75rem; font-family:monospace; color:var(--text-primary); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${key}</div>
      <div class="factor-bar-wrap" style="height:6px; background:#F4F6FA; border-radius:3px; overflow:hidden;">
        <div class="factor-bar" style="width:${pct}%; height:100%; background:${barColor} !important; border-radius:3px; transition:width 0.4s ease;"></div>
      </div>
      <div class="factor-score" style="font-family:var(--font-num); font-size:0.78rem; font-weight:700; text-align:right; color:${numVal > 0 ? '#1A56DB' : 'var(--text-muted)'};">
        ${Number.isFinite(numVal) ? numVal.toFixed(2) : val}
      </div>
    `;

    container.appendChild(row);
  });

  if (ignoredKeys.length > 0) {
    const ignoredBox = document.createElement('div');
    ignoredBox.style.marginTop = '0.75rem';
    ignoredBox.style.padding = '0.65rem';
    ignoredBox.style.borderRadius = '8px';
    ignoredBox.style.background = '#F8F9FC';
    ignoredBox.style.border = '1px solid #E8ECF2';
    ignoredBox.style.fontSize = '0.72rem';
    ignoredBox.style.color = 'var(--text-muted)';
    ignoredBox.innerHTML = `
      <strong style="color:var(--text-secondary);">직접 반영되지 않은 입력값</strong>
      <div style="margin-top:4px; line-height:1.5;">
        ${ignoredKeys.slice(0, 12).join(', ')}
        ${ignoredKeys.length > 12 ? ` 외 ${ignoredKeys.length - 12}개` : ''}
      </div>
    `;

    container.appendChild(ignoredBox);
  }
}

function initForecastChart() {
  const ctx = document.getElementById('forecastChart');

  if (!ctx) {
    return;
  }

  forecastChart = new Chart(ctx.getContext('2d'), {
    type: 'line',
    data: {
      labels: ['현재', '2일 뒤', '5일 뒤', '8일 뒤', '10일 뒤(예측)'],
      datasets: [
        {
          label: '모델 기반 추정 경로',
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

  await loadDefaultFeatureValues();
  await loadTopbarDefaultPredictions();
  await renderDynamicSliders();

  triggerDebouncedSimulation();
}

window.addEventListener('DOMContentLoaded', initApp);