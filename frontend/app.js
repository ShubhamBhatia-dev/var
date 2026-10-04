/**
 * app.js
 * ======
 * Market Risk Engine — Dashboard Application
 *
 * Data flow:
 *   Browser → fetch() → FastAPI /api/* → JSON → Charts / Tables
 *
 * All values displayed are fetched from the Python risk engine.
 * No metrics are hard-coded in this file.
 *
 * Chart library: Chart.js v4 (loaded via CDN)
 */

'use strict';

// ─────────────────────────────────────────────────────────────
// CONFIGURATION
// ─────────────────────────────────────────────────────────────
const API_BASE = '/api';
const RETRY_DELAY_MS = 3000;
const MAX_RETRIES = 5;

// Chart.js global defaults for dark theme
Chart.defaults.color = '#334155';
Chart.defaults.borderColor = '#cbd5e1';
Chart.defaults.font.family = "'Inter', sans-serif";

// ─────────────────────────────────────────────────────────────
// APPLICATION STATE
// ─────────────────────────────────────────────────────────────
const State = {
  summary: null,
  pnl: null,
  risk: null,
  backtest: null,
  violations: null,
  volatility: null,
  distribution: null,
  portfolio: null,
  scenarios: null,
  config: null,
  charts: {},
  tableData: [],
  tableFilter: 'all',
  tableSort: { col: 'date', dir: 'asc' },
  tablePage: 1,
  tablePageSize: 25,
};

// ─────────────────────────────────────────────────────────────
// UTILITY FUNCTIONS
// ─────────────────────────────────────────────────────────────

/**
 * Format a dollar amount with commas and optional decimals.
 */
function fmtUSD(val, decimals = 0) {
  if (val == null || isNaN(val)) return '—';
  const abs = Math.abs(val);
  const formatted = abs.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  return (val < 0 ? '-' : '') + '$' + formatted;
}

/**
 * Format as percentage.
 */
function fmtPct(val, decimals = 2) {
  if (val == null || isNaN(val)) return '—';
  return (val * 100).toFixed(decimals) + '%';
}

/**
 * Format as annualized percentage (already annualized input).
 */
function fmtAnnVol(val) {
  if (val == null || isNaN(val)) return '—';
  return (val * 100).toFixed(2) + '%';
}

/**
 * Format a plain number with fixed decimals.
 */
function fmtNum(val, decimals = 3) {
  if (val == null || isNaN(val)) return '—';
  return Number(val).toFixed(decimals);
}

/**
 * Fetch JSON from the API with retry logic.
 */
async function fetchAPI(endpoint, retries = MAX_RETRIES) {
  for (let i = 0; i < retries; i++) {
    try {
      const resp = await fetch(API_BASE + endpoint);
      if (resp.status === 503) {
        // Engine still initializing
        setLoadingStatus(`Engine initializing... (attempt ${i + 1}/${retries})`);
        await new Promise(r => setTimeout(r, RETRY_DELAY_MS));
        continue;
      }
      if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
      return await resp.json();
    } catch (err) {
      if (i === retries - 1) throw err;
      setLoadingStatus(`Retrying... (${err.message})`);
      await new Promise(r => setTimeout(r, RETRY_DELAY_MS));
    }
  }
  throw new Error('Max retries exceeded');
}

function setLoadingStatus(msg) {
  const el = document.getElementById('loading-status');
  if (el) el.textContent = msg;
}

function showApp() {
  document.getElementById('loading-screen').style.display = 'none';
  const app = document.getElementById('app');
  app.classList.add('visible');
}

// ─────────────────────────────────────────────────────────────
// INITIALIZATION
// ─────────────────────────────────────────────────────────────

async function init() {
  try {
    setLoadingStatus('Fetching summary data...');
    State.summary = await fetchAPI('/summary');

    setLoadingStatus('Loading P&L time series...');
    State.pnl = await fetchAPI('/pnl');

    setLoadingStatus('Computing distribution...');
    State.distribution = await fetchAPI('/distribution');

    setLoadingStatus('Loading backtesting results...');
    State.backtest = await fetchAPI('/backtest');

    setLoadingStatus('Fetching volatility data...');
    State.volatility = await fetchAPI('/volatility');

    setLoadingStatus('Loading portfolio...');
    State.portfolio = await fetchAPI('/portfolio');

    setLoadingStatus('Fetching scenarios...');
    State.scenarios = await fetchAPI('/scenarios');

    setLoadingStatus('Fetching risk metrics...');
    State.risk = await fetchAPI('/risk');

    setLoadingStatus('Rendering dashboard...');

    renderHeader();
    renderKPICards();
    renderOverviewTestResults();
    renderRiskSummaryTable();
    renderResidualRisk();
    renderOverviewPnLChart();
    renderMainPnLChart();
    renderViolationChart();
    renderDistributionChart();
    renderDistributionStats();
    renderTenDayComparison();
    renderVolatilityChart();
    renderPortfolioDonut();
    renderPortfolioDetailTable();
    renderScenarios();
    buildTableData();
    renderTable();

    showApp();
    setupTabs();

  } catch (err) {
    setLoadingStatus(`Error: ${err.message}. Is the Python server running?`);
    console.error('Dashboard initialization failed:', err);
  }
}

// ─────────────────────────────────────────────────────────────
// HEADER
// ─────────────────────────────────────────────────────────────

function renderHeader() {
  const s = State.summary;
  document.getElementById('hdr-date').textContent = s.as_of_date || '—';
  document.getElementById('hdr-confidence').textContent =
    s.confidence_level ? (s.confidence_level * 100).toFixed(0) + '%' : '99%';
  document.getElementById('hdr-window').textContent =
    s.historical_window ? s.historical_window + 'D' : '500D';
  document.getElementById('data-status-text').textContent =
    'Data: ' + (s.data_status || 'Loaded');
}

// ─────────────────────────────────────────────────────────────
// KPI CARDS
// ─────────────────────────────────────────────────────────────

function renderKPICards() {
  const s = State.summary;

  document.getElementById('kpi-port-val').textContent = fmtUSD(s.portfolio_value);
  document.getElementById('kpi-var-1d').textContent = fmtUSD(s.var_1d_99);
  document.getElementById('kpi-es-1d').textContent = fmtUSD(s.es_1d_99);
  document.getElementById('kpi-var-10d').textContent = fmtUSD(s.var_10d_99);
  document.getElementById('kpi-es-10d').textContent = fmtUSD(s.es_10d_99);
  document.getElementById('kpi-violations').textContent = s.violations ?? '—';
  document.getElementById('kpi-violations-sub').textContent =
    `of ${s.observations ?? '—'} observations`;

  const rateEl = document.getElementById('kpi-rate');
  const rateCard = document.getElementById('kpi-rate-card');
  rateEl.textContent = fmtPct(s.violation_rate);

  // Color the exception rate card based on severity
  const expected = s.expected_violation_rate || 0.01;
  const actual = s.violation_rate || 0;
  if (actual > expected * 2) {
    rateCard.classList.add('danger');
    rateCard.style.setProperty('--kpi-accent', 'var(--red)');
    rateEl.style.color = 'var(--red)';
  } else if (actual > expected * 1.5) {
    rateCard.classList.add('warning');
    rateCard.style.setProperty('--kpi-accent', 'var(--amber)');
    rateEl.style.color = 'var(--amber)';
  } else {
    rateCard.classList.add('success');
    rateCard.style.setProperty('--kpi-accent', 'var(--green)');
    rateEl.style.color = 'var(--green)';
  }

  document.getElementById('pnl-obs-badge').textContent =
    `${State.pnl?.total_records ?? '—'} obs`;
}

// ─────────────────────────────────────────────────────────────
// RISK SUMMARY TABLE
// ─────────────────────────────────────────────────────────────

function renderRiskSummaryTable() {
  const s = State.summary;
  const r = State.risk;
  const rows = [
    ['Portfolio Notional', fmtUSD(s.portfolio_value)],
    ['1D VaR (99%)', fmtUSD(s.var_1d_99)],
    ['1D ES (99%)', fmtUSD(s.es_1d_99)],
    ['10D VaR (99%) [Overlapping Scenarios]', fmtUSD(s.var_10d_99)],
    ['10D VaR (Sqrt-of-Time Approx.)', fmtUSD(s.var_10d_sqrt_approx) + ' ⓘ'],
    ['10D ES (99%)', fmtUSD(s.es_10d_99)],
    ['VaR as % of Portfolio', fmtPct((s.var_1d_99 || 0) / (s.portfolio_value || 1))],
    ['ES/VaR Ratio', fmtNum(r?.distribution_summary?.es_to_var_ratio)],
    ['Observations (Backtest)', s.observations ?? '—'],
    ['VaR Exceptions', s.violations ?? '—'],
    ['Exception Rate', fmtPct(s.violation_rate)],
    ['Expected Rate (99% VaR)', fmtPct(s.expected_violation_rate)],
    ['As of Date', s.as_of_date],
  ];

  const table = document.getElementById('risk-summary-table');
  table.innerHTML = rows.map(([label, value]) => `
    <tr>
      <td>${label}</td>
      <td>${value}</td>
    </tr>
  `).join('');
}

// ─────────────────────────────────────────────────────────────
// RESIDUAL RISK ADD-ON
// ─────────────────────────────────────────────────────────────

function renderResidualRisk() {
  const rr = State.risk?.residual_risk;
  if (!rr) return;

  const metrics = [
    ['Base VaR (1D)', fmtUSD(State.summary?.var_1d_99)],
    ['Add-On', fmtUSD(rr.addon_usd)],
    ['Adjusted Estimate', fmtUSD((State.summary?.var_1d_99 || 0) + (rr.addon_usd || 0))],
  ];

  const container = document.getElementById('residual-metrics');
  container.innerHTML = metrics.map(([label, val]) => `
    <div class="residual-metric">
      <div class="residual-metric-label">${label}</div>
      <div class="residual-metric-value">${val}</div>
    </div>
  `).join('');

  // Add regime badge
  const regime = rr.volatility_regime || 'NORMAL';
  const regimeHtml = `
    <div class="flex-between mt-8">
      <span class="text-muted text-xs">Volatility Regime</span>
      <span class="test-result-badge ${regime === 'ELEVATED' ? 'fail' : 'pass'}">${regime}</span>
    </div>
    <div class="text-muted text-xs mt-8">
      Current Vol: ${fmtAnnVol(rr.current_vol_20d_annualized)} ·
      Threshold (80th pct): ${fmtAnnVol(rr.vol_threshold_annualized)} ·
      Multiplier k: ${rr.addon_multiplier}
    </div>
  `;
  const card = document.getElementById('residual-risk-section');
  if (card) card.insertAdjacentHTML('beforeend', regimeHtml);
}

// ─────────────────────────────────────────────────────────────
// TEST RESULT RENDERING (shared)
// ─────────────────────────────────────────────────────────────

function renderTestCard(title, result, subtitle = '') {
  const status = (result?.test_result || 'INCONCLUSIVE').toUpperCase();
  const badgeClass = status === 'PASS' ? 'pass' : status === 'FAIL' ? 'fail' : 'inconclusive';
  const pValue = result?.p_value ?? result?.p_value_independence ?? result?.p_value_conditional_coverage;
  const lr = result?.lr_statistic ?? result?.lr_independence ?? result?.lr_conditional_coverage;

  return `
    <div class="test-card">
      <div class="test-card-title">${title}</div>
      <div class="test-result-badge ${badgeClass}">${status}</div>
      <div class="test-detail">
        ${lr != null ? `<div>LR Statistic: <strong>${fmtNum(lr)}</strong></div>` : ''}
        ${pValue != null ? `<div>p-value: <strong>${fmtNum(pValue)}</strong></div>` : ''}
        ${subtitle ? `<div style="margin-top:4px; opacity:0.8">${subtitle}</div>` : ''}
      </div>
    </div>
  `;
}

function renderOverviewTestResults() {
  const bt = State.backtest;
  const container = document.getElementById('overview-test-results');
  if (!container || !bt) return;

  container.innerHTML = [
    renderTestCard('Kupiec POF', bt.kupiec_pof, 'Correct frequency?'),
    renderTestCard('Independence', bt.christoffersen_independence, 'Non-clustered?'),
    renderTestCard('Cond. Coverage', bt.conditional_coverage, 'Joint test (χ² 2df)'),
  ].join('');
}

function renderBacktestResults() {
  const bt = State.backtest;
  if (!bt) return;

  // Stats grid
  const summary = bt.summary;
  const statsGrid = document.getElementById('backtest-stats-grid');
  if (statsGrid) {
    const stats = [
      ['Total Observations', summary.n_observations],
      ['Expected Violations', fmtNum(summary.expected_violations, 1)],
      ['Actual Violations', summary.n_violations],
      ['Violation Rate', fmtPct(summary.violation_rate)],
      ['Expected Rate', fmtPct(summary.expected_violation_rate)],
      ['Basel Zone (ref)', summary.basel_zone_reference || '—'],
    ];
    statsGrid.innerHTML = stats.map(([label, val]) => `
      <div class="backtest-stat-card">
        <div class="backtest-stat-label">${label}</div>
        <div class="backtest-stat-value">${val}</div>
      </div>
    `).join('');
  }

  // Test results
  const testContainer = document.getElementById('backtest-test-results');
  if (testContainer) {
    testContainer.innerHTML = [
      renderTestCard(
        'Kupiec POF Test',
        bt.kupiec_pof,
        'H₀: violation rate = 1% (Kupiec 1995)'
      ),
      renderTestCard(
        'Christoffersen Independence',
        bt.christoffersen_independence,
        'H₀: violations are i.i.d. (Christoffersen 1998)'
      ),
      renderTestCard(
        'Conditional Coverage',
        bt.conditional_coverage,
        'Joint: frequency AND independence (2 df)'
      ),
    ].join('');
  }
}

// ─────────────────────────────────────────────────────────────
// OVERVIEW P&L CHART (mini version)
// ─────────────────────────────────────────────────────────────

function renderOverviewPnLChart() {
  const records = State.pnl?.records || [];
  if (records.length === 0) return;

  // Use last 250 records for overview
  const recent = records.slice(-250);
  const labels = recent.map(r => r.date);
  const pnlData = recent.map(r => r.realized_pnl);
  const varData = recent.map(r => -r.var_1d);

  const violations = recent.map(r => r.violation === 1 ? r.realized_pnl : null);

  const ctx = document.getElementById('overview-pnl-chart')?.getContext('2d');
  if (!ctx) return;

  if (State.charts['overview-pnl']) State.charts['overview-pnl'].destroy();

  State.charts['overview-pnl'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Realized P&L',
          data: pnlData,
          borderColor: 'rgba(100, 160, 255, 0.7)',
          backgroundColor: 'transparent',
          borderWidth: 1,
          pointRadius: 0,
          tension: 0,
        },
        {
          label: '-VaR Threshold',
          data: varData,
          borderColor: 'rgba(239, 68, 68, 0.5)',
          borderDash: [4, 3],
          backgroundColor: 'transparent',
          borderWidth: 1,
          pointRadius: 0,
          tension: 0,
        },
        {
          label: 'Violations',
          data: violations,
          type: 'scatter',
          backgroundColor: 'rgba(239, 68, 68, 0.85)',
          pointRadius: 4,
          pointStyle: 'triangle',
        },
      ],
    },
    options: buildChartOptions({ hideGrid: true, compact: true }),
  });
}

// ─────────────────────────────────────────────────────────────
// MAIN P&L CHART
// ─────────────────────────────────────────────────────────────

function renderMainPnLChart() {
  const records = State.pnl?.records || [];
  if (records.length === 0) return;

  const labels = records.map(r => r.date);
  const pnlData = records.map(r => r.realized_pnl);
  const varThreshold = records.map(r => -r.var_1d); // negative = loss side
  const violations = records.map(r => r.violation === 1 ? r.realized_pnl : null);

  const ctx = document.getElementById('main-pnl-chart')?.getContext('2d');
  if (!ctx) return;

  if (State.charts['main-pnl']) State.charts['main-pnl'].destroy();

  State.charts['main-pnl'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Realized P&L',
          data: pnlData,
          borderColor: '#60a5fa',
          backgroundColor: 'rgba(96, 165, 250, 0.06)',
          borderWidth: 1.2,
          pointRadius: 0,
          fill: true,
          tension: 0,
          order: 2,
        },
        {
          label: '−VaR 99% (Loss Threshold)',
          data: varThreshold,
          borderColor: 'rgba(239, 68, 68, 0.7)',
          borderDash: [5, 4],
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0,
          order: 1,
        },
        {
          label: 'VaR Breach',
          data: violations,
          type: 'scatter',
          backgroundColor: '#ef4444',
          borderColor: '#fca5a5',
          borderWidth: 1,
          pointRadius: 5,
          pointStyle: 'triangle',
          order: 0,
        },
      ],
    },
    options: {
      ...buildChartOptions({}),
      plugins: {
        ...buildChartOptions({}).plugins,
        tooltip: {
          mode: 'index',
          intersect: false,
          backgroundColor: 'rgba(255,255,255,0.95)',
          titleColor: '#0f172a',
          bodyColor: '#334155',
          borderColor: '#cbd5e1',
          borderWidth: 1,
          titleFont: { family: "'JetBrains Mono'", size: 11 },
          bodyFont: { family: "'JetBrains Mono'", size: 11 },
          callbacks: {
            label(ctx) {
              const val = ctx.parsed.y;
              if (val == null) return null;
              const prefix = ctx.dataset.label;
              const formatted = fmtUSD(val);
              return `${prefix}: ${formatted}`;
            },
            afterBody(items) {
              const rec = records[items[0]?.dataIndex];
              if (rec?.violation === 1) {
                return [`⚠ VaR BREACH on ${rec.date}`, `Loss exceeds VaR by ${fmtUSD(-rec.realized_pnl - rec.var_1d)}`];
              }
              return [];
            },
          },
        },
      },
    },
  });
}

// ─────────────────────────────────────────────────────────────
// VIOLATION CHART
// ─────────────────────────────────────────────────────────────

function renderViolationChart() {
  const records = State.pnl?.records || [];
  if (records.length === 0) return;

  const labels = records.map(r => r.date);
  const normalPnL = records.map(r => r.violation === 0 ? r.realized_pnl : null);
  const violationPnL = records.map(r => r.violation === 1 ? r.realized_pnl : null);

  const ctx = document.getElementById('violation-chart')?.getContext('2d');
  if (!ctx) return;

  if (State.charts['violation']) State.charts['violation'].destroy();

  State.charts['violation'] = new Chart(ctx, {
    type: 'scatter',
    data: {
      labels,
      datasets: [
        {
          label: 'Normal Days',
          data: records.map((r, i) => r.violation === 0 ? { x: i, y: r.realized_pnl } : null).filter(Boolean),
          backgroundColor: 'rgba(96, 165, 250, 0.5)',
          pointRadius: 3,
        },
        {
          label: 'VaR Breaches',
          data: records.map((r, i) => r.violation === 1 ? { x: i, y: r.realized_pnl } : null).filter(Boolean),
          backgroundColor: '#ef4444',
          borderColor: '#fca5a5',
          borderWidth: 1,
          pointRadius: 7,
          pointStyle: 'triangle',
        },
      ],
    },
    options: {
      responsive: true,
      animation: { duration: 600 },
      plugins: {
        legend: { display: true, position: 'top', labels: { font: { size: 11 } } },
        tooltip: {
          backgroundColor: 'rgba(255,255,255,0.95)',
          titleColor: '#0f172a',
          bodyColor: '#334155',
          borderColor: '#cbd5e1',
          borderWidth: 1,
          callbacks: {
            label(ctx) {
              const rec = records[ctx.parsed.x];
              if (!rec) return '';
              return [
                `Date: ${rec.date}`,
                `P&L: ${fmtUSD(rec.realized_pnl)}`,
                `VaR: ${fmtUSD(rec.var_1d)}`,
                rec.violation === 1 ? `⚠ BREACH: ${fmtUSD(-rec.realized_pnl - rec.var_1d)} excess` : 'No breach',
              ];
            },
          },
        },
      },
      scales: {
        x: { display: false },
        y: {
          grid: { color: '#cbd5e1' },
          ticks: {
            callback: v => fmtUSD(v, 0),
            font: { size: 10, family: "'JetBrains Mono'" },
          },
        },
      },
    },
  });
}

// ─────────────────────────────────────────────────────────────
// DISTRIBUTION CHART
// ─────────────────────────────────────────────────────────────

function renderDistributionChart() {
  const dist = State.distribution;
  if (!dist) return;

  const bins = dist.histogram_bins;
  const counts = dist.histogram_counts;
  const varLine = -dist.var_99;
  const esLine = -dist.es_99;

  // Build bar colors: red for tail (< -VaR), normal for rest
  const barColors = bins.slice(0, -1).map(bin => {
    const midpoint = bin + (bins[1] - bins[0]) / 2;
    return midpoint < varLine ? 'rgba(239, 68, 68, 0.65)' : 'rgba(203, 213, 225, 0.8)';
  });

  const binMidpoints = bins.slice(0, -1).map((b, i) => {
    return ((b + bins[i + 1]) / 2).toFixed(0);
  });

  const ctx = document.getElementById('dist-chart')?.getContext('2d');
  if (!ctx) return;

  if (State.charts['dist']) State.charts['dist'].destroy();

  State.charts['dist'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: binMidpoints,
      datasets: [
        {
          label: 'P&L Frequency',
          data: counts,
          backgroundColor: barColors,
          borderColor: 'transparent',
          borderWidth: 0,
          barPercentage: 1.0,
          categoryPercentage: 1.0,
        },
      ],
    },
    options: {
      responsive: true,
      animation: { duration: 800 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title(items) {
              const mid = parseFloat(items[0].label);
              return `P&L ≈ ${fmtUSD(mid)}`;
            },
            label(ctx) {
              return `Frequency: ${ctx.parsed.y} scenarios`;
            },
          },
          backgroundColor: 'rgba(255,255,255,0.95)',
          titleColor: '#0f172a',
          bodyColor: '#334155',
          borderColor: '#cbd5e1',
          borderWidth: 1,
        },
        annotation: undefined,
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: {
            maxTicksLimit: 10,
            callback: (_, i) => {
              const label = binMidpoints[i];
              if (!label) return '';
              const val = parseFloat(label);
              return val >= 1e6 || val <= -1e6
                ? (val / 1e6).toFixed(1) + 'M'
                : (val / 1e3).toFixed(0) + 'K';
            },
            font: { size: 9, family: "'JetBrains Mono'" },
          },
        },
        y: {
          grid: { color: '#cbd5e1' },
          ticks: { font: { size: 9 } },
        },
      },
    },
    plugins: [
      {
        id: 'var_es_lines',
        afterDraw(chart) {
          const { ctx: c, scales: { x, y } } = chart;

          // Find pixel positions of VaR and ES bin indices
          const drawVLine = (xVal, color, label) => {
            // Find the bin index closest to xVal
            let closestIdx = 0;
            let minDist = Infinity;
            binMidpoints.forEach((mid, i) => {
              const dist = Math.abs(parseFloat(mid) - xVal);
              if (dist < minDist) { minDist = dist; closestIdx = i; }
            });

            const xPixel = x.getPixelForTick(closestIdx);
            if (xPixel == null) return;

            c.save();
            c.beginPath();
            c.moveTo(xPixel, y.top);
            c.lineTo(xPixel, y.bottom);
            c.strokeStyle = color;
            c.lineWidth = 2;
            c.setLineDash([5, 4]);
            c.stroke();

            c.fillStyle = color;
            c.font = 'bold 10px JetBrains Mono';
            c.textAlign = 'center';
            c.fillText(label, xPixel, y.top + 12);
            c.restore();
          };

          drawVLine(varLine, '#00d4ff', `VaR ${fmtUSD(dist.var_99, 0)}`);
          drawVLine(esLine, '#f59e0b', `ES ${fmtUSD(dist.es_99, 0)}`);
        },
      },
    ],
  });

  document.getElementById('dist-scenarios-badge').textContent =
    `${dist.n_scenarios} scenarios`;
}

// ─────────────────────────────────────────────────────────────
// DISTRIBUTION STATS TABLE
// ─────────────────────────────────────────────────────────────

function renderDistributionStats() {
  const ds = State.risk?.distribution_summary;
  if (!ds) return;

  const rows = [
    ['Scenarios', ds.n_scenarios],
    ['Mean P&L', fmtUSD(ds.mean_pnl)],
    ['Std Dev P&L', fmtUSD(ds.std_pnl)],
    ['Min (Worst Scenario)', fmtUSD(ds.min_pnl)],
    ['Max (Best Scenario)', fmtUSD(ds.max_pnl)],
    ['Skewness', fmtNum(ds.skewness)],
    ['Excess Kurtosis', fmtNum(ds.kurtosis)],
    ['1st Percentile', fmtUSD(ds.pct_1)],
    ['5th Percentile', fmtUSD(ds.pct_5)],
    ['95th Percentile', fmtUSD(ds.pct_95)],
    ['99th Percentile', fmtUSD(ds.pct_99)],
    ['ES/VaR Ratio', fmtNum(ds.es_to_var_ratio)],
  ];

  const table = document.getElementById('dist-stats-table');
  if (!table) return;
  table.innerHTML = rows.map(([l, v]) => `<tr><td>${l}</td><td>${v}</td></tr>`).join('');
}

// ─────────────────────────────────────────────────────────────
// 10-DAY COMPARISON
// ─────────────────────────────────────────────────────────────

function renderTenDayComparison() {
  const s = State.summary;
  const rows = [
    ['10D VaR (Overlapping Scenarios) [PRIMARY]', fmtUSD(s.var_10d_99)],
    ['10D ES (Overlapping Scenarios) [PRIMARY]', fmtUSD(s.es_10d_99)],
    ['10D VaR √10 Approx. [SECONDARY — i.i.d. assumption]', fmtUSD(s.var_10d_sqrt_approx)],
    ['Difference (Scenarios vs. √10)', fmtUSD((s.var_10d_99 || 0) - (s.var_10d_sqrt_approx || 0))],
    ['1D VaR (basis)', fmtUSD(s.var_1d_99)],
    ['√10 × 1D VaR', fmtUSD((s.var_1d_99 || 0) * Math.sqrt(10))],
  ];

  const table = document.getElementById('tenday-comparison-table');
  if (!table) return;
  table.innerHTML = rows.map(([l, v]) => `<tr><td>${l}</td><td>${v}</td></tr>`).join('');
}

// ─────────────────────────────────────────────────────────────
// VOLATILITY CHART
// ─────────────────────────────────────────────────────────────

function renderVolatilityChart() {
  const records = State.volatility?.records || [];
  if (records.length === 0) return;

  const labels = records.map(r => r.date);
  const vol20 = records.map(r => r.vol_20d != null ? r.vol_20d * 100 : null);
  const vol60 = records.map(r => r.vol_60d != null ? r.vol_60d * 100 : null);
  const threshold = (State.volatility?.high_vol_threshold || 0) * 100;

  const ctx = document.getElementById('vol-chart')?.getContext('2d');
  if (!ctx) return;

  if (State.charts['vol']) State.charts['vol'].destroy();

  // Build high-vol background regions
  State.charts['vol'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: '20D Volatility (Ann.)',
          data: vol20,
          borderColor: '#00d4ff',
          backgroundColor: 'rgba(0, 212, 255, 0.05)',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
          fill: false,
        },
        {
          label: '60D Volatility (Ann.)',
          data: vol60,
          borderColor: '#a78bfa',
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          borderDash: [4, 3],
          pointRadius: 0,
          tension: 0.3,
          fill: false,
        },
        {
          label: `Elevated Regime Threshold (${fmtNum(threshold, 1)}%)`,
          data: labels.map(() => threshold),
          borderColor: 'rgba(245, 158, 11, 0.4)',
          borderDash: [8, 4],
          backgroundColor: 'transparent',
          borderWidth: 1,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      animation: { duration: 800 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: true, position: 'top', labels: { font: { size: 11 } } },
        tooltip: {
          backgroundColor: 'rgba(255,255,255,0.95)',
          titleColor: '#0f172a',
          bodyColor: '#334155',
          borderColor: '#cbd5e1',
          borderWidth: 1,
          callbacks: {
            label(ctx) {
              const val = ctx.parsed.y;
              return `${ctx.dataset.label}: ${val != null ? val.toFixed(2) + '%' : '—'}`;
            },
          },
        },
      },
      scales: {
        x: {
          type: 'category',
          grid: { display: false },
          ticks: {
            maxTicksLimit: 12,
            font: { size: 10 },
            callback(val, i) {
              return labels[i]?.substring(0, 10) || '';
            },
          },
        },
        y: {
          grid: { color: '#cbd5e1' },
          ticks: {
            callback: v => v.toFixed(1) + '%',
            font: { size: 10, family: "'JetBrains Mono'" },
          },
          title: { display: true, text: 'Annualized Volatility (%)', color: '#4b5e7a', font: { size: 10 } },
        },
      },
    },
  });

  document.getElementById('current-vol-20d').textContent =
    fmtAnnVol(State.volatility?.current_vol_20d);
}

// ─────────────────────────────────────────────────────────────
// PORTFOLIO DONUT CHART
// ─────────────────────────────────────────────────────────────

function renderPortfolioDonut() {
  const assets = State.portfolio?.assets || [];
  if (assets.length === 0) return;

  const labels = assets.map(a => a.display_name);
  const values = assets.map(a => a.weight_pct);

  const colors = [
    '#00d4ff', '#3b82f6', '#8b5cf6',
    '#f59e0b', '#ef4444', '#10b981',
    '#fb923c', '#ec4899',
  ];

  const ctx = document.getElementById('donut-chart')?.getContext('2d');
  if (!ctx) return;

  if (State.charts['donut']) State.charts['donut'].destroy();

  State.charts['donut'] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors.slice(0, assets.length),
        borderColor: '#ffffff',
        borderWidth: 3,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      animation: { duration: 800 },
      plugins: {
        legend: {
          position: 'right',
          labels: {
            font: { size: 11 },
            color: '#334155',
            padding: 12,
            generateLabels(chart) {
              const data = chart.data;
              return data.labels.map((label, i) => ({
                text: `${label}  ${values[i].toFixed(1)}%`,
                fillStyle: colors[i],
                hidden: false,
                index: i,
              }));
            },
          },
        },
        tooltip: {
          backgroundColor: 'rgba(255,255,255,0.95)',
          titleColor: '#0f172a',
          bodyColor: '#334155',
          borderColor: '#cbd5e1',
          borderWidth: 1,
          callbacks: {
            label(ctx) {
              const asset = assets[ctx.dataIndex];
              return [
                `Weight: ${ctx.parsed.toFixed(1)}%`,
                `Value: ${fmtUSD(asset.position_value_usd)}`,
              ];
            },
          },
        },
      },
    },
  });
}

// ─────────────────────────────────────────────────────────────
// PORTFOLIO DETAIL TABLE
// ─────────────────────────────────────────────────────────────

function renderPortfolioDetailTable() {
  const assets = State.portfolio?.assets || [];
  const riskContribs = State.portfolio?.risk_contributions || [];

  const rcMap = {};
  riskContribs.forEach(rc => { rcMap[rc.asset] = rc; });

  const tbody = document.getElementById('portfolio-detail-body');
  if (!tbody) return;

  tbody.innerHTML = assets.map(asset => {
    const rc = rcMap[asset.asset] || {};
    const barWidth = (asset.weight * 100).toFixed(1);
    const standaloneVol = rc.standalone_vol_ann != null
      ? fmtAnnVol(rc.standalone_vol_ann) : '—';
    const riskContrib = rc.pct_risk_contribution != null
      ? fmtPct(rc.pct_risk_contribution) : '—';

    return `
      <tr>
        <td>
          <div style="font-weight:600; color: var(--text-primary)">${asset.display_name}</div>
          <div style="font-size:0.65rem; color: var(--text-muted)">${asset.asset}</div>
          <div style="margin-top: 4px; background: var(--border); height: 3px; border-radius: 2px; overflow: hidden;">
            <div class="portfolio-asset-bar" style="width: ${barWidth}%"></div>
          </div>
        </td>
        <td style="color: var(--text-muted)">${asset.asset_class}</td>
        <td class="mono">${fmtPct(asset.weight)}</td>
        <td class="mono">${fmtUSD(asset.position_value_usd)}</td>
        <td class="mono">${standaloneVol}</td>
        <td class="mono">${riskContrib}</td>
      </tr>
    `;
  }).join('');
}

// ─────────────────────────────────────────────────────────────
// SCENARIOS
// ─────────────────────────────────────────────────────────────

function renderScenarios() {
  const scenarios = State.scenarios?.scenarios || [];
  const container = document.getElementById('scenario-grid');
  if (!container) return;

  if (scenarios.length === 0) {
    container.innerHTML = '<div class="text-muted">No scenario data available.</div>';
    return;
  }

  container.innerHTML = scenarios.map(sc => {
    const pnl = sc.portfolio_pnl || 0;
    const isNeg = pnl < 0;

    // Get the first few asset shocks
    const shockKeys = Object.keys(sc).filter(k => k.endsWith('_shock_pct'));

    const shocksHTML = shockKeys.slice(0, 4).map(key => {
      const asset = key.replace('_shock_pct', '');
      const val = sc[key];
      const isNegShock = val < 0;
      return `
        <div class="scenario-shock-item">
          <span>${asset}</span>
          <span class="scenario-shock-val ${isNegShock ? 'neg' : 'pos'}">
            ${isNegShock ? '' : '+'}${fmtNum(val, 2)}%
          </span>
        </div>
      `;
    }).join('');

    return `
      <div class="scenario-card">
        <div class="scenario-name">${sc.scenario_name}</div>
        <div class="scenario-date">${sc.date}</div>
        <div class="scenario-pnl ${isNeg ? 'negative' : 'positive'}">
          ${fmtUSD(pnl)}
        </div>
        ${sc.cumulative_5d_pnl != null
          ? `<div style="font-size:0.68rem; color: var(--text-muted); margin-bottom: 8px;">
              5D Cumulative: ${fmtUSD(sc.cumulative_5d_pnl)}
            </div>`
          : ''}
        <div class="scenario-shocks">
          <div class="scenario-shock-item" style="color: var(--text-muted); font-size: 0.6rem; text-transform: uppercase; margin-bottom: 2px;">
            <span>Asset</span><span>1D Shock</span>
          </div>
          ${shocksHTML}
        </div>
      </div>
    `;
  }).join('');
}

// ─────────────────────────────────────────────────────────────
// DATA TABLE
// ─────────────────────────────────────────────────────────────

function buildTableData() {
  const pnlRecords = State.pnl?.records || [];
  const volMap = {};
  (State.volatility?.records || []).forEach(r => { volMap[r.date] = r; });

  State.tableData = pnlRecords.map(r => ({
    date: r.date,
    pnl: r.realized_pnl,
    var: r.var_1d,
    es: r.es_1d,
    vol: volMap[r.date]?.vol_20d ?? null,
    violation: r.violation,
  }));
}

function filterTable() {
  State.tablePage = 1;
  renderTable();
}

function setFilter(filter) {
  State.tableFilter = filter;
  document.getElementById('filter-all').classList.toggle('active', filter === 'all');
  document.getElementById('filter-violations').classList.toggle('active', filter === 'violations');
  State.tablePage = 1;
  renderTable();
}

function sortTable(col) {
  if (State.tableSort.col === col) {
    State.tableSort.dir = State.tableSort.dir === 'asc' ? 'desc' : 'asc';
  } else {
    State.tableSort.col = col;
    State.tableSort.dir = 'asc';
  }
  renderTable();
}

function renderTable() {
  const search = document.getElementById('table-search')?.value?.toLowerCase() || '';
  const { col, dir } = State.tableSort;

  // Filter
  let data = State.tableData.filter(r => {
    if (State.tableFilter === 'violations' && r.violation !== 1) return false;
    if (search && !r.date.includes(search)) return false;
    return true;
  });

  // Sort
  data.sort((a, b) => {
    let av = a[col] ?? '', bv = b[col] ?? '';
    if (typeof av === 'number' && typeof bv === 'number') {
      return dir === 'asc' ? av - bv : bv - av;
    }
    return dir === 'asc'
      ? String(av).localeCompare(String(bv))
      : String(bv).localeCompare(String(av));
  });

  // Paginate
  const total = data.length;
  const pages = Math.ceil(total / State.tablePageSize);
  const page = Math.min(State.tablePage, pages || 1);
  const start = (page - 1) * State.tablePageSize;
  const pageData = data.slice(start, start + State.tablePageSize);

  // Render rows
  const tbody = document.getElementById('table-body');
  if (!tbody) return;

  tbody.innerHTML = pageData.map(r => {
    const pnlClass = r.pnl > 0 ? 'pnl-positive' : 'pnl-negative';
    const isViolation = r.violation === 1;
    return `
      <tr class="${isViolation ? 'violation-row' : ''}">
        <td class="date-col">${r.date}</td>
        <td class="${pnlClass}">${fmtUSD(r.pnl)}</td>
        <td>${fmtUSD(r.var)}</td>
        <td>${fmtUSD(r.es)}</td>
        <td>${r.vol != null ? fmtAnnVol(r.vol) : '—'}</td>
        <td>
          <span class="violation-flag ${isViolation ? 'yes' : 'no'}">
            ${isViolation ? '✕' : '·'}
          </span>
        </td>
      </tr>
    `;
  }).join('');

  // Pagination info
  document.getElementById('pagination-info').textContent =
    `Showing ${start + 1}–${Math.min(start + State.tablePageSize, total)} of ${total} records`;

  // Pagination controls
  const paginationEl = document.getElementById('pagination-controls');
  const btns = [];
  btns.push(`<button class="page-btn" onclick="goPage(${page - 1})" ${page <= 1 ? 'disabled' : ''}>‹ Prev</button>`);

  const rangeStart = Math.max(1, page - 2);
  const rangeEnd = Math.min(pages, page + 2);
  for (let p = rangeStart; p <= rangeEnd; p++) {
    btns.push(`<button class="page-btn ${p === page ? 'active' : ''}" onclick="goPage(${p})">${p}</button>`);
  }

  btns.push(`<button class="page-btn" onclick="goPage(${page + 1})" ${page >= pages ? 'disabled' : ''}>Next ›</button>`);
  paginationEl.innerHTML = btns.join('');

  // Update sort indicators
  ['date', 'pnl', 'var', 'es', 'vol', 'violation'].forEach(c => {
    const th = document.getElementById(`th-${c}`);
    if (!th) return;
    th.classList.remove('sort-asc', 'sort-desc');
    if (col === c) th.classList.add(dir === 'asc' ? 'sort-asc' : 'sort-desc');
  });
}

function goPage(page) {
  State.tablePage = page;
  renderTable();
}

// ─────────────────────────────────────────────────────────────
// TAB NAVIGATION
// ─────────────────────────────────────────────────────────────

function setupTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  const panels = document.querySelectorAll('.tab-panel');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;

      tabs.forEach(t => t.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      document.getElementById(`tab-${target}`)?.classList.add('active');

      // Lazy-render charts on tab switch (for accurate canvas sizing)
      if (target === 'pnl-backtest') {
        renderMainPnLChart();
        renderViolationChart();
        renderBacktestResults();
      }
    });
  });
}

// ─────────────────────────────────────────────────────────────
// CHART OPTIONS BUILDER
// ─────────────────────────────────────────────────────────────

function buildChartOptions({ hideGrid = false, compact = false } = {}) {
  return {
    responsive: true,
    animation: { duration: 600 },
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        display: !compact,
        position: 'top',
        labels: { font: { size: 11 }, color: '#334155' },
      },
      tooltip: {
        backgroundColor: 'rgba(255,255,255,0.95)',
          titleColor: '#0f172a',
          bodyColor: '#334155',
        borderColor: '#cbd5e1',
        borderWidth: 1,
        titleFont: { family: "'JetBrains Mono'", size: 10 },
        bodyFont: { family: "'JetBrains Mono'", size: 10 },
      },
    },
    scales: {
      x: {
        type: 'category',
        grid: { display: !hideGrid, color: '#cbd5e1' },
        ticks: {
          maxTicksLimit: compact ? 6 : 12,
          font: { size: 9 },
          callback(val, i) {
            const labels = this.chart.data.labels;
            return labels[i]?.substring(0, 10) || '';
          },
        },
      },
      y: {
        grid: { color: hideGrid ? 'transparent' : '#cbd5e1' },
        ticks: {
          font: { size: 9, family: "'JetBrains Mono'" },
          callback(v) {
            const abs = Math.abs(v);
            if (abs >= 1e6) return (v / 1e6).toFixed(1) + 'M';
            if (abs >= 1e3) return (v / 1e3).toFixed(0) + 'K';
            return v.toFixed(0);
          },
        },
      },
    },
  };
}

// ─────────────────────────────────────────────────────────────
// EXPORT / DOWNLOAD
// ─────────────────────────────────────────────────────────────

function downloadCSV(type) {
  let url;
  if (type === 'pnl') url = API_BASE + '/export/pnl';
  else if (type === 'risk') url = API_BASE + '/export/risk_report';
  else return;

  const a = document.createElement('a');
  a.href = url;
  a.download = '';
  a.click();
}

// ─────────────────────────────────────────────────────────────
// ENTRY POINT
// ─────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', init);
