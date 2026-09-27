/* ════════════════════════════════════════════════════════════════
   ArthShastraAI — Frontend Logic
   ════════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initTabs();
    initFileUpload();
    initChat();
    initMobileToggle();
});

/* ── Globals ──────────────────────────────────────────────────── */
let currentReturnsData = null; // stored after analysis completes
let currentPeriodsPerYear = null;

/* ── Navigation ───────────────────────────────────────────────── */
function initNavigation() {
    document.querySelectorAll('.nav-item[data-page]').forEach(item => {
        item.addEventListener('click', e => {
            e.preventDefault();
            const page = item.dataset.page;
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            const target = document.getElementById('page-' + page);
            if (target) target.classList.add('active');
            // close sidebar on mobile
            document.querySelector('.sidebar')?.classList.remove('open');
        });
    });
}

/* ── Tabs ─────────────────────────────────────────────────────── */
function initTabs() {
    document.querySelectorAll('.tabs').forEach(tabBar => {
        tabBar.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const group = tabBar.dataset.group;
                tabBar.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                document.querySelectorAll(`.tab-content[data-group="${group}"]`).forEach(tc => tc.classList.remove('active'));
                const target = document.getElementById(tab.dataset.target);
                if (target) target.classList.add('active');
            });
        });
    });
}

/* ── File Upload & Analysis ───────────────────────────────────── */
function initFileUpload() {
    const fileInput = document.getElementById('csv-file');
    const dropZone = document.getElementById('drop-zone');
    if (!fileInput || !dropZone) return;

    dropZone.addEventListener('dragover', e => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--accent-blue)';
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = '';
    });
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.style.borderColor = '';
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect(fileInput.files[0]);
        }
    });
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) handleFileSelect(fileInput.files[0]);
    });
}

function handleFileSelect(file) {
    if (!file) return;
    document.getElementById('file-name').textContent = `📄 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    document.getElementById('upload-info').classList.remove('hidden');
}

function runAnalysis() {
    const fileInput = document.getElementById('csv-file');
    if (!fileInput.files.length) {
        showAlert('Please select a CSV file first.', 'warning');
        return;
    }
    const dateCol = document.getElementById('date-col').value.trim();
    const fillMethod = document.getElementById('fill-method').value;
    const returnType = document.getElementById('return-type').value;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('date_col', dateCol);
    formData.append('fill_method', fillMethod);
    formData.append('return_type', returnType);

    showSpinner('Running portfolio analysis...');

    fetch('/api/analyze', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            hideSpinner();
            if (data.error) {
                showAlert(data.error, 'error');
                return;
            }
            renderResults(data);
        })
        .catch(err => {
            hideSpinner();
            showAlert('Analysis failed: ' + err.message, 'error');
        });
}

function renderResults(data) {
    document.getElementById('results-section').classList.remove('hidden');

    // Store for later use
    currentReturnsData = data;

    // KPIs
    document.getElementById('kpi-best-return').textContent = data.kpis.best_return;
    document.getElementById('kpi-avg-vol').textContent = data.kpis.avg_vol;
    document.getElementById('kpi-best-sharpe').textContent = data.kpis.best_sharpe;
    document.getElementById('kpi-max-dd').textContent = data.kpis.max_drawdown;

    // Summary table
    renderTable('summary-table', data.summary_stats);

    // Insights
    const insightsEl = document.getElementById('insights-list');
    insightsEl.innerHTML = data.insights.map(i => `<div class="insight-item">${i}</div>`).join('');

    // Charts
    renderChartImages(data.charts);

    // Efficient Frontier
    if (data.efficient_frontier) {
        renderEfficientFrontier(data.efficient_frontier);
    }

    // Stress test
    if (data.stress_test) {
        renderStressTest(data.stress_test);
    }

    // Market Regimes
    if (data.regimes) {
        renderRegimes(data.regimes);
    }

    // Scroll to results
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
}

function renderTable(containerId, tableData) {
    if (!tableData || !tableData.columns || !tableData.rows) return;
    const container = document.getElementById(containerId);
    let html = '<div class="table-wrapper"><table class="data-table"><thead><tr>';
    html += '<th>Asset</th>';
    tableData.columns.forEach(c => { html += `<th>${c}</th>`; });
    html += '</tr></thead><tbody>';
    tableData.rows.forEach(row => {
        html += '<tr>';
        html += `<td><strong>${row.asset}</strong></td>`;
        tableData.columns.forEach(c => {
            html += `<td>${row[c] ?? ''}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table></div>';
    container.innerHTML = html;
}

function renderChartImages(charts) {
    if (!charts) return;
    const mapping = {
        cumulative_returns: 'chart-cumulative',
        risk_return: 'chart-risk-return',
        correlation: 'chart-correlation',
        rolling_vol: 'chart-rolling-vol'
    };
    for (const [key, elementId] of Object.entries(mapping)) {
        const el = document.getElementById(elementId);
        if (el && charts[key]) {
            el.innerHTML = `<img src="data:image/png;base64,${charts[key]}" alt="${key}">`;
        }
    }
}

function renderEfficientFrontier(ef) {
    // Chart
    const chartEl = document.getElementById('chart-ef');
    if (chartEl && ef.chart) {
        chartEl.innerHTML = `<img src="data:image/png;base64,${ef.chart}" alt="Efficient Frontier">`;
    }

    // Portfolio compositions
    const compEl = document.getElementById('ef-compositions');
    if (!compEl || !ef.portfolios) return;

    let html = '';
    const icons = { 'Maximum Sharpe Ratio': '🟢', 'Global Minimum Volatility': '🔵', 'Equal Weight': '🟡' };
    for (const [name, pf] of Object.entries(ef.portfolios)) {
        html += `<div class="card">`;
        html += `<div class="card-header">${icons[name] || '⚪'} ${name}</div>`;
        html += `<div class="metrics-grid">`;
        html += `<div class="metric-card"><div class="metric-card-title">Expected Return</div><div class="metric-card-value">${pf.return}</div></div>`;
        html += `<div class="metric-card"><div class="metric-card-title">Volatility</div><div class="metric-card-value">${pf.volatility}</div></div>`;
        html += `<div class="metric-card"><div class="metric-card-title">Sharpe Ratio</div><div class="metric-card-value">${pf.sharpe}</div></div>`;
        html += `</div>`;
        if (pf.weights) {
            html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>Asset</th><th>Allocation</th></tr></thead><tbody>';
            pf.weights.forEach(w => {
                html += `<tr><td>${w.asset}</td><td>${w.allocation}</td></tr>`;
            });
            html += '</tbody></table></div>';
        }
        html += `</div>`;
    }
    compEl.innerHTML = html;
}

function renderStressTest(stress) {
    // Scenario cards
    const cardsEl = document.getElementById('stress-cards');
    if (!cardsEl) return;

    const icons = { '2008 Financial Crisis': '💥', 'COVID-19 Crash': '🦠', 'Rising Interest Rates': '📈', 'Normal': '📊' };
    let html = '<div class="metrics-grid">';
    for (const [scenario, metrics] of Object.entries(stress.scenarios)) {
        if (scenario === 'Normal') continue;
        html += `<div class="scenario-card">`;
        html += `<div class="scenario-icon">${icons[scenario] || '⚠️'}</div>`;
        html += `<div class="scenario-name">${scenario}</div>`;
        html += `<div class="scenario-metric"><div class="scenario-metric-label">Stressed Return</div><div class="scenario-metric-value">${metrics['Annualized Return']}</div></div>`;
        html += `<div class="scenario-metric"><div class="scenario-metric-label">VaR (5%)</div><div class="scenario-metric-value">${metrics['VaR(5%)']}</div></div>`;
        html += `<div class="scenario-metric"><div class="scenario-metric-label">CVaR (5%)</div><div class="scenario-metric-value">${metrics['CVaR(5%)']}</div></div>`;
        html += `<div class="scenario-metric"><div class="scenario-metric-label">Max Drawdown</div><div class="scenario-metric-value">${metrics['Max Drawdown']}</div></div>`;
        html += `</div>`;
    }
    html += '</div>';
    cardsEl.innerHTML = html;

    // Comparison table
    const tableEl = document.getElementById('stress-table');
    if (tableEl && stress.scenarios) {
        let thtml = '<div class="table-wrapper"><table class="data-table"><thead><tr><th>Scenario</th><th>Ann. Return</th><th>Ann. Vol</th><th>VaR (5%)</th><th>CVaR (5%)</th><th>Max Drawdown</th></tr></thead><tbody>';
        for (const [scenario, m] of Object.entries(stress.scenarios)) {
            thtml += `<tr><td><strong>${scenario}</strong></td>`;
            thtml += `<td>${m['Annualized Return']}</td>`;
            thtml += `<td>${m['Annualized Vol']}</td>`;
            thtml += `<td>${m['VaR(5%)']}</td>`;
            thtml += `<td>${m['CVaR(5%)']}</td>`;
            thtml += `<td>${m['Max Drawdown']}</td></tr>`;
        }
        thtml += '</tbody></table></div>';
        tableEl.innerHTML = thtml;
    }

    // Stress chart
    const stressChartEl = document.getElementById('chart-stress');
    if (stressChartEl && stress.chart) {
        stressChartEl.innerHTML = `<img src="data:image/png;base64,${stress.chart}" alt="Stress Test">`;
    }
}

function renderRegimes(regimes) {
    // Regime pills
    const pillsEl = document.getElementById('regime-pills');
    if (pillsEl && regimes.distribution) {
        const icons = { 'Bull / Low Vol': '🟢', 'Bull / High Vol': '🟡', 'Bear / Low Vol': '🟠', 'Bear / High Vol': '🔴' };
        let html = '<div class="regime-row">';
        for (const [name, pct] of Object.entries(regimes.distribution)) {
            html += `<div class="regime-pill">`;
            html += `<div class="regime-pill-icon">${icons[name] || '⚪'}</div>`;
            html += `<div class="regime-pill-value">${pct}</div>`;
            html += `<div class="regime-pill-name">${name}</div>`;
            html += `</div>`;
        }
        html += '</div>';
        pillsEl.innerHTML = html;
    }

    // Regime performance table
    const tableEl = document.getElementById('regime-table');
    if (tableEl && regimes.performance) {
        renderTable('regime-table', regimes.performance);
    }

    // Regime chart
    const chartEl = document.getElementById('chart-regimes');
    if (chartEl && regimes.chart) {
        chartEl.innerHTML = `<img src="data:image/png;base64,${regimes.chart}" alt="Market Regimes">`;
    }
}

/* ── AI Analysis ──────────────────────────────────────────────── */
function runAIAnalysis() {
    if (!currentReturnsData) {
        showAlert('Please run the portfolio analysis first.', 'warning');
        return;
    }

    const aiContainer = document.getElementById('ai-result');
    aiContainer.innerHTML = '<div class="alert alert-info">🤖 ArthShastraAI is generating your analysis...</div>';

    fetch('/api/ai-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: currentReturnsData.session_id })
    })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                aiContainer.innerHTML = `<div class="alert alert-error">❌ ${data.error}</div>`;
            } else {
                aiContainer.innerHTML = `<div class="ai-response">${data.analysis}</div>`;
            }
        })
        .catch(err => {
            aiContainer.innerHTML = `<div class="alert alert-error">Failed: ${err.message}</div>`;
        });
}

/* ── Chat (Q&A) ───────────────────────────────────────────────── */
function initChat() {
    const input = document.getElementById('chat-input');
    if (!input) return;
    input.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
}

function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const question = input.value.trim();
    if (!question) return;

    const container = document.getElementById('chat-messages');
    // Add user message
    container.innerHTML += `
        <div class="chat-message user">
            <div class="chat-avatar">👤</div>
            <div class="chat-bubble">${escapeHtml(question)}</div>
        </div>`;
    input.value = '';
    container.scrollTop = container.scrollHeight;

    // Add typing indicator
    const typingId = 'typing-' + Date.now();
    container.innerHTML += `
        <div class="chat-message assistant" id="${typingId}">
            <div class="chat-avatar">🤖</div>
            <div class="chat-bubble">Thinking...</div>
        </div>`;
    container.scrollTop = container.scrollHeight;

    fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    })
        .then(r => r.json())
        .then(data => {
            const typingEl = document.getElementById(typingId);
            if (typingEl) {
                if (data.error) {
                    typingEl.querySelector('.chat-bubble').innerHTML = `<span style="color:var(--accent-red)">Error: ${data.error}</span>`;
                } else {
                    typingEl.querySelector('.chat-bubble').innerHTML = data.answer;
                }
            }
            container.scrollTop = container.scrollHeight;
        })
        .catch(err => {
            const typingEl = document.getElementById(typingId);
            if (typingEl) {
                typingEl.querySelector('.chat-bubble').innerHTML = `<span style="color:var(--accent-red)">Failed: ${err.message}</span>`;
            }
        });
}

function clearChat() {
    const container = document.getElementById('chat-messages');
    if (container) container.innerHTML = '';
    fetch('/api/chat/clear', { method: 'POST' });
}

/* ── API Key ──────────────────────────────────────────────────── */
function saveApiKey() {
    const key = document.getElementById('api-key-input').value.trim();
    if (!key) {
        showAlert('Please enter an API key.', 'warning');
        return;
    }

    fetch('/api/set-key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: key })
    })
        .then(r => r.json())
        .then(data => {
            const badge = document.getElementById('ai-status');
            if (data.success) {
                badge.className = 'status-badge success';
                badge.textContent = '✅ AI enabled';
                showAlert('API key saved. AI features are now enabled.', 'success');
            } else {
                badge.className = 'status-badge warning';
                badge.textContent = '⚠️ AI disabled';
                showAlert(data.error || 'Failed to set API key.', 'error');
            }
        });
}

/* ── Mobile Toggle ────────────────────────────────────────────── */
function initMobileToggle() {
    const toggle = document.getElementById('mobile-toggle');
    const sidebar = document.querySelector('.sidebar');
    if (toggle && sidebar) {
        toggle.addEventListener('click', () => sidebar.classList.toggle('open'));
    }
}

/* ── Helpers ───────────────────────────────────────────────────── */
function showSpinner(text) {
    const overlay = document.getElementById('spinner-overlay');
    const spinText = document.getElementById('spinner-text');
    if (spinText) spinText.textContent = text || 'Processing...';
    if (overlay) overlay.classList.add('active');
}

function hideSpinner() {
    const overlay = document.getElementById('spinner-overlay');
    if (overlay) overlay.classList.remove('active');
}

function showAlert(message, type) {
    const container = document.getElementById('alert-area');
    if (!container) return;
    const id = 'alert-' + Date.now();
    container.innerHTML = `<div class="alert alert-${type}" id="${id}">${message}</div>`;
    setTimeout(() => {
        const el = document.getElementById(id);
        if (el) el.remove();
    }, 6000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function downloadSummaryCSV() {
    if (!currentReturnsData || !currentReturnsData.session_id) {
        showAlert('Run an analysis first.', 'warning');
        return;
    }
    window.open(`/api/download-summary?session_id=${currentReturnsData.session_id}`, '_blank');
}
