"""
ArthShastraAI — Flask Application
Institutional-grade portfolio analytics powered by AI.
Exports `app` at module level for Vercel deployment.
"""

import os
import io
import json
import uuid
import base64
import traceback

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for serverless
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, jsonify, session, send_file

import finance_toolkit as ftk

# ══════════════════════════════════════════════════════════════════
# Flask App — the `app` variable Vercel looks for
# ══════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'arthshastra-secret-key-change-in-prod')

# ── In-memory session store (serverless = no filesystem persistence) ─
_sessions = {}   # session_id -> { returns_df, summary, ... }

# ── Default API key (prefer env var) ─────────────────────────────
DEFAULT_API_KEY = os.environ.get('GOOGLE_API_KEY', '')

# ── LLM helpers ──────────────────────────────────────────────────
def _get_llm(api_key: str, model: str = 'gemini-2.5-flash'):
    """Return a LangChain ChatGoogleGenerativeAI instance, or None."""
    if not api_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        os.environ['GOOGLE_API_KEY'] = api_key
        return ChatGoogleGenerativeAI(
            model=model, temperature=0.6, google_api_key=api_key
        )
    except Exception:
        return None


def _get_api_key():
    """Resolve API key: session > env."""
    return session.get('api_key', DEFAULT_API_KEY) or ''


# ══════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    api_key = _get_api_key()
    ai_enabled = bool(api_key)
    masked = ('•' * 8 + api_key[-4:]) if len(api_key) > 4 else ''
    return render_template('index.html', ai_enabled=ai_enabled, api_key_masked=masked)


@app.route('/api/set-key', methods=['POST'])
def set_api_key():
    data = request.get_json(silent=True) or {}
    key = data.get('api_key', '').strip()
    if not key:
        return jsonify({'success': False, 'error': 'No key provided'})
    session['api_key'] = key
    llm = _get_llm(key)
    if llm:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Could not initialise AI with this key'})


# ══════════════════════════════════════════════════════════════════
# Portfolio Analysis
# ══════════════════════════════════════════════════════════════════

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main portfolio analysis endpoint."""
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Read CSV
        df = pd.read_csv(file)
        if df.empty:
            return jsonify({'error': 'CSV file is empty'}), 400

        # Identify date column
        date_col = request.form.get('date_col', '').strip()
        if not date_col:
            # Auto-detect: first column that parses as datetime
            for col in df.columns:
                try:
                    pd.to_datetime(df[col])
                    date_col = col
                    break
                except Exception:
                    continue
        if not date_col:
            return jsonify({'error': 'Could not detect a date column. Please specify one.'}), 400

        # Parse dates
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            return jsonify({'error': f"Could not parse '{date_col}' as dates: {e}"}), 400

        df_sorted = df.set_index(date_col).sort_index()

        # Numeric columns only
        numeric_cols = [c for c in df_sorted.columns if pd.api.types.is_numeric_dtype(df_sorted[c])]
        if not numeric_cols:
            return jsonify({'error': 'No numeric columns found in CSV'}), 400

        prices_df = df_sorted[numeric_cols].copy()

        # Fill missing values
        fill_method = request.form.get('fill_method', 'ffill')
        if fill_method == 'ffill':
            prices_df = prices_df.ffill()
        elif fill_method == 'interpolate':
            prices_df = prices_df.interpolate()
        prices_df = prices_df.dropna()

        if prices_df.empty:
            return jsonify({'error': 'No data remaining after processing'}), 400

        # Calculate returns
        return_type = request.form.get('return_type', 'simple')
        if return_type == 'log':
            returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
        else:
            returns_df = prices_df.pct_change().dropna()

        if returns_df.empty:
            return jsonify({'error': 'Not enough data to compute returns'}), 400

        # Detect frequency
        periods_per_year = ftk.detect_frequency(returns_df)

        # Summary stats
        raw_summary = ftk.summary_stats(returns_df, riskfree_rate=0.03)
        er = ftk.annualize_rets(returns_df)
        cov = returns_df.cov() * periods_per_year
        insights = ftk.generate_portfolio_insights(returns_df, raw_summary)

        # Store in memory for AI analysis / download
        sid = str(uuid.uuid4())
        _sessions[sid] = {
            'returns_df': returns_df,
            'raw_summary': raw_summary,
            'er': er,
            'cov': cov,
            'insights': insights,
            'periods_per_year': periods_per_year,
        }

        # ── Build response ───────────────────────────────────────
        result = {
            'session_id': sid,
            'kpis': _build_kpis(raw_summary),
            'summary_stats': _build_summary_table(raw_summary),
            'insights': insights,
            'charts': _build_charts(returns_df, raw_summary, periods_per_year),
            'efficient_frontier': _build_ef(er, cov),
            'stress_test': _build_stress(returns_df),
            'regimes': _build_regimes(returns_df),
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}', 'traceback': traceback.format_exc()}), 500


# ── Response builders ────────────────────────────────────────────

def _build_kpis(summary):
    return {
        'best_return': f"{summary['Annualized Return'].max():.2%}",
        'avg_vol': f"{summary['Annualized Vol'].mean():.2%}",
        'best_sharpe': f"{summary['Sharpe Ratio'].max():.3f}",
        'max_drawdown': f"{summary['Max Drawdown'].min():.2%}",
    }


def _build_summary_table(summary):
    cols = list(summary.columns)
    rows = []
    for asset in summary.index:
        row = {'asset': str(asset)}
        for col in cols:
            val = summary.loc[asset, col]
            if col in ('Annualized Return', 'Annualized Vol', 'Parametric VaR (5%)', 'Historic CVaR (5%)', 'Max Drawdown'):
                row[col] = f"{val:.2%}"
            elif col == 'Sharpe Ratio':
                row[col] = f"{val:.3f}"
            else:
                row[col] = f"{val:.4f}"
        rows.append(row)
    return {'columns': cols, 'rows': rows}


def _fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#1a2332', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _build_charts(returns_df, summary, periods_per_year):
    charts = {}

    # Style
    plt.rcParams.update({
        'text.color': '#e2e8f0',
        'axes.labelcolor': '#94a3b8',
        'xtick.color': '#64748b',
        'ytick.color': '#64748b',
        'axes.edgecolor': '#2a3a50',
        'axes.facecolor': '#1a2332',
        'figure.facecolor': '#1a2332',
        'grid.color': '#2a3a50',
    })

    # 1. Cumulative returns
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    cum = (1 + returns_df).cumprod()
    for col in cum.columns:
        ax1.plot(cum.index, cum[col], label=col, linewidth=2)
    ax1.set_title('Cumulative Returns Over Time', fontsize=14, color='#fff')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    charts['cumulative_returns'] = _fig_to_base64(fig1)

    # 2. Risk-Return scatter
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    sc = ax2.scatter(
        summary['Annualized Vol'], summary['Annualized Return'],
        s=120, alpha=0.8, c=summary['Sharpe Ratio'], cmap='RdYlGn'
    )
    for i, asset in enumerate(summary.index):
        ax2.annotate(asset,
                     (summary.iloc[i]['Annualized Vol'], summary.iloc[i]['Annualized Return']),
                     xytext=(6, 6), textcoords='offset points', fontsize=9, color='#e2e8f0')
    ax2.set_xlabel('Annualized Volatility (Risk)')
    ax2.set_ylabel('Annualized Return')
    ax2.set_title('Risk-Return Profile (colour = Sharpe Ratio)', fontsize=13, color='#fff')
    plt.colorbar(sc, label='Sharpe Ratio')
    ax2.grid(True, alpha=0.3)
    charts['risk_return'] = _fig_to_base64(fig2)

    # 3. Correlation heatmap
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(returns_df.corr(), annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax3, fmt='.2f',
                annot_kws={'color': '#e2e8f0', 'fontsize': 9})
    ax3.set_title('Pearson Correlation of Returns', color='#fff', fontsize=13)
    charts['correlation'] = _fig_to_base64(fig3)

    # 4. Rolling volatility
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    window = min(max(len(returns_df) // 10, 10), 60)
    rolling_vol = returns_df.rolling(window=window).std() * np.sqrt(periods_per_year)
    for col in rolling_vol.columns:
        ax4.plot(rolling_vol.index, rolling_vol[col], label=col, alpha=0.85)
    ax4.set_title(f'Rolling {window}-Period Annualised Volatility', fontsize=13, color='#fff')
    ax4.set_ylabel('Volatility')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    charts['rolling_vol'] = _fig_to_base64(fig4)

    return charts


def _build_ef(er, cov, riskfree_rate=0.03):
    """Build efficient frontier chart + portfolio data."""
    try:
        w_msr = ftk.msr(riskfree_rate, er, cov)
        w_gmv = ftk.gmv(cov)
        w_ew = np.repeat(1 / len(er), len(er))

        fig, ax = plt.subplots(figsize=(11, 7))
        plt.rcParams.update({
            'axes.facecolor': '#1a2332', 'figure.facecolor': '#1a2332',
            'text.color': '#e2e8f0', 'axes.labelcolor': '#94a3b8',
            'xtick.color': '#64748b', 'ytick.color': '#64748b',
        })
        ftk.plot_ef(n_points=30, er=er, cov=cov, ax=ax,
                    show_cml=True, show_ew=True, show_gmv=True,
                    riskfree_rate=riskfree_rate)
        ax.set_title('Efficient Frontier with Optimal Portfolios', fontsize=15, color='#fff')
        ax.set_xlabel('Annualised Volatility (Risk)')
        ax.set_ylabel('Annualised Return')
        ax.grid(True, alpha=0.3)
        chart_b64 = _fig_to_base64(fig)

        portfolios = {}
        for name, w in [('Maximum Sharpe Ratio', w_msr),
                        ('Global Minimum Volatility', w_gmv),
                        ('Equal Weight', w_ew)]:
            ret = ftk.portfolio_return(w, er)
            vol = ftk.portfolio_vol(w, cov)
            sharpe = (ret - riskfree_rate) / vol if vol > 0 else 0
            weights_list = [{'asset': str(er.index[i]), 'allocation': f"{w[i]:.1%}"}
                           for i in range(len(w))]
            portfolios[name] = {
                'return': f"{ret:.2%}",
                'volatility': f"{vol:.2%}",
                'sharpe': f"{sharpe:.3f}",
                'weights': weights_list,
            }

        return {'chart': chart_b64, 'portfolios': portfolios}
    except Exception as e:
        return {'error': str(e)}


def _build_stress(returns_df):
    """Build stress test results."""
    try:
        n_assets = len(returns_df.columns)
        weights = np.repeat(1 / n_assets, n_assets)
        stress_results = ftk.stress_test_report(returns_df, weights)

        if not stress_results:
            return None

        formatted = {}
        for scenario, metrics in stress_results.items():
            formatted[scenario] = {
                'Annualized Return': f"{metrics['Annualized Return']:.2%}",
                'Annualized Vol': f"{metrics['Annualized Vol']:.2%}",
                'VaR(5%)': f"{metrics['VaR(5%)']:.2%}",
                'CVaR(5%)': f"{metrics['CVaR(5%)']:.2%}",
                'Max Drawdown': f"{metrics['Max Drawdown']:.2%}",
            }

        # Build comparison chart
        plt.rcParams.update({
            'axes.facecolor': '#1a2332', 'figure.facecolor': '#1a2332',
            'text.color': '#e2e8f0', 'axes.labelcolor': '#94a3b8',
            'xtick.color': '#64748b', 'ytick.color': '#64748b',
        })
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        scenarios = list(stress_results.keys())
        var_vals = [stress_results[s]['VaR(5%)'] for s in scenarios]
        cvar_vals = [stress_results[s]['CVaR(5%)'] for s in scenarios]
        x = np.arange(len(scenarios))
        w = 0.35
        ax1.bar(x - w / 2, var_vals, w, label='VaR (5%)', color='#ff9800', alpha=0.85)
        ax1.bar(x + w / 2, cvar_vals, w, label='CVaR (5%)', color='#d32f2f', alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios, rotation=25, ha='right', fontsize=8)
        ax1.set_title('VaR & CVaR by Scenario', fontsize=13, color='#fff')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        rets = [stress_results[s]['Annualized Return'] for s in scenarios]
        colors = ['#4caf50' if r > 0 else '#d32f2f' for r in rets]
        ax2.bar(range(len(scenarios)), rets, color=colors, alpha=0.85)
        ax2.set_title('Annualised Return by Scenario', fontsize=13, color='#fff')
        ax2.set_ylabel('Return')
        ax2.axhline(0, color='white', linestyle='-', alpha=0.3)
        ax2.set_xticks(range(len(scenarios)))
        ax2.set_xticklabels(scenarios, rotation=25, ha='right', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        chart_b64 = _fig_to_base64(fig)

        return {'scenarios': formatted, 'chart': chart_b64}
    except Exception as e:
        return {'error': str(e)}


def _build_regimes(returns_df):
    """Build market regime data."""
    try:
        window = min(60, max(10, len(returns_df) // 3))
        regimes, rolling_ret, rolling_vol = ftk.detect_market_regimes(returns_df, window=window)
        regime_perf = ftk.regime_performance_summary(returns_df, regimes)

        # Distribution
        valid = regimes[regimes != 'Insufficient Data']
        distribution = {}
        if len(valid) > 0:
            counts = valid.value_counts()
            pcts = counts / len(valid) * 100
            for name, pct in pcts.items():
                distribution[name] = f"{pct:.1f}%"

        # Performance table
        perf_table = None
        if not regime_perf.empty:
            cols = list(regime_perf.columns)
            rows = []
            for regime_name in regime_perf.index:
                row = {'asset': str(regime_name)}
                for col in cols:
                    val = regime_perf.loc[regime_name, col]
                    if col in ('Mean Return (Ann.)', 'Volatility (Ann.)'):
                        row[col] = f"{val:.2%}"
                    elif col == 'Sharpe Ratio':
                        row[col] = f"{val:.3f}"
                    elif col == '% of Total':
                        row[col] = f"{val:.1f}%"
                    else:
                        row[col] = str(int(val)) if col == 'Periods' else f"{val}"
                rows.append(row)
            perf_table = {'columns': cols, 'rows': rows}

        # Regime timeline chart
        plt.rcParams.update({
            'axes.facecolor': '#1a2332', 'figure.facecolor': '#1a2332',
            'text.color': '#e2e8f0', 'axes.labelcolor': '#94a3b8',
            'xtick.color': '#64748b', 'ytick.color': '#64748b',
        })
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        cum = (1 + returns_df.mean(axis=1)).cumprod()
        color_map = {
            'Bull / Low Vol': '#4caf50', 'Bull / High Vol': '#ffeb3b',
            'Bear / Low Vol': '#ff9800', 'Bear / High Vol': '#d32f2f',
            'Insufficient Data': '#bdbdbd'
        }
        ax1.plot(cum.index, cum.values, color='white', linewidth=1, alpha=0.6)

        prev, start_i = None, 0
        for i in range(len(regimes)):
            cur = regimes.iloc[i]
            if cur != prev and prev is not None:
                ax1.axvspan(regimes.index[start_i], regimes.index[i - 1],
                            alpha=0.2, color=color_map.get(prev, '#bdbdbd'))
                start_i = i
            prev = cur
        if prev:
            ax1.axvspan(regimes.index[start_i], regimes.index[-1],
                        alpha=0.2, color=color_map.get(prev, '#bdbdbd'))

        ax1.set_title('Cumulative Returns with Regime Overlay', fontsize=13, color='#fff')
        ax1.set_ylabel('Cumulative Return')
        ax1.grid(True, alpha=0.3)

        from matplotlib.patches import Patch
        ax1.legend(handles=[
            Patch(facecolor=c, alpha=0.4, label=r)
            for r, c in color_map.items() if r != 'Insufficient Data'
        ], loc='upper left', fontsize=8)

        valid_vol = rolling_vol.dropna()
        if len(valid_vol) > 0:
            ax2.plot(valid_vol.index, valid_vol.values, color='#38bdf8', linewidth=1.5)
            ax2.axhline(valid_vol.median(), color='red', linestyle='--', alpha=0.5,
                        label=f'Median: {valid_vol.median():.2%}')
            ax2.set_title('Rolling Annualised Volatility', fontsize=13, color='#fff')
            ax2.set_ylabel('Volatility')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        chart_b64 = _fig_to_base64(fig)

        return {
            'distribution': distribution,
            'performance': perf_table,
            'chart': chart_b64,
        }
    except Exception as e:
        return {'error': str(e)}


# ══════════════════════════════════════════════════════════════════
# AI Analysis
# ══════════════════════════════════════════════════════════════════

@app.route('/api/ai-analysis', methods=['POST'])
def ai_analysis():
    data = request.get_json(silent=True) or {}
    sid = data.get('session_id', '')
    sess = _sessions.get(sid)
    if not sess:
        return jsonify({'error': 'No analysis session found. Please run portfolio analysis first.'})

    api_key = _get_api_key()
    llm = _get_llm(api_key)

    summary_df = sess['raw_summary']
    returns_df = sess['returns_df']

    if not llm:
        # Fallback basic advice
        return jsonify({'analysis': _basic_advice_html(summary_df)})

    prompt_template = """
You are an expert financial advisor. Analyse this portfolio data and give clear, actionable insights.

## Portfolio Summary Statistics:
{summary}

## Context:
- Assets: {num_assets}
- Period: {start} to {end}
- Observations: {obs}

Provide a concise analysis covering:
1. **Performance** — which assets performed best/worst and why it matters
2. **Risk** — highlight any concerning volatility or drawdown figures
3. **Diversification** — comment on correlation and concentration risk
4. **Recommendations** — 3 specific, actionable suggestions
5. **Key Takeaway** — one sentence summary for a non-expert investor

Be direct and reference actual numbers from the data. Format your response in HTML.
"""
    try:
        from langchain.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        result = chain.invoke({
            'summary': summary_df.to_markdown(),
            'num_assets': len(summary_df),
            'start': returns_df.index[0].strftime('%Y-%m-%d'),
            'end': returns_df.index[-1].strftime('%Y-%m-%d'),
            'obs': len(returns_df),
        })
        return jsonify({'analysis': result.content})
    except Exception as e:
        return jsonify({'error': f'AI analysis error: {str(e)}',
                       'analysis': _basic_advice_html(summary_df)})


def _basic_advice_html(summary_df):
    best = summary_df['Annualized Return'].idxmax()
    worst = summary_df['Annualized Return'].idxmin()
    avg_sharpe = summary_df['Sharpe Ratio'].mean()
    risk_note = ('Portfolio shows good risk-adjusted returns.'
                 if avg_sharpe > 0.5
                 else 'Consider rebalancing to improve risk-adjusted returns.')
    vol_note = ('High volatility detected — review position sizing.'
                if summary_df['Annualized Vol'].max() > 0.25
                else 'Volatility is within acceptable range.')
    return f"""
    <h3>Portfolio Analysis Summary</h3>
    <p><strong>Best performer:</strong> {best} — {summary_df.loc[best, 'Annualized Return']:.2%} annual return</p>
    <p><strong>Worst performer:</strong> {worst} — {summary_df.loc[worst, 'Annualized Return']:.2%} annual return</p>
    <p><strong>Portfolio avg Sharpe Ratio:</strong> {avg_sharpe:.3f}</p>
    <h4>Recommendations</h4>
    <ul>
        <li>{risk_note}</li>
        <li>{vol_note}</li>
        <li>Review highly correlated assets for concentration risk.</li>
    </ul>
    """


# ══════════════════════════════════════════════════════════════════
# Chat Q&A
# ══════════════════════════════════════════════════════════════════

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided'})

    api_key = _get_api_key()
    llm = _get_llm(api_key)

    if not llm:
        return jsonify({'error': 'AI not available. Please set a valid Gemini API key in the sidebar.'})

    system_prompt = """
You are ArthShastraAI, an expert financial educator and assistant.
Provide clear, insightful, and helpful answers to financial questions.
Give definitions, explanations, and practical advice.
Always be helpful and direct — never refuse to answer a reasonable finance question.
Format your response in HTML with proper tags (p, ul, li, strong, em, h3, h4).

QUESTION: {question}

ANSWER:
"""
    try:
        from langchain.prompts import PromptTemplate
        chain = PromptTemplate.from_template(system_prompt) | llm
        answer = chain.invoke({'question': question}).content
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': f'Chat error: {str(e)}'})


@app.route('/api/chat/clear', methods=['POST'])
def chat_clear():
    return jsonify({'success': True})


# ══════════════════════════════════════════════════════════════════
# Download
# ══════════════════════════════════════════════════════════════════

@app.route('/api/download-summary')
def download_summary():
    sid = request.args.get('session_id', '')
    sess = _sessions.get(sid)
    if not sess:
        return jsonify({'error': 'No session found'}), 404

    csv_data = sess['raw_summary'].to_csv()
    buf = io.BytesIO(csv_data.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='text/csv', as_attachment=True,
                     download_name=f'portfolio_summary_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv')


# ══════════════════════════════════════════════════════════════════
# Local dev server
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)