import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import traceback
import finance_toolkit as ftk

# ── Page config — must be the very first Streamlit call ───────────────────────
st.set_page_config(
    page_title="ArthShastraAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── API Key ────────────────────────────────────────────────────────────────────
# The app checks environment variable, Streamlit secrets, or sidebar input
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GEMINI_API_KEY:
    try:
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

if GEMINI_API_KEY:
    os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY


try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.warning("LangChain not available. AI features disabled.")

# ── Load LLM ──────────────────────────────────────────────────────────────────
llm = None  # initialised after sidebar renders (needs API key input)

# ── Chat History ───────────────────────────────────────────────────────────────
CHAT_HISTORY_PATH = os.path.join("data", "chat_history.json")
os.makedirs("data", exist_ok=True)

def load_chat_history():
    try:
        if os.path.exists(CHAT_HISTORY_PATH):
            with open(CHAT_HISTORY_PATH, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_chat_history(history):
    try:
        with open(CHAT_HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# ── Data processing helpers ────────────────────────────────────────────────────
def detect_data_frequency(df_index):
    periods_per_year = ftk.detect_frequency(pd.Series(index=df_index))
    freq_map = {252: "Daily", 52: "Weekly", 12: "Monthly", 4: "Quarterly", 1: "Annual"}
    return freq_map.get(periods_per_year, "Unknown"), periods_per_year

def general_csv_preprocessor(df):
    st.write("#### 1. Data Preview")
    st.dataframe(df.head(10))

    st.write("#### 2. Configure Your Data")
    col1, col2 = st.columns(2)

    date_col = col1.selectbox("Select the date column", [''] + list(df.columns))
    if not date_col:
        st.warning("Please select a date column.")
        return None, None

    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        st.error(f"Could not parse '{date_col}' as dates: {e}")
        return None, None

    df_sorted = df.set_index(date_col).sort_index()
    freq_name, periods_per_year = detect_data_frequency(df_sorted.index)
    st.info(f"Detected frequency: **{freq_name}** ({periods_per_year} periods/year)")

    numeric_cols = [c for c in df_sorted.columns if pd.api.types.is_numeric_dtype(df_sorted[c])]
    if not numeric_cols:
        st.error("No numeric columns found.")
        return None, None

    asset_cols = col2.multiselect(
        "Select asset columns (prices / values)",
        numeric_cols,
        default=numeric_cols[:min(10, len(numeric_cols))]
    )
    if not asset_cols:
        st.warning("Please select at least one asset column.")
        return None, None

    st.write("#### 3. Processing Options")
    col3, col4 = st.columns(2)
    fill_method = col3.selectbox("Handle missing values:", ["Forward fill", "Drop rows", "Linear interpolation"])
    return_type = col4.selectbox("Return calculation:", ["Simple returns", "Log returns"])

    prices_df = df_sorted[asset_cols].copy()
    if fill_method == "Forward fill":
        prices_df = prices_df.ffill()
    elif fill_method == "Linear interpolation":
        prices_df = prices_df.interpolate()
    prices_df = prices_df.dropna()

    if prices_df.empty:
        st.error("No data remaining after processing.")
        return None, None

    if return_type == "Simple returns":
        returns_df = prices_df.pct_change().dropna()
    else:
        returns_df = np.log(prices_df / prices_df.shift(1)).dropna()

    st.write("#### 4. Processed Returns")
    st.dataframe(returns_df.head(10))
    st.success(f"✅ {len(returns_df)} periods × {len(asset_cols)} assets ready for analysis")
    return returns_df, periods_per_year

# ── Visualizations ─────────────────────────────────────────────────────────────
def create_visualizations(returns_df, summary_stats_df, periods_per_year):
    # Cumulative returns
    st.subheader("📈 Cumulative Returns")
    cum = (1 + returns_df).cumprod()
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    for col in cum.columns:
        ax1.plot(cum.index, cum[col], label=col, linewidth=2)
    ax1.set_title("Cumulative Returns Over Time", fontsize=15)
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # Risk-Return scatter
    st.subheader("⚖️ Risk vs Return")
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    sc = ax2.scatter(
        summary_stats_df['Annualized Vol'],
        summary_stats_df['Annualized Return'],
        s=120, alpha=0.8,
        c=summary_stats_df['Sharpe Ratio'], cmap='RdYlGn'
    )
    for i, asset in enumerate(summary_stats_df.index):
        ax2.annotate(asset,
                     (summary_stats_df.iloc[i]['Annualized Vol'],
                      summary_stats_df.iloc[i]['Annualized Return']),
                     xytext=(6, 6), textcoords='offset points', fontsize=9)
    ax2.set_xlabel("Annualized Volatility (Risk)")
    ax2.set_ylabel("Annualized Return")
    ax2.set_title("Risk-Return Profile  (colour = Sharpe Ratio)", fontsize=13)
    plt.colorbar(sc, label='Sharpe Ratio')
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

    # Correlation heatmap
    st.subheader("🔗 Asset Correlation Matrix")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(returns_df.corr(), annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax3, fmt='.2f')
    ax3.set_title("Pearson Correlation of Returns")
    st.pyplot(fig3)
    plt.close(fig3)

    # Rolling volatility
    st.subheader("📉 Rolling Volatility")
    window = min(max(len(returns_df) // 10, 10), 60)
    rolling_vol = returns_df.rolling(window=window).std() * np.sqrt(periods_per_year)
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    for col in rolling_vol.columns:
        ax4.plot(rolling_vol.index, rolling_vol[col], label=col, alpha=0.85)
    ax4.set_title(f"Rolling {window}-Period Annualised Volatility", fontsize=13)
    ax4.set_ylabel("Volatility")
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

# ── AI Portfolio Advice ────────────────────────────────────────────────────────
def get_portfolio_advice(summary_df, returns_df):
    if not llm:
        return _basic_advice(summary_df)

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

Be direct and reference actual numbers from the data.
"""
    try:
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        with st.spinner("🤖 ArthShastraAI is generating your analysis..."):
            result = chain.invoke({
                "summary": summary_df.to_markdown(),
                "num_assets": len(summary_df),
                "start": returns_df.index[0].strftime("%Y-%m-%d"),
                "end": returns_df.index[-1].strftime("%Y-%m-%d"),
                "obs": len(returns_df)
            })
        return result.content
    except Exception as e:
        st.error(f"AI error: {e}")
        return _basic_advice(summary_df)

def _basic_advice(summary_df):
    best = summary_df['Annualized Return'].idxmax()
    worst = summary_df['Annualized Return'].idxmin()
    avg_sharpe = summary_df['Sharpe Ratio'].mean()
    advice = f"""
## Portfolio Analysis Summary

**Best performer:** {best} — {summary_df.loc[best,'Annualized Return']:.2%} annual return  
**Worst performer:** {worst} — {summary_df.loc[worst,'Annualized Return']:.2%} annual return  
**Portfolio avg Sharpe Ratio:** {avg_sharpe:.3f}

### Recommendations
- {"Portfolio shows good risk-adjusted returns." if avg_sharpe > 0.5 else "Consider rebalancing to improve risk-adjusted returns."}
- {"High volatility detected — review position sizing." if summary_df['Annualized Vol'].max() > 0.25 else "Volatility is within acceptable range."}
- Review highly correlated assets for concentration risk.
"""
    return advice

# ── Efficient Frontier ─────────────────────────────────────────────────────────
def show_efficient_frontier(er, cov, riskfree_rate=0.03):
    try:
        w_msr = ftk.msr(riskfree_rate, er, cov)
        w_gmv = ftk.gmv(cov)
        w_ew  = np.repeat(1/len(er), len(er))

        fig, ax = plt.subplots(figsize=(11, 7))
        ftk.plot_ef(n_points=30, er=er, cov=cov, ax=ax,
                    show_cml=True, show_ew=True, show_gmv=True,
                    riskfree_rate=riskfree_rate)
        ax.set_title("Efficient Frontier with Optimal Portfolios", fontsize=15)
        ax.set_xlabel("Annualised Volatility (Risk)")
        ax.set_ylabel("Annualised Return")
        ax.plot([], [], 'o-', color='blue',       label='Efficient Frontier')
        ax.plot([], [], 'o', color='green',        markersize=10, label='Max Sharpe Ratio')
        ax.plot([], [], 'o', color='midnightblue', markersize=10, label='Min Volatility')
        ax.plot([], [], 'o', color='goldenrod',    markersize=10, label='Equal Weight')
        ax.plot([], [], '--', color='green',       label='Capital Market Line')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        # Show portfolio compositions
        portfolios = {
            'Maximum Sharpe Ratio': (w_msr, '🟢'),
            'Global Minimum Volatility': (w_gmv, '🔵'),
            'Equal Weight': (w_ew, '🟡'),
        }
        st.subheader("Optimal Portfolio Compositions")
        for name, (w, icon) in portfolios.items():
            ret = ftk.portfolio_return(w, er)
            vol = ftk.portfolio_vol(w, cov)
            sharpe = (ret - riskfree_rate) / vol if vol > 0 else 0
            with st.expander(f"{icon} {name}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Expected Return", f"{ret:.2%}")
                    st.metric("Volatility",      f"{vol:.2%}")
                    st.metric("Sharpe Ratio",    f"{sharpe:.3f}")
                with c2:
                    wdf = pd.DataFrame({'Asset': er.index, 'Allocation': w})
                    wdf['Allocation'] = wdf['Allocation'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(wdf, hide_index=True, use_container_width=True)
    except Exception as e:
        st.error(f"Efficient Frontier error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
">
    <h1 style="color:#ffffff; font-size:2.6rem; font-weight:800; margin:0; letter-spacing:-0.5px;">
        📊 ArthShastraAI
    </h1>
    <p style="color:#94a3b8; font-size:1.05rem; margin-top:0.5rem; margin-bottom:1.5rem;">
        Institutional-grade portfolio analytics — powered by AI, built for retail &amp; professional investors.
    </p>
    <div style="display:flex; gap:1.2rem; flex-wrap:wrap;">
        <div style="background:rgba(255,255,255,0.07); border-radius:10px; padding:0.8rem 1.2rem; min-width:150px;">
            <div style="color:#38bdf8; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px;">Problem Solved</div>
            <div style="color:#fff; font-size:0.9rem; margin-top:0.3rem;">Retail investors have no access to institutional risk analytics</div>
        </div>
        <div style="background:rgba(255,255,255,0.07); border-radius:10px; padding:0.8rem 1.2rem; min-width:150px;">
            <div style="color:#34d399; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px;">Risk Analytics</div>
            <div style="color:#fff; font-size:0.9rem; margin-top:0.3rem;">Sharpe, VaR, CVaR, Drawdown &amp; Efficient Frontier in seconds</div>
        </div>
        <div style="background:rgba(255,255,255,0.07); border-radius:10px; padding:0.8rem 1.2rem; min-width:150px;">
            <div style="color:#f472b6; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px;">Stress Testing</div>
            <div style="color:#fff; font-size:0.9rem; margin-top:0.3rem;">Simulate 2008 Crisis &amp; COVID crash impact on your portfolio</div>
        </div>
        <div style="background:rgba(255,255,255,0.07); border-radius:10px; padding:0.8rem 1.2rem; min-width:150px;">
            <div style="color:#fb923c; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px;">AI Layer</div>
            <div style="color:#fff; font-size:0.9rem; margin-top:0.3rem;">Gemini 3.6 Flash gives plain-English advice on your exact numbers</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — API KEY + NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("📊 ArthShastraAI")

# API key input
_has_key = bool(GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE")
with st.sidebar.expander("🔑 Gemini API Key & Model", expanded=not _has_key):
    if not _has_key:
        st.markdown("Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).")
    user_key = st.text_input(
        "Gemini API Key",
        value=GEMINI_API_KEY if _has_key else "",
        type="password"
    )
    selected_model = st.selectbox(
        "Select Model",
        ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
        index=0,
        help="Select Google Gemini model"
    )
    if user_key:
        GEMINI_API_KEY = user_key.strip()
        os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY

# Load LLM after key is known
@st.cache_resource
def load_llm(api_key: str, model_name: str = "gemini-3.6-flash"):
    if not LANGCHAIN_AVAILABLE or not api_key:
        return None
    try:
        os.environ['GOOGLE_API_KEY'] = api_key
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.6, google_api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"AI load error: {e}")
        return None

llm = load_llm(GEMINI_API_KEY, selected_model) if GEMINI_API_KEY else None

st.sidebar.markdown("---")
if llm:
    st.sidebar.success("✅ AI features enabled")
else:
    st.sidebar.warning("⚠️ Enter a valid API key to enable AI features")

mode = st.sidebar.radio("Navigation", ["Portfolio Analyzer", "Ask Anything (Q&A)"])
st.sidebar.markdown("---")
st.sidebar.markdown("""
**How to use:**
1. Paste your Gemini API key above
2. Go to **Portfolio Analyzer**
3. Upload a CSV with dates + prices
""")

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: PORTFOLIO ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Portfolio Analyzer":
    st.header("🗂️ Portfolio Analyzer")
    st.markdown("""
    Upload a CSV with historical asset prices. ArthShastraAI will:
    - Auto-detect data frequency and calculate returns
    - Compute Sharpe Ratio, VaR, CVaR, Max Drawdown
    - Plot cumulative returns, correlations, and risk-return profile
    - Run Markowitz Efficient Frontier optimisation
    - Generate AI-powered portfolio insights using Gemini
    """)

    uploaded_file = st.file_uploader(
        "Upload your portfolio CSV",
        type="csv",
        help="One date column + one numeric column per asset/ticker"
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns")

            result = general_csv_preprocessor(df)
            if result and result[0] is not None:
                returns_df, periods_per_year = result

                with st.spinner("Running portfolio analysis..."):
                    raw_summary  = ftk.summary_stats(returns_df, riskfree_rate=0.03)
                    er           = ftk.annualize_rets(returns_df)
                    cov          = returns_df.cov() * ftk.detect_frequency(returns_df)
                    insights     = ftk.generate_portfolio_insights(returns_df, raw_summary)

                # ── Top KPI bar ─────────────────────────────────────────────
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("🏆 Best Return",   f"{raw_summary['Annualized Return'].max():.2%}")
                k2.metric("📉 Avg Volatility", f"{raw_summary['Annualized Vol'].mean():.2%}")
                k3.metric("⭐ Best Sharpe",    f"{raw_summary['Sharpe Ratio'].max():.3f}")
                k4.metric("🕳️ Max Drawdown",  f"{raw_summary['Max Drawdown'].min():.2%}")

                st.markdown("---")

                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 Summary Stats",
                    "📈 Visualizations",
                    "🤖 AI Analysis",
                    "🎯 Efficient Frontier",
                    "🏦 Stress Testing",
                    "🔄 Market Regimes"
                ])

                # ── Tab 1: Summary Stats ────────────────────────────────────
                with tab1:
                    st.subheader("Portfolio Summary Statistics")
                    fmt = raw_summary.copy()
                    pct_cols = ['Annualized Return', 'Annualized Vol',
                                'Parametric VaR (5%)', 'Historic CVaR (5%)', 'Max Drawdown']
                    for col in pct_cols:
                        if col in fmt.columns:
                            fmt[col] = fmt[col].apply(lambda x: f"{x:.2%}")
                    if 'Sharpe Ratio' in fmt.columns:
                        fmt['Sharpe Ratio'] = fmt['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
                    st.dataframe(fmt, use_container_width=True)

                    st.markdown("#### 🔍 Key Findings")
                    for insight in insights:
                        st.write(f"• {insight}")

                    # Download
                    st.download_button(
                        "⬇️ Download Summary CSV",
                        data=raw_summary.to_csv(),
                        file_name=f"portfolio_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                # ── Tab 2: Visualizations ───────────────────────────────────
                with tab2:
                    create_visualizations(returns_df, raw_summary, periods_per_year)

                # ── Tab 3: AI Analysis ──────────────────────────────────────
                with tab3:
                    st.subheader("🤖 AI-Powered Portfolio Analysis")
                    if llm:
                        st.markdown(get_portfolio_advice(raw_summary, returns_df))
                    else:
                        st.warning("AI not available. Showing basic analysis.")
                        st.markdown(_basic_advice(raw_summary))

                # ── Tab 4: Efficient Frontier ───────────────────────────────
                with tab4:
                    st.subheader("🎯 Efficient Frontier — Optimal Allocations")
                    st.markdown("""
                    **What this shows:** For any level of risk, there is an optimal mix of assets that
                    maximises return. This chart (Markowitz, 1952) plots that frontier and highlights
                    three special portfolios:
                    - 🟢 **Max Sharpe Ratio** — best return per unit of risk
                    - 🔵 **Min Volatility** — lowest possible risk
                    - 🟡 **Equal Weight** — naïve 1/N benchmark
                    """)
                    show_efficient_frontier(er, cov)

                # ── Tab 5: Stress Testing ───────────────────────────────────
                with tab5:
                    st.subheader("🏦 Portfolio Stress Testing")
                    st.markdown("""
                    Simulate how your portfolio would have performed under three major historical
                    market crises. This shows downside risk beyond normal market conditions.
                    """)
                    try:
                        n_assets = len(returns_df.columns)
                        stress_weights = np.repeat(1/n_assets, n_assets)
                        stress_results = ftk.stress_test_report(returns_df, stress_weights)

                        if stress_results:
                            scenario_names = [s for s in stress_results if s != 'Normal']
                            icons = {'2008 Financial Crisis': '💥', 'COVID-19 Crash': '🦠', 'Rising Interest Rates': '📈'}
                            normal_ret = stress_results['Normal']['Annualized Return']

                            st.markdown("### Scenario Impact vs Normal Conditions")
                            cols_s = st.columns(len(scenario_names))
                            for col_s, scenario in zip(cols_s, scenario_names):
                                with col_s:
                                    icon = icons.get(scenario, '⚠️')
                                    stressed_ret = stress_results[scenario]['Annualized Return']
                                    st.markdown(f"**{icon} {scenario}**")
                                    st.metric("Stressed Return",    f"{stressed_ret:.2%}",
                                              delta=f"{stressed_ret - normal_ret:.2%}")
                                    st.metric("VaR (5%)",   f"{stress_results[scenario]['VaR(5%)']:.2%}")
                                    st.metric("CVaR (5%)",  f"{stress_results[scenario]['CVaR(5%)']:.2%}")
                                    st.metric("Max Drawdown",f"{stress_results[scenario]['Max Drawdown']:.2%}")

                            st.markdown("### Comparison Table")
                            table = {}
                            for s, m in stress_results.items():
                                table[s] = {
                                    'Ann. Return':   f"{m['Annualized Return']:.2%}",
                                    'Ann. Vol':      f"{m['Annualized Vol']:.2%}",
                                    'VaR (5%)':      f"{m['VaR(5%)']:.2%}",
                                    'CVaR (5%)':     f"{m['CVaR(5%)']:.2%}",
                                    'Max Drawdown':  f"{m['Max Drawdown']:.2%}",
                                }
                            st.dataframe(pd.DataFrame(table).T, use_container_width=True)

                            # Chart
                            st.markdown("### Visual Comparison")
                            fig_s, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(14, 5))
                            scenarios_all = list(stress_results.keys())
                            var_vals  = [stress_results[s]['VaR(5%)']  for s in scenarios_all]
                            cvar_vals = [stress_results[s]['CVaR(5%)'] for s in scenarios_all]
                            x = np.arange(len(scenarios_all))
                            w = 0.35
                            ax_s1.bar(x - w/2, var_vals,  w, label='VaR (5%)',  color='#ff9800', alpha=0.85)
                            ax_s1.bar(x + w/2, cvar_vals, w, label='CVaR (5%)', color='#d32f2f', alpha=0.85)
                            ax_s1.set_xticks(x)
                            ax_s1.set_xticklabels(scenarios_all, rotation=25, ha='right', fontsize=8)
                            ax_s1.set_title('VaR & CVaR by Scenario', fontsize=13)
                            ax_s1.legend(); ax_s1.grid(True, alpha=0.3, axis='y')

                            rets_all = [stress_results[s]['Annualized Return'] for s in scenarios_all]
                            colors   = ['#4caf50' if r > 0 else '#d32f2f' for r in rets_all]
                            ax_s2.bar(range(len(scenarios_all)), rets_all, color=colors, alpha=0.85)
                            ax_s2.set_title('Annualised Return by Scenario', fontsize=13)
                            ax_s2.set_ylabel('Return')
                            ax_s2.axhline(0, color='black', linestyle='-', alpha=0.3)
                            ax_s2.set_xticks(range(len(scenarios_all)))
                            ax_s2.set_xticklabels(scenarios_all, rotation=25, ha='right', fontsize=8)
                            ax_s2.grid(True, alpha=0.3, axis='y')
                            plt.tight_layout()
                            st.pyplot(fig_s)
                            plt.close(fig_s)
                    except Exception as e:
                        st.error(f"Stress test error: {e}")

                # ── Tab 6: Market Regimes ───────────────────────────────────
                with tab6:
                    st.subheader("🔄 Market Regime Detection")
                    st.markdown("""
                    Identifies whether the market was in a **Bull or Bear** phase with
                    **High or Low Volatility** across your data period — and shows how
                    your portfolio performed in each regime.
                    """)
                    try:
                        regime_window = st.slider(
                            "Rolling window (periods)",
                            min_value=10,
                            max_value=min(120, len(returns_df)//2),
                            value=min(60, len(returns_df)//3)
                        )
                        regimes, rolling_ret, rolling_vol_r = ftk.detect_market_regimes(
                            returns_df, window=regime_window
                        )
                        regime_perf = ftk.regime_performance_summary(returns_df, regimes)

                        # Regime distribution
                        valid = regimes[regimes != 'Insufficient Data']
                        if len(valid) > 0:
                            counts = valid.value_counts()
                            pcts   = counts / len(valid) * 100
                            icons_r = {'Bull / Low Vol':'🟢','Bull / High Vol':'🟡',
                                       'Bear / Low Vol':'🟠','Bear / High Vol':'🔴'}
                            st.markdown("### Time Spent in Each Regime")
                            reg_cols = st.columns(len(counts))
                            for col_r, (name, pct) in zip(reg_cols, pcts.items()):
                                with col_r:
                                    st.metric(f"{icons_r.get(name,'⚪')} {name}",
                                              f"{pct:.1f}%",
                                              help=f"{counts[name]} periods")

                        if not regime_perf.empty:
                            st.markdown("### Performance by Regime")
                            disp = regime_perf.copy()
                            for c in ['Mean Return (Ann.)', 'Volatility (Ann.)']:
                                if c in disp.columns:
                                    disp[c] = disp[c].apply(lambda x: f"{x:.2%}")
                            if 'Sharpe Ratio' in disp.columns:
                                disp['Sharpe Ratio'] = disp['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
                            if 'Periods'    in disp.columns: disp['Periods']    = disp['Periods'].astype(int)
                            if '% of Total' in disp.columns: disp['% of Total'] = disp['% of Total'].apply(lambda x: f"{x:.1f}%")
                            st.dataframe(disp, use_container_width=True)

                        # Regime timeline
                        st.markdown("### Regime Timeline")
                        fig_r, (ax_r1, ax_r2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
                        cum_r = (1 + returns_df.mean(axis=1)).cumprod()
                        color_map = {'Bull / Low Vol':'#4caf50','Bull / High Vol':'#ffeb3b',
                                     'Bear / Low Vol':'#ff9800','Bear / High Vol':'#d32f2f',
                                     'Insufficient Data':'#bdbdbd'}
                        ax_r1.plot(cum_r.index, cum_r.values, color='black', linewidth=1, alpha=0.6)
                        prev, start_i = None, 0
                        for i in range(len(regimes)):
                            cur = regimes.iloc[i]
                            if cur != prev and prev is not None:
                                ax_r1.axvspan(regimes.index[start_i], regimes.index[i-1],
                                              alpha=0.2, color=color_map.get(prev,'#bdbdbd'))
                                start_i = i
                            prev = cur
                        if prev:
                            ax_r1.axvspan(regimes.index[start_i], regimes.index[-1],
                                          alpha=0.2, color=color_map.get(prev,'#bdbdbd'))
                        ax_r1.set_title('Cumulative Returns with Regime Overlay', fontsize=13)
                        ax_r1.set_ylabel('Cumulative Return')
                        ax_r1.grid(True, alpha=0.3)
                        from matplotlib.patches import Patch
                        ax_r1.legend(handles=[
                            Patch(facecolor=c, alpha=0.4, label=r)
                            for r, c in color_map.items() if r != 'Insufficient Data'
                        ], loc='upper left', fontsize=8)

                        valid_vol = rolling_vol_r.dropna()
                        if len(valid_vol) > 0:
                            ax_r2.plot(valid_vol.index, valid_vol.values, color='#1976d2', linewidth=1.5)
                            ax_r2.axhline(valid_vol.median(), color='red', linestyle='--', alpha=0.5,
                                          label=f'Median: {valid_vol.median():.2%}')
                            ax_r2.set_title('Rolling Annualised Volatility', fontsize=13)
                            ax_r2.set_ylabel('Volatility')
                            ax_r2.legend(fontsize=8)
                            ax_r2.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_r)
                        plt.close(fig_r)
                    except Exception as e:
                        st.error(f"Regime detection error: {e}")

        except Exception as e:
            st.error(f"Error processing file: {e}")
            if st.checkbox("Show traceback"):
                import traceback as tb
                st.code(tb.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: ASK ANYTHING (Q&A)
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Ask Anything (Q&A)":
    st.header("💬 Financial Q&A — Ask Anything")
    st.markdown("Ask any finance question — portfolio theory, market concepts, investing strategies, risk management. Powered by **Gemini 3.6 Flash**.")

    if not llm:
        st.error("AI not available. Please check your API key.")
    else:
        # Render chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a finance question..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            system_prompt = """
You are ArthShastraAI, an expert financial educator and assistant.
Provide clear, insightful, and helpful answers to financial questions.
Give definitions, explanations, and practical advice.
Always be helpful and direct — never refuse to answer a reasonable finance question.

QUESTION: {question}

ANSWER:
"""
            try:
                with st.spinner("ArthShastraAI is thinking..."):
                    chain = PromptTemplate.from_template(system_prompt) | llm
                    answer = chain.invoke({"question": prompt}).content

                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                save_chat_history(st.session_state.chat_history)
            except Exception as e:
                st.error(f"Error: {e}")

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            save_chat_history([])
            st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; padding:16px;'>"
    "ArthShastraAI · Built by Dhawal Khandelwal · Streamlit + Google Gemini + Python"
    "</div>",
    unsafe_allow_html=True
)