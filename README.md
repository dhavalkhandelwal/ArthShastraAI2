# 📊 ArthShastraAI — AI-Powered Portfolio Analytics Platform

> **Institutional-grade portfolio risk analytics, democratised for retail and professional investors.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com)
[![Google Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?logo=google)](https://ai.google.dev)
[![Vercel](https://img.shields.io/badge/Deploy-Vercel-black?logo=vercel)](https://vercel.com)

---

## 🎯 Problem Statement

Retail and HNI investors in India manage ₹40+ trillion in mutual funds and direct equities — yet they have **zero access** to the risk management tools used by institutional players:

- **Banks** run Basel III stress tests across economic crisis scenarios
- **Hedge funds** decompose returns using Fama-French factor models  
- **Portfolio managers** optimise allocations using mean-variance (Markowitz) theory

These tools live behind Bloomberg terminals and proprietary systems costing $25,000+/year. **ArthShastraAI closes this gap** — bringing institutional-grade analytics to anyone with a CSV file of their portfolio.

---

## 💡 Business Impact

| Metric | Value |
|--------|-------|
| **Target Market** | 10M+ active retail investors in India |
| **Tools Democratised** | Basel III stress-testing, Fama-French factor model, Efficient Frontier optimisation |
| **Decision Speed** | Full portfolio risk report generated in < 30 seconds vs. days manually |
| **Cost Reduction** | 100% free vs. $25,000+/year for Bloomberg/FactSet equivalents |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ArthShastraAI                               │
├──────────────┬──────────────────────────┬───────────────────────────┤
│  Flask API   │     Analytics Engine     │     AI Layer              │
│              │                          │                           │
│  /api/       │  ┌────────────────────┐  │  ┌──────────────────────┐ │
│  analyze     │  │ finance_toolkit.py │  │  │ LangChain            │ │
│  ai-analysis │  │ - Sharpe / VaR     │  │  │ Gemini 2.5 Flash     │ │
│  chat        │  │ - CVaR / Drawdown  │  │  │ (Google AI Studio)   │ │
│  set-key     │  │ - Efficient Frontier│  │  └──────────────────────┘ │
│              │  │ - Stress Testing   │  │                           │
│              │  │ - Regime Detection │  │                           │
│              │  └────────────────────┘  │                           │
└──────────────┴──────────────────────────┴───────────────────────────┘
                        │
               ┌────────▼────────┐
               │  HTML/CSS/JS    │
               │  Dark Theme UI  │
               │  Vercel Deploy  │
               └─────────────────┘
```

---

## ✨ Features

### 📊 Portfolio Analyzer (6 Analysis Tabs)

| Tab | Feature | Technical Depth |
|-----|---------|----------------|
| Summary Stats | Annualised return, vol, Sharpe, VaR, CVaR, Max Drawdown | Parametric (Gaussian) + Historic CVaR |
| Visualisations | Cumulative returns, Risk-Return scatter, Correlation heatmap, Rolling vol | Matplotlib charts rendered as base64 |
| AI Analysis | GPT-grade narrative insights on your exact numbers | Gemini 2.5 Flash via LangChain |
| Efficient Frontier | Max Sharpe / GMV / Equal Weight portfolios | Scipy SLSQP constrained optimisation |
| **Stress Testing** | 2008 Crisis, COVID-19, Rising Rates scenarios | **Basel III / CCAR inspired** |
| **Market Regimes** | Bull/Bear × High/Low Vol detection over rolling windows | Regime-conditional performance attribution |

### 🤖 Ask Anything (Q&A)
- Chat with Gemini about any finance topic
- Portfolio theory, market concepts, investing strategies, risk management

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------:|
| **Backend** | Flask (Python) |
| **Frontend** | HTML5 + CSS3 + Vanilla JS |
| **AI / LLM** | Google Gemini 2.5 Flash via `langchain-google-genai` |
| **Quantitative Finance** | Custom `finance_toolkit.py` (NumPy, SciPy, Pandas) |
| **Optimisation** | `scipy.optimize.minimize` (SLSQP) |
| **Visualisation** | Matplotlib, Seaborn (server-side rendering) |
| **Deployment** | Vercel (serverless Python) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A Google AI Studio API key (free at [aistudio.google.com](https://aistudio.google.com))

### Local Development

```bash
# Clone the repository
git clone https://github.com/<your-username>/ArthShastraAI.git
cd ArthShastraAI

# Install dependencies
pip install -r requirements.txt

# Set your API key (optional — can also set in the UI)
export GOOGLE_API_KEY="your_key_here"   # Linux/macOS
$env:GOOGLE_API_KEY="your_key_here"    # Windows PowerShell

# Run the app
python app.py
```

Then open http://localhost:5000 in your browser.

### Deploy to Vercel

1. Push your code to GitHub
2. Import the repo at [vercel.com/new](https://vercel.com/new)
3. Set environment variable `GOOGLE_API_KEY` in Vercel dashboard
4. Deploy — that's it!

### Input Format

Your portfolio CSV should have:
- One **date column** (any parseable format: `YYYY-MM-DD`, `DD/MM/YYYY`, etc.)
- One or more **numeric price/value columns** (one column per asset/ticker)

Example:
```csv
Date,RELIANCE,INFY,TCS,HDFC
2023-01-02,2500.0,1450.0,3200.0,1600.0
2023-01-03,2510.0,1462.0,3215.0,1598.0
...
```

A `sample_portfolio.csv` is included for quick testing.

---

## 📐 Quantitative Methods

### Risk Metrics
- **VaR (Parametric)**: Gaussian assumption — $\text{VaR}_\alpha = -(\mu + z_\alpha \cdot \sigma)$
- **CVaR (Historic)**: Expected shortfall beyond the VaR threshold
- **Max Drawdown**: Peak-to-trough decline in cumulative wealth index

### Portfolio Optimisation (Efficient Frontier)
- **Maximum Sharpe Ratio**: $\max_w \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}$ subject to $\sum w_i = 1, w_i \geq 0$
- **Global Minimum Variance**: $\min_w w^T \Sigma w$
- **Capital Market Line**: Tangency line from risk-free rate to MSR portfolio

---

## 📁 Project Structure

```
ArthShastraAI/
├── app.py                  # Flask application (exports `app` for Vercel)
├── finance_toolkit.py      # Quantitative finance engine (750+ lines)
├── requirements.txt        # Python dependencies
├── vercel.json             # Vercel deployment configuration
├── sample_portfolio.csv    # Demo data for testing
├── templates/
│   └── index.html          # Main HTML template
├── static/
│   ├── style.css           # CSS design system
│   └── app.js              # Frontend JavaScript
└── data/
    └── ...
```

---

## 🔒 Security Note

The app reads `GOOGLE_API_KEY` from the environment variable. You can also set it via the UI sidebar. **Never commit an active API key to a public repository.** Regenerate your key at [aistudio.google.com](https://aistudio.google.com) if it has been exposed.

---

## 👤 Author

**Dhawal Khandelwal**  
Built as a personal project demonstrating applied ML, quantitative finance, and full-stack AI development.

---

*ArthShastra (अर्थशास्त्र) — the ancient Indian treatise on statecraft and economic policy by Kautilya. This project applies that spirit of rigorous analytical thinking to modern portfolio management.*
