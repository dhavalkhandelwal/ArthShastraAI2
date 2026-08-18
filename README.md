# 📊 ArthShastraAI — AI-Powered Portfolio Analytics Platform

> **Institutional-grade portfolio risk analytics, democratised for retail and professional investors.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Google Gemini](https://img.shields.io/badge/Gemini-3.6%20Flash-4285F4?logo=google)](https://ai.google.dev)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-purple)](https://github.com/facebookresearch/faiss)

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
| **RAG Knowledge Base** | Upload annual reports, prospectuses, or research — chat with documents via AI |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ArthShastraAI                               │
├──────────────┬──────────────────────────┬───────────────────────────┤
│  Portfolio   │     AI Analytics Layer   │     RAG Q&A Layer         │
│  Optimizer   │                          │                           │
│  Pro         │  ┌────────────────────┐  │  ┌──────────────────────┐ │
│              │  │ finance_toolkit.py │  │  │ LangChain + FAISS    │ │
│  CSV Upload  │  │ - Sharpe / VaR     │  │  │ Vector Store         │ │
│     ↓        │  │ - CVaR / Drawdown  │  │  │ HuggingFace Embeds   │ │
│  Returns     │  │ - Efficient Frontier│  │  │ (all-MiniLM-L6-v2)  │ │
│  Engine      │  │ - Factor Analysis  │  │  └──────────┬───────────┘ │
│              │  │ - Stress Testing   │  │             │             │
│              │  │ - Regime Detection │  │  ┌──────────▼───────────┐ │
│              │  │ - Risk Budgeting   │  │  │  Gemini 3.6 Flash    │ │
│              │  └────────────────────┘  │  │  (Google AI Studio)  │ │
│              │                          │  └──────────────────────┘ │
└──────────────┴──────────────────────────┴───────────────────────────┘
                        │
               ┌────────▼────────┐
               │  Streamlit UI   │
               │  9-tab Dashboard│
               │  PDF Report Gen │
               └─────────────────┘
```

---

## ✨ Features

### 📊 Portfolio Optimizer Pro (9 Analysis Tabs)

| Tab | Feature | Technical Depth |
|-----|---------|----------------|
| Summary Stats | Annualised return, vol, Sharpe, VaR, CVaR, Max Drawdown | Parametric (Gaussian) + Historic CVaR |
| Visualisations | Cumulative returns, Risk-Return scatter, Correlation heatmap, Rolling vol | Seaborn + Matplotlib |
| AI Analysis | GPT-grade narrative insights on your exact numbers | Gemini 3.6 Flash via LangChain |
| Efficient Frontier | Max Sharpe / GMV / Equal Weight portfolios | Scipy SLSQP constrained optimisation |
| **Stress Testing** | 2008 Crisis, COVID-19, Rising Rates scenarios | **Basel III / CCAR inspired** |
| **Factor Analysis** | Alpha, Market Beta, Size (SMB), Value (HML), Momentum | **Fama-French 4-factor OLS regression** |
| **Risk Budgeting** | Marginal & Component Risk Contribution per asset | **Euler decomposition of portfolio variance** |
| **Market Regimes** | Bull/Bear × High/Low Vol detection over rolling windows | Regime-conditional performance attribution |
| Insights & Report | One-click executive PDF report | ReportLab professional PDF generation |

### 🤖 Ask Anything (RAG Q&A)
- Hybrid retrieval: checks document knowledge base first, falls back to Gemini's general financial expertise
- FAISS vector store with `all-MiniLM-L6-v2` sentence embeddings
- Persistent chat history with fallback storage

### 📚 Upload Knowledge
- Upload `.txt`, `.md`, `.csv` files into the FAISS knowledge base
- Chunked with `RecursiveCharacterTextSplitter` (1000 chars, 100 overlap)
- Real-time Q&A grounding on uploaded content

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit |
| **AI / LLM** | Google Gemini 3.6 Flash via `langchain-google-genai` |
| **RAG / Embeddings** | LangChain, FAISS, HuggingFace `all-MiniLM-L6-v2` |
| **Quantitative Finance** | Custom `finance_toolkit.py` (NumPy, SciPy, Pandas) |
| **Optimisation** | `scipy.optimize.minimize` (SLSQP), `PyPortfolioOpt` |
| **Visualisation** | Matplotlib, Seaborn |
| **PDF Reports** | ReportLab |
| **Data** | Pandas, NumPy |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A Google AI Studio API key (free at [aistudio.google.com](https://aistudio.google.com))

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/ArthShastraAI.git
cd ArthShastraAI

# Install dependencies
pip install -r requirements.txt

# Set your API key (recommended over hardcoding)
export GOOGLE_API_KEY="your_key_here"   # Linux/macOS
$env:GOOGLE_API_KEY="your_key_here"    # Windows PowerShell

# Run the app
streamlit run app.py
```

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

### Factor Model (Fama-French Inspired)
OLS regression of each asset's returns against 4 synthetic factors:
$$r_i = \alpha_i + \beta_M \cdot r_M + \beta_{SMB} \cdot SMB + \beta_{HML} \cdot HML + \beta_{Mom} \cdot Mom + \varepsilon_i$$

### Risk Budgeting (Euler Decomposition)
- **Marginal Risk Contribution**: $MRC_i = \frac{(\Sigma w)_i}{\sqrt{w^T \Sigma w}}$
- **Component Risk Contribution**: $CRC_i = w_i \cdot MRC_i$

---

## 📁 Project Structure

```
ArthShastraAI/
├── app.py                  # Main Streamlit application (1200+ lines)
├── finance_toolkit.py      # Quantitative finance engine (750+ lines)
├── requirements.txt        # Python dependencies
├── sample_portfolio.csv    # Demo data for testing
├── generate_sample_data.py # Script to regenerate sample data
└── data/
    ├── faiss_index/        # Persistent FAISS vector store
    └── chat_history.json   # Persistent Q&A chat history
```

---

## 🔒 Security Note

The app reads `GOOGLE_API_KEY` from the environment variable first (`os.environ.get`), then from Streamlit secrets, before falling back to any bundled value. **Never commit an active API key to a public repository.** Regenerate your key at [aistudio.google.com](https://aistudio.google.com) if it has been exposed.

---

## 👤 Author

**Dhawal Khandelwal**  
Built as a personal project demonstrating applied ML, quantitative finance, and full-stack AI development.

---

*ArthShastra (अर्थशास्त्र) — the ancient Indian treatise on statecraft and economic policy by Kautilya. This project applies that spirit of rigorous analytical thinking to modern portfolio management.*
