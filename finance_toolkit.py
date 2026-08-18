import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import math
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def detect_frequency(returns_data):
    """
    Detect the frequency of the data (daily, weekly, monthly, etc.)
    Returns periods_per_year for annualization.

    Thresholds:
        <= 4 days avg gap  -> 252  (daily/business-day; business days avg ~1.4 days)
        <= 10 days         -> 52   (weekly)
        <= 35 days         -> 12   (monthly)
        <= 100 days        -> 4    (quarterly)
        else               -> 1    (annual)
    """
    if isinstance(returns_data, pd.DataFrame):
        index = returns_data.index
    else:
        index = returns_data.index

    if len(index) < 2:
        return 252  # Default to daily

    # Calculate average calendar-day gap between observations
    time_diffs = pd.Series(index).diff().dropna()
    avg_diff = time_diffs.mean()

    if avg_diff <= pd.Timedelta(days=4):
        return 252  # Daily or business-day
    elif avg_diff <= pd.Timedelta(days=10):
        return 52   # Weekly
    elif avg_diff <= pd.Timedelta(days=35):
        return 12   # Monthly
    elif avg_diff <= pd.Timedelta(days=100):
        return 4    # Quarterly
    else:
        return 1    # Annual

def annualize_rets(r, periods_per_year=None):
    """
    Annualizes a set of returns
    """
    if periods_per_year is None:
        periods_per_year = detect_frequency(r)
    
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    if n_periods == 0:
        return 0
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year=None):
    """
    Annualizes the vol of a set of returns
    """
    if periods_per_year is None:
        periods_per_year = detect_frequency(r)
    
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year=None):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    if periods_per_year is None:
        periods_per_year = detect_frequency(r)
    
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    if ann_vol == 0:
        return 0
    return ann_ex_ret/ann_vol



def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns.
    returns a DataFrame with columns for
    the wealth index, 
    the previous peaks, and 
    the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})

def max_drawdown(r):
    """
    Calculate maximum drawdown
    """
    dd = drawdown(r)
    return dd.Drawdown.min()

def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level) if len(r) > 0 else 0
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        if len(r) == 0:
            return 0
        is_beyond = r <= -var_historic(r, level=level)
        if is_beyond.sum() == 0:
            return 0
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def var_gaussian(r, level=5):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    return -(r.mean() + z*r.std(ddof=0))

def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 matrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5

def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    try:
        weights = minimize(portfolio_vol, init_guess,
                           args=(cov,), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,return_is_target),
                           bounds=bounds)
        return weights.x
    except:
        return init_guess

def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        if vol == 0:
            return -np.inf
        return -(r - riskfree_rate)/vol
    
    try:
        weights = minimize(neg_sharpe, init_guess,
                           args=(riskfree_rate, er, cov), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        return weights.x
    except:
        return init_guess

def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def optimal_weights(n_points, er, cov):
    """
    Returns a list of weights that represent a grid of n_points on the efficient frontier
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef(n_points, er, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False, ax=None):
    """
    Plots the multi-asset efficient frontier on a given matplotlib axis.
    """
    # If no axis is provided, create a new one
    if ax is None:
        ax = plt.gca()

    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    
    # *** CORRECTION: Plot on the provided 'ax' object ***
    ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend, ax=ax)
    
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
        
    return ax

def correlation_analysis(returns_df):
    """
    Perform correlation analysis and return insights
    """
    corr_matrix = returns_df.corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # High correlation threshold
                high_corr_pairs.append({
                    'asset1': corr_matrix.columns[i],
                    'asset2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    return corr_matrix, high_corr_pairs

def rolling_metrics(returns_df, window=12):
    """
    Calculate rolling metrics for trend analysis
    """
    periods_per_year = detect_frequency(returns_df)
    rolling_vol = returns_df.rolling(window=window).std() * np.sqrt(periods_per_year)
    rolling_sharpe = returns_df.rolling(window=window).apply(
        lambda x: sharpe_ratio(pd.Series(x), 0.03, periods_per_year=periods_per_year) if len(x) == window else np.nan
    )
    
    return rolling_vol, rolling_sharpe

def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    periods_per_year = detect_frequency(r)
    
    ann_r = r.aggregate(annualize_rets, periods_per_year=periods_per_year)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=periods_per_year)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year)
    dd = r.aggregate(max_drawdown)
    var5 = r.aggregate(var_gaussian)
    hist_cvar5 = r.aggregate(cvar_historic)
    
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Parametric VaR (5%)": var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })

def generate_portfolio_insights(returns_df, summary_stats_df):
    """
    Generate comprehensive portfolio insights
    """
    insights = []
    
    # Performance insights
    best_performer = summary_stats_df['Annualized Return'].idxmax()
    worst_performer = summary_stats_df['Annualized Return'].idxmin()
    insights.append(f"🏆 Best performing asset: {best_performer} ({summary_stats_df.loc[best_performer, 'Annualized Return']:.2%} annual return)")
    insights.append(f"📉 Worst performing asset: {worst_performer} ({summary_stats_df.loc[worst_performer, 'Annualized Return']:.2%} annual return)")
    
    # Risk insights
    riskiest = summary_stats_df['Annualized Vol'].idxmax()
    safest = summary_stats_df['Annualized Vol'].idxmin()
    insights.append(f"⚠️ Highest risk asset: {riskiest} ({summary_stats_df.loc[riskiest, 'Annualized Vol']:.2%} volatility)")
    insights.append(f"🛡️ Lowest risk asset: {safest} ({summary_stats_df.loc[safest, 'Annualized Vol']:.2%} volatility)")
    
    # Sharpe ratio insights
    best_sharpe = summary_stats_df['Sharpe Ratio'].idxmax()
    insights.append(f"📊 Best risk-adjusted returns: {best_sharpe} (Sharpe ratio: {summary_stats_df.loc[best_sharpe, 'Sharpe Ratio']:.3f})")
    
    # Drawdown insights
    max_dd_asset = summary_stats_df['Max Drawdown'].idxmin()  # Most negative
    insights.append(f"📉 Largest drawdown: {max_dd_asset} ({summary_stats_df.loc[max_dd_asset, 'Max Drawdown']:.2%})")
    
    # Correlation insights
    corr_matrix, high_corr_pairs = correlation_analysis(returns_df)
    if high_corr_pairs:
        insights.append("🔗 High correlation pairs found:")
        for pair in high_corr_pairs[:3]:  # Show top 3
            insights.append(f"   • {pair['asset1']} & {pair['asset2']}: {pair['correlation']:.3f}")
    
    return insights



def apply_stress_scenario(returns_df, scenario_name):
    """
    Applies predefined stress scenarios to the returns data.
    """
    scenarios = {
        '2008 Financial Crisis': {
            'return_shock': -0.40,
            'vol_multiplier': 3.0,
            'recovery_factor': 0.0,
            'description': 'Severe market downturn mimicking the 2008 Global Financial Crisis with extreme volatility spike'
        },
        'COVID-19 Crash': {
            'return_shock': -0.35,
            'vol_multiplier': 2.5,
            'recovery_factor': 0.4,
            'description': 'Sharp sudden decline similar to March 2020 crash with rapid partial recovery'
        },
        'Rising Interest Rates': {
            'return_shock': -0.20,
            'vol_multiplier': 1.5,
            'recovery_factor': 0.1,
            'description': 'Gradual decline in equity values due to monetary tightening and sector rotation'
        }
    }
    
    if scenario_name not in scenarios:
        return None, None
        
    try:
        scenario = scenarios[scenario_name]
        stressed_returns = returns_df * scenario['vol_multiplier'] + scenario['return_shock'] / len(returns_df)
        return stressed_returns, scenario
    except Exception as e:
        warnings.warn(f"Stress scenario application failed: {str(e)}")
        return None, None

def stress_test_report(returns_df, weights=None):
    """
    Runs stress tests on portfolio and generates a report.
    """
    try:
        if weights is None:
            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
            
        portfolio_returns = returns_df.dot(weights)
        
        normal_ann_ret = portfolio_returns.mean() * 252
        normal_ann_vol = portfolio_returns.std() * np.sqrt(252)
        normal_var = np.percentile(portfolio_returns, 5)
        normal_cvar = portfolio_returns[portfolio_returns <= normal_var].mean()
        
        cum_ret = (1 + portfolio_returns).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        normal_max_dd = drawdown.min()
        
        results = {
            'Normal': {
                'VaR(5%)': normal_var,
                'CVaR(5%)': normal_cvar,
                'Annualized Return': normal_ann_ret,
                'Annualized Vol': normal_ann_vol,
                'Max Drawdown': normal_max_dd,
                'Capital Adequacy Ratio': max(0, min(100, 1 / (abs(normal_cvar) * np.sqrt(252)) if normal_cvar != 0 else 100))
            }
        }
        
        scenarios = ['2008 Financial Crisis', 'COVID-19 Crash', 'Rising Interest Rates']
        for scenario_name in scenarios:
            stressed_returns_df, _ = apply_stress_scenario(returns_df, scenario_name)
            if stressed_returns_df is not None:
                stressed_portfolio_returns = stressed_returns_df.dot(weights)
                stressed_ann_ret = stressed_portfolio_returns.mean() * 252
                stressed_ann_vol = stressed_portfolio_returns.std() * np.sqrt(252)
                stressed_var = np.percentile(stressed_portfolio_returns, 5)
                stressed_cvar = stressed_portfolio_returns[stressed_portfolio_returns <= stressed_var].mean()
                
                s_cum_ret = (1 + stressed_portfolio_returns).cumprod()
                s_running_max = s_cum_ret.cummax()
                s_drawdown = (s_cum_ret - s_running_max) / s_running_max
                stressed_max_dd = s_drawdown.min()
                
                car = 100
                if stressed_cvar != 0:
                    car = max(0, min(100, 1 / (abs(stressed_cvar) * np.sqrt(252))))
                    
                results[scenario_name] = {
                    'VaR(5%)': stressed_var,
                    'CVaR(5%)': stressed_cvar,
                    'Annualized Return': stressed_ann_ret,
                    'Annualized Vol': stressed_ann_vol,
                    'Max Drawdown': stressed_max_dd,
                    'Capital Adequacy Ratio': car
                }
                
        return results
    except Exception as e:
        warnings.warn(f"Stress test report failed: {str(e)}")
        return {}

def factor_analysis(returns_df):
    """
    Performs factor analysis using synthetic proxy factors.
    """
    import numpy.linalg as npl
    try:
        n_assets = len(returns_df.columns)
        n_periods = len(returns_df)
        
        if n_assets < 2 or n_periods < 10:
            return pd.DataFrame(), pd.DataFrame()
            
        factors_df = pd.DataFrame(index=returns_df.index)
        factors_df['Market'] = returns_df.mean(axis=1)
        
        vols = returns_df.std()
        median_vol = vols.median()
        high_vol_assets = vols[vols > median_vol].index
        low_vol_assets = vols[vols <= median_vol].index
        
        if len(high_vol_assets) > 0 and len(low_vol_assets) > 0:
            factors_df['Size'] = returns_df[high_vol_assets].mean(axis=1) - returns_df[low_vol_assets].mean(axis=1)
        else:
            factors_df['Size'] = 0
            
        means = returns_df.mean()
        median_mean = means.median()
        high_mean_assets = means[means > median_mean].index
        low_mean_assets = means[means <= median_mean].index
        
        if len(high_mean_assets) > 0 and len(low_mean_assets) > 0:
            factors_df['Value'] = returns_df[high_mean_assets].mean(axis=1) - returns_df[low_mean_assets].mean(axis=1)
        else:
            factors_df['Value'] = 0
            
        window = min(60, n_periods // 4)
        if window > 10:
            rolling_means = returns_df.rolling(window=window).mean().shift(1)
            momentum_factor = pd.Series(0, index=returns_df.index, dtype=float)
            for i in range(window, n_periods):
                current_rolling = rolling_means.iloc[i]
                if not current_rolling.isna().all():
                    median_roll = current_rolling.median()
                    winners = current_rolling[current_rolling > median_roll].index
                    losers = current_rolling[current_rolling <= median_roll].index
                    if len(winners) > 0 and len(losers) > 0:
                        momentum_factor.iloc[i] = returns_df[winners].iloc[i].mean() - returns_df[losers].iloc[i].mean()
            factors_df['Momentum'] = momentum_factor
        else:
            factors_df['Momentum'] = 0
            
        results_list = []
        X = factors_df[['Market', 'Size', 'Value', 'Momentum']].values
        X_with_const = np.column_stack([np.ones(n_periods), X])
        
        for col in returns_df.columns:
            y = returns_df[col].values
            valid_idx = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
            if valid_idx.sum() > 4:
                X_valid = X_with_const[valid_idx]
                y_valid = y[valid_idx]
                coef, residuals, rank, s = npl.lstsq(X_valid, y_valid, rcond=None)
                
                y_mean = np.mean(y_valid)
                ss_tot = np.sum((y_valid - y_mean)**2)
                ss_res = np.sum((y_valid - X_valid.dot(coef))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                results_list.append({
                    'Asset': col,
                    'Alpha': coef[0],
                    'Market Beta': coef[1],
                    'Size (SMB)': coef[2],
                    'Value (HML)': coef[3],
                    'Momentum': coef[4],
                    'R-squared': r_squared
                })
                
        results_df = pd.DataFrame(results_list).set_index('Asset')
        return results_df, factors_df
    except Exception as e:
        warnings.warn(f"Factor analysis failed: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def risk_contribution_analysis(weights, cov_matrix, asset_names):
    """
    Calculates marginal and component risk contributions.
    """
    try:
        if isinstance(cov_matrix, pd.DataFrame):
            cov = cov_matrix.values
        else:
            cov = cov_matrix
            
        weights = np.array(weights)
        portfolio_var = weights.T @ cov @ weights
        portfolio_vol = np.sqrt(portfolio_var)
        
        mrc = (cov @ weights) / portfolio_vol
        crc = weights * mrc
        pct_contrib = (crc / portfolio_vol) * 100
        
        df = pd.DataFrame({
            'Asset': asset_names,
            'Weight': weights,
            'Weight %': weights * 100,
            'Marginal Risk Contribution': mrc,
            'Component Risk Contribution': crc,
            'Risk Contribution %': pct_contrib
        })
        return df
    except Exception as e:
        warnings.warn(f"Risk contribution analysis failed: {str(e)}")
        return pd.DataFrame()

def detect_market_regimes(returns_df, window=60):
    """
    Detects market regimes based on returns and volatility.
    Uses detect_frequency() to correctly annualize for daily/weekly/monthly data.
    """
    try:
        portfolio_returns = returns_df.mean(axis=1)
        periods_per_year = detect_frequency(returns_df)  # dynamic — not hardcoded to 252
        
        rolling_mean = portfolio_returns.rolling(window=window).mean()
        rolling_std = portfolio_returns.rolling(window=window).std()
        
        rolling_ann_ret = rolling_mean * periods_per_year
        rolling_ann_vol = rolling_std * np.sqrt(periods_per_year)
        
        median_vol = rolling_ann_vol.median()
        
        regimes_series = pd.Series('Insufficient Data', index=returns_df.index)
        
        bull_low_mask = (rolling_ann_ret > 0) & (rolling_ann_vol <= median_vol)
        bull_high_mask = (rolling_ann_ret > 0) & (rolling_ann_vol > median_vol)
        bear_low_mask = (rolling_ann_ret <= 0) & (rolling_ann_vol <= median_vol)
        bear_high_mask = (rolling_ann_ret <= 0) & (rolling_ann_vol > median_vol)
        
        regimes_series[bull_low_mask] = 'Bull / Low Vol'
        regimes_series[bull_high_mask] = 'Bull / High Vol'
        regimes_series[bear_low_mask] = 'Bear / Low Vol'
        regimes_series[bear_high_mask] = 'Bear / High Vol'
        
        regimes_series[rolling_ann_ret.isna() | rolling_ann_vol.isna()] = 'Insufficient Data'
        
        return regimes_series, rolling_ann_ret, rolling_ann_vol
    except Exception as e:
        warnings.warn(f"Regime detection failed: {str(e)}")
        return pd.Series(), pd.Series(), pd.Series()

def regime_performance_summary(returns_df, regimes):
    """
    Summarizes performance metrics by market regime.
    Column names are deliberately kept stable so app.py can reference them safely.
    """
    try:
        portfolio_returns = returns_df.mean(axis=1)
        periods_per_year = detect_frequency(returns_df)  # dynamic frequency
        results = []
        
        unique_regimes = regimes.unique()
        for regime in unique_regimes:
            if regime == 'Insufficient Data':
                continue
                
            regime_returns = portfolio_returns[regimes == regime]
            count = len(regime_returns)
            pct_total = count / len(regimes) * 100
            
            if count > 0:
                ann_ret = regime_returns.mean() * periods_per_year
                ann_vol = regime_returns.std() * np.sqrt(periods_per_year)
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            else:
                ann_ret = 0
                ann_vol = 0
                sharpe = 0
                
            results.append({
                'Regime': regime,
                'Periods': count,
                '% of Total': pct_total,
                'Mean Return (Ann.)': ann_ret,
                'Volatility (Ann.)': ann_vol,
                'Sharpe Ratio': sharpe
            })
            
        if results:
            return pd.DataFrame(results).set_index('Regime')
        else:
            return pd.DataFrame()
    except Exception as e:
        warnings.warn(f"Regime performance summary failed: {str(e)}")
        return pd.DataFrame()

def generate_pdf_report(summary_stats_df, insights_list, stress_results=None, portfolio_name='Portfolio'):
    """
    Generates a professional PDF report.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    import io
    from datetime import datetime
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#1a237e'), spaceAfter=12)
        subtitle_style = ParagraphStyle('SubtitleStyle', parent=styles['Normal'], fontSize=12, textColor=colors.gray, spaceAfter=24)
        heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#1a237e'), spaceBefore=18, spaceAfter=12)
        normal_style = styles['Normal']
        
        elements = []
        
        # Title
        elements.append(Paragraph('ArthShastraAI - Portfolio Analysis Report', title_style))
        elements.append(Paragraph(f'Generated on {datetime.now().strftime("%Y-%m-%d")}', subtitle_style))
        
        # Section 1: Executive Summary
        elements.append(Paragraph('1. Executive Summary', heading_style))
        for insight in insights_list:
            elements.append(Paragraph(f'• {insight}', normal_style))
            elements.append(Spacer(1, 6))
            
        # Section 2: Summary Statistics
        elements.append(Paragraph('2. Summary Statistics', heading_style))
        
        table_data = [['Asset/Metric'] + list(summary_stats_df.columns)]
        for index, row in summary_stats_df.iterrows():
            formatted_row = [index]
            for val in row:
                if isinstance(val, float):
                    formatted_row.append(f"{val:.4f}")
                else:
                    formatted_row.append(str(val))
            table_data.append(formatted_row)
            
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
        ]))
        elements.append(t)
        
        # Section 3: Risk Metrics
        elements.append(Paragraph('3. Key Risk Metrics', heading_style))
        elements.append(Paragraph("A high-level view of portfolio risk indicators.", normal_style))
        
        # Section 4: Stress Test Results
        if stress_results is not None:
            elements.append(Paragraph('4. Stress Test Results', heading_style))
            
            stress_data = [['Scenario', 'VaR(5%)', 'CVaR(5%)', 'Max Drawdown', 'Capital Adequacy Ratio']]
            for scenario, metrics in stress_results.items():
                stress_data.append([
                    scenario,
                    f"{metrics.get('VaR(5%)', 0):.2%}",
                    f"{metrics.get('CVaR(5%)', 0):.2%}",
                    f"{metrics.get('Max Drawdown', 0):.2%}",
                    f"{metrics.get('Capital Adequacy Ratio', 0):.2f}"
                ])
                
            t_stress = Table(stress_data)
            t_stress.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
            ]))
            elements.append(t_stress)
            
        # Section 5: Recommendations
        elements.append(Paragraph('5. Key Recommendations', heading_style))
        if 'Sharpe Ratio' in summary_stats_df.columns:
            mean_sharpe = summary_stats_df['Sharpe Ratio'].mean()
            if mean_sharpe > 1:
                elements.append(Paragraph("• Excellent risk-adjusted returns overall. Focus on capital preservation.", normal_style))
            elif mean_sharpe > 0.5:
                elements.append(Paragraph("• Good risk-adjusted returns. Consider tactical rebalancing to optimize further.", normal_style))
            else:
                elements.append(Paragraph("• Sub-optimal risk-adjusted returns. Re-evaluate asset allocation.", normal_style))
                
        elements.append(Spacer(1, 24))
        elements.append(Paragraph('ArthShastraAI - Comprehensive Financial Analysis', ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.gray, alignment=1)))
        
        doc.build(elements)
        buffer.seek(0)
        return buffer
    except Exception as e:
        warnings.warn(f"PDF generation failed: {str(e)}")
        return None