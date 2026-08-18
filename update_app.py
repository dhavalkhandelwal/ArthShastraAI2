import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update Tabs Definition
content = content.replace(
    'tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([',
    'tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(['
)
content = content.replace('"🎲 Monte Carlo",\n', '')
content = content.replace('\'🎲 Monte Carlo\',\n', '')

# 2. Update Summary Stats formatting
old_summary = '''                        percentage_cols = ['Annualized Return', 'Annualized Vol', 'Cornish-Fisher VaR (5%)', 
                                         'Historic CVaR (5%)', 'Max Drawdown']
                        
                        for col in percentage_cols:
                            if col in formatted_summary.columns:
                                formatted_summary[col] = formatted_summary[col].apply(lambda x: f"{x:.2%}")
                        
                        if 'Sharpe Ratio' in formatted_summary.columns:
                            formatted_summary['Sharpe Ratio'] = formatted_summary['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
                        if 'Skewness' in formatted_summary.columns:
                            formatted_summary['Skewness'] = formatted_summary['Skewness'].apply(lambda x: f"{x:.3f}")
                        if 'Kurtosis' in formatted_summary.columns:
                            formatted_summary['Kurtosis'] = formatted_summary['Kurtosis'].apply(lambda x: f"{x:.3f}")'''

new_summary = '''                        percentage_cols = ['Annualized Return', 'Annualized Vol', 'Parametric VaR (5%)', 
                                         'Historic CVaR (5%)', 'Max Drawdown']
                        
                        for col in percentage_cols:
                            if col in formatted_summary.columns:
                                formatted_summary[col] = formatted_summary[col].apply(lambda x: f"{x:.2%}")
                        
                        if 'Sharpe Ratio' in formatted_summary.columns:
                            formatted_summary['Sharpe Ratio'] = formatted_summary['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")'''

content = content.replace(old_summary, new_summary)

# 3. Delete 'with tab5:' block which goes until 'with tab6:'
pattern = re.compile(r'                    with tab5:.*?                    with tab6:', re.DOTALL)
content = re.sub(pattern, '                    with tab6:', content)

# 4. Shift tabs
content = content.replace('with tab6:', 'with tab5:')
content = content.replace('with tab7:', 'with tab6:')
content = content.replace('with tab8:', 'with tab7:')
content = content.replace('with tab9:', 'with tab8:')
content = content.replace('with tab10:', 'with tab9:')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('app.py updated successfully.')
