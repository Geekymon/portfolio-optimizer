import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize
import datetime as dt

class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.01):
        self.original_tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.download_data()
        if hasattr(self, 'data'):
            self.calculate_returns()
        
    def download_data(self):
        """Download stock data using yfinance with batch processing"""
        try:
            with st.spinner('Downloading stock data...'):
                data = yf.download(
                    self.original_tickers,
                    start=self.start_date,
                    end=self.end_date,
                    group_by='ticker',
                    progress=False
                )
                
                self.data = pd.DataFrame()
                valid_tickers = []
                
                for ticker in self.original_tickers:
                    if ticker in data.columns.get_level_values(0):
                        close_col = 'Adj Close' if 'Adj Close' in data[ticker].columns else 'Close'
                        self.data[ticker] = data[ticker][close_col]
                        valid_tickers.append(ticker)
                
                if self.data.empty:
                    st.error("Could not fetch data for any of the provided tickers.")
                    st.stop()
                
                self.tickers = valid_tickers
                self.data.ffill(inplace=True)
                self.data.bfill(inplace=True)
                
                missing_data = self.data.isnull().sum()
                if missing_data.any():
                    st.warning(f"Missing data handled for: {missing_data[missing_data > 0].index.tolist()}")
                
                if len(self.data) < 252:
                    st.warning("Warning: Less than one year of data available. Results may be less reliable.")
                    
        except Exception as e:
            st.error(f"Data download error: {str(e)}")
            st.stop()
            
    def calculate_returns(self):
        """Calculate returns with numerical stability"""
        self.returns = self.data.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov() + np.eye(len(self.tickers)) * 1e-9
        
    def portfolio_performance(self, weights):
        """Calculate portfolio metrics with weight validation"""
        weights = np.array(weights)
        weights /= weights.sum()  # Ensure exact sum to 1
        returns = np.sum(self.mean_returns * weights) * 252
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        return returns, volatility
        
    def negative_sharpe_ratio(self, weights):
        """Calculate risk-adjusted return metric"""
        p_ret, p_vol = self.portfolio_performance(weights)
        if p_vol < 1e-6:
            return np.inf
        return -(p_ret - self.risk_free_rate) / p_vol
        
    def optimize_portfolio(self):
        """Portfolio optimization with error handling"""
        num_assets = len(self.tickers)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        try:
            result = minimize(
                self.negative_sharpe_ratio,
                x0=np.ones(num_assets)/num_assets,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                st.warning(f"Optimization warning: {result.message}")
                
            return result
            
        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            return None
        
    def efficient_frontier(self, num_portfolios=2000):
        """Generate efficient frontier with progress tracking"""
        returns = []
        volatilities = []
        progress_bar = st.progress(0)
        
        for i in range(num_portfolios):
            weights = np.random.random(len(self.tickers))
            weights /= weights.sum()
            p_ret, p_vol = self.portfolio_performance(weights)
            returns.append(p_ret)
            volatilities.append(p_vol)
            if i % 100 == 0:
                progress_bar.progress((i+1)/num_portfolios)
                
        progress_bar.empty()
        return returns, volatilities

def main():
    st.set_page_config(layout="wide", page_title="Portfolio Optimizer 2.0")
    
    if 'tickers' not in st.session_state:
        st.session_state.tickers = "AAPL,MSFT,GOOGL,AMZN,BND"
    
    st.title("Enhanced MPT Portfolio Optimizer")
    st.markdown("""
    **Improved Features:**
    - Batch data downloading
    - Automatic invalid ticker handling
    - Numerical stability improvements
    - Real-time progress tracking
    """)
    
    st.sidebar.header("Configuration")
    tickers_input = st.sidebar.text_area(
        "Assets (comma-separated)", 
        st.session_state.tickers,
        help="Include stocks (AAPL) and bonds (BND) for diversification"
    )
    
    today = dt.date.today()
    start_date = st.sidebar.date_input(
        "Start Date", 
        today - dt.timedelta(days=365*3),
        max_value=today - dt.timedelta(days=252)
    )
    
    end_date = st.sidebar.date_input(
        "End Date", today,
        min_value=start_date + dt.timedelta(days=252)
    )
    
    if st.sidebar.button("Optimize Portfolio"):
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        st.session_state.tickers = ", ".join(tickers)
        
        if len(tickers) < 2:
            st.error("At least 2 assets required for portfolio construction")
            return
            
        optimizer = PortfolioOptimizer(tickers, start_date, end_date)
        
        result = optimizer.optimize_portfolio()
        if not result:
            return
            
        optimal_weights = result.x
        opt_ret, opt_vol = optimizer.portfolio_performance(optimal_weights)
        
        assert np.isclose(optimal_weights.sum(), 1), "Weight sum validation failed"
        assert opt_vol > 0, "Volatility validation failed"
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Optimal Allocation")
            weights_df = pd.DataFrame({
                'Asset': optimizer.tickers,
                'Weight': optimal_weights
            }).sort_values('Weight', ascending=False)
            
            fig_pie = go.Figure(go.Pie(
                labels=weights_df['Asset'],
                values=weights_df['Weight'],
                hole=0.4,
                textinfo='label+percent'
            ))
            fig_pie.update_layout(
                title='Portfolio Composition',
                showlegend=False
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            st.subheader("Efficient Frontier")
            returns, volatilities = optimizer.efficient_frontier()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=volatilities, y=returns,
                mode='markers',
                name='Possible Portfolios',
                marker=dict(
                    color=(np.array(returns) - optimizer.risk_free_rate)/np.array(volatilities),
                    colorscale='Viridis',
                    size=7,
                    showscale=True,
                    colorbar=dict(title='Sharpe Ratio')
                )
            ))
            
            fig.add_trace(go.Scatter(
                x=[opt_vol], y=[opt_ret],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star-diamond'
                )
            ))
            
            fig.update_layout(
                title='Risk-Return Tradeoff',
                xaxis_title='Annualized Volatility',
                yaxis_title='Annualized Return',
                hovermode='x unified',
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Advanced Analytics"):
            tab1, tab2, tab3 = st.tabs(["Correlations", "Returns", "Optimization Details"])
            
            with tab1:
                corr_matrix = optimizer.returns.corr()
                st.plotly_chart(go.Figure(
                    go.Heatmap(
                        z=corr_matrix,
                        x=optimizer.tickers,
                        y=optimizer.tickers,
                        colorscale='RdBu',
                        zmid=0
                    )
                ), use_container_width=True)
            
            with tab2:
                cumulative_returns = (1 + optimizer.returns).cumprod()
                st.plotly_chart(go.Figure(
                    data=[go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns[col],
                        name=col
                    ) for col in optimizer.tickers]  # Fixed reference
                ), use_container_width=True)
            
            with tab3:
                st.json({
                    "optimization_parameters": {
                        "method": "SLSQP",
                        "max_iterations": 1000,
                        "risk_free_rate": optimizer.risk_free_rate,
                        "assets_considered": len(optimizer.tickers)
                    },
                    "validation_checks": {
                        "weight_sum": float(optimal_weights.sum()),
                        "positive_volatility": opt_vol > 0
                    }
                })

if __name__ == "__main__":
    main()
