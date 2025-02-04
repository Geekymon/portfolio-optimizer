import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize
import datetime as dt

class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.download_data()
        if hasattr(self, 'data'):  # Only proceed if data download was successful
            self.calculate_returns()
        
    def download_data(self):
        """Download stock data using yfinance"""
        try:
            # Initialize empty DataFrame to store closing prices
            self.data = pd.DataFrame()
            
            with st.spinner('Downloading stock data...'):
                for ticker in self.tickers:
                    # Download data for each ticker individually
                    ticker_data = yf.download(ticker, 
                                           start=self.start_date, 
                                           end=self.end_date,
                                           progress=False)
                    if ticker_data.empty:
                        st.error(f"No data available for {ticker}")
                        continue
                    
                    # Use 'Close' price if 'Adj Close' is not available
                    if 'Adj Close' in ticker_data.columns:
                        self.data[ticker] = ticker_data['Adj Close']
                    else:
                        self.data[ticker] = ticker_data['Close']
                        
            if self.data.empty:
                st.error("Could not fetch data for any of the provided tickers.")
                st.stop()
                
            # Check for missing data
            missing_data = self.data.isnull().sum()
            if missing_data.any():
                st.warning(f"Some tickers have missing data points: {missing_data[missing_data > 0]}")
                
            # Check data sufficiency
            if len(self.data) < 252:  # One year of trading days
                st.warning("Warning: Less than one year of data available. Results may be less reliable.")
                
        except Exception as e:
            st.error(f"Error in data download: {str(e)}")
            st.stop()
            
    def calculate_returns(self):
        """Calculate daily returns and covariance matrix"""
        self.returns = self.data.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
    def portfolio_performance(self, weights):
        """Calculate portfolio return and volatility"""
        weights = np.array(weights)
        returns = np.sum(self.mean_returns * weights) * 252
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        return returns, volatility
        
    def negative_sharpe_ratio(self, weights):
        """Calculate negative Sharpe ratio for minimization"""
        p_ret, p_vol = self.portfolio_performance(weights)
        rf = 0.01  # Risk-free rate assumption
        if p_vol == 0:
            return np.inf
        return -(p_ret - rf) / p_vol
        
    def optimize_portfolio(self):
        """Find the optimal portfolio weights"""
        num_assets = len(self.tickers)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1/num_assets] * num_assets)
        
        try:
            result = minimize(self.negative_sharpe_ratio, 
                            initial_weights,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
            
            if not result.success:
                st.warning(f"Optimization may not have converged: {result.message}")
            
            return result
            
        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            return None
        
    def efficient_frontier(self, num_portfolios=1000):
        """Generate points for efficient frontier"""
        returns = []
        volatilities = []
        
        with st.spinner('Generating efficient frontier...'):
            for _ in range(num_portfolios):
                weights = np.random.random(len(self.tickers))
                weights /= np.sum(weights)
                p_ret, p_vol = self.portfolio_performance(weights)
                returns.append(p_ret)
                volatilities.append(p_vol)
                
        return returns, volatilities

def main():
    st.set_page_config(layout="wide", page_title="Portfolio Optimizer")
    
    st.title("Modern Portfolio Theory Optimizer")
    st.markdown("""
    This application helps you optimize your investment portfolio using Modern Portfolio Theory.
    Enter stock tickers separated by commas and select a date range to analyze.
    """)
    
    # Sidebar inputs
    st.sidebar.header("Portfolio Settings")
    tickers_input = st.sidebar.text_area(
        "Enter stock tickers (comma-separated)", 
        "AAPL,MSFT,GOOGL,AMZN",
        help="Enter valid stock tickers separated by commas (e.g., AAPL,MSFT,GOOGL)"
    )
    
    # Date inputs with validation
    today = dt.date.today()
    default_start = today - dt.timedelta(days=365*3)  # 3 years of data
    
    start_date = st.sidebar.date_input(
        "Start Date", 
        default_start,
        max_value=today - dt.timedelta(days=252)  # Must have at least 1 year of data
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        today,
        min_value=start_date + dt.timedelta(days=252)
    )
    
    if st.sidebar.button("Optimize Portfolio"):
        # Clean and validate tickers
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
        
        if not tickers:
            st.error("Please enter at least one valid ticker.")
            return
            
        if len(tickers) < 2:
            st.error("Please enter at least two tickers for portfolio optimization.")
            return
            
        # Initialize optimizer
        optimizer = PortfolioOptimizer(tickers, start_date, end_date)
        
        if not hasattr(optimizer, 'data'):
            return  # Error already displayed by the optimizer
            
        # Get optimal portfolio
        result = optimizer.optimize_portfolio()
        if result is None:
            return
            
        optimal_weights = result.x
        opt_ret, opt_vol = optimizer.portfolio_performance(optimal_weights)
        
        # Generate efficient frontier
        returns, volatilities = optimizer.efficient_frontier()
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Optimal Portfolio Weights")
            weights_df = pd.DataFrame({
                'Stock': tickers,
                'Weight': optimal_weights * 100
            })
            st.dataframe(weights_df.style.format({'Weight': '{:.2f}%'}))
            
            st.subheader("Portfolio Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['Expected Annual Return', 'Annual Volatility', 'Sharpe Ratio'],
                'Value': [
                    f'{opt_ret*100:.2f}%',
                    f'{opt_vol*100:.2f}%',
                    f'{(opt_ret - 0.01)/opt_vol:.2f}'  # Using 1% risk-free rate
                ]
            })
            st.dataframe(metrics_df)
        
        with col2:
            st.subheader("Efficient Frontier")
            fig = go.Figure()
            
            # Plot random portfolios
            fig.add_trace(go.Scatter(
                x=volatilities,
                y=returns,
                mode='markers',
                name='Random Portfolios',
                marker=dict(
                    size=5,
                    color='lightblue',
                    opacity=0.6
                )
            ))
            
            # Plot optimal portfolio
            fig.add_trace(go.Scatter(
                x=[opt_vol],
                y=[opt_ret],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star'
                )
            ))
            
            fig.update_layout(
                title='Efficient Frontier',
                xaxis_title='Annual Volatility',
                yaxis_title='Expected Annual Return',
                showlegend=True,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Display correlation matrix heatmap
        st.subheader("Correlation Matrix")
        correlation_matrix = optimizer.returns.corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=tickers,
            y=tickers,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        fig_corr.update_layout(
            title='Stock Correlation Matrix',
            template='plotly_white'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Display historical prices
        st.subheader("Historical Prices")
        fig_prices = go.Figure()
        for ticker in tickers:
            normalized_price = optimizer.data[ticker] / optimizer.data[ticker].iloc[0] * 100
            fig_prices.add_trace(go.Scatter(
                x=optimizer.data.index,
                y=normalized_price,
                name=ticker,
                mode='lines'
            ))
        fig_prices.update_layout(
            title='Normalized Stock Prices (Base=100)',
            xaxis_title='Date',
            yaxis_title='Price (Normalized)',
            template='plotly_white'
        )
        st.plotly_chart(fig_prices, use_container_width=True)

if __name__ == "__main__":
    main()