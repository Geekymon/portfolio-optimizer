import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def get_stock_data(ticker, start_date, end_date):
    """
    Download stock data using yfinance
    """
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock

def create_features(df):
    """
    Create technical indicators as features
    """
    # Calculate moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # Calculate price momentum
    df['momentum'] = df['Close'] - df['Close'].shift(5)
    
    # Calculate volatility
    df['volatility'] = df['Close'].rolling(window=20).std()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate percentage change
    df['price_change'] = df['Close'].pct_change()
    
    # Target variable - next day's closing price
    df['target'] = df['Close'].shift(-1)
    
    return df

def prepare_data(df):
    """
    Prepare data for modeling
    """
    # Drop rows with NaN values
    df = df.dropna()
    
    # Features for training
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                      'MA5', 'MA20', 'momentum', 'volatility', 'RSI', 'price_change']
    
    X = df[feature_columns]
    y = df['target']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                        test_size=0.2, 
                                                        shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    """
    Train XGBoost model
    """
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R-squared Score: {r2:.2f}')
    
    return predictions

def plot_predictions(y_test, predictions):
    """
    Plot actual vs predicted prices
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, predictions, label='Predicted')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Parameters
    ticker = "AAPL"  # Example with Apple stock
    start_date = "2020-01-01"
    end_date = "2024-02-05"
    
    # Get data
    df = get_stock_data(ticker, start_date, end_date)
    
    # Create features
    df = create_features(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    predictions = evaluate_model(model, X_test, y_test)
    
    # Plot results
    plot_predictions(y_test, predictions)
    
    # Make prediction for next day
    last_data = X_test[-1].reshape(1, -1)
    next_day_prediction = model.predict(last_data)[0]
    print(f"\nPredicted price for next day: ${next_day_prediction:.2f}")