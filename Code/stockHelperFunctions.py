import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Global variables for multiple models
models = {}
scalers = {}
feature_names = []

# SET YOUR ALPHA VANTAGE API KEY HERE
ALPHA_VANTAGE_API_KEY = "FAKELB8GSSKL5EAL"

    # Create output file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"weekly_stock_prediction_analysis_{timestamp}.txt"

class OutputLogger:
    """Class to write output to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Redirect output to both console and file
logger = OutputLogger(output_file)

# ALL 42 SYMBOLS - NO CONFUSION, JUST THE LIST
ALL_42_SYMBOLS = [
    # Major Indices (3)
    "SPY", "QQQ", "IWM",
    
    # Mega Cap Stocks (3)  
    "AAPL", "MSFT", "GOOGL",
    
    # Growth Stocks (3)
    "NFLX", "CRM", "ADBE",
    
    # Dividend Stocks (3)
    "KO", "PEP", "WMT",
    
    # Sector ETFs (3)
    "XLK", "XLF", "XLV",
    
    # International ETFs (3)
    "VEA", "VWO", "IEFA",
    
    # Bond ETFs (3)
    "AGG", "BND", "TLT",
    
    # Commodity ETFs (3)
    "GLD", "SLV", "USO",
    
    # Target Date Funds (3)
    "FDKLX", "FFIJX", "FDVLX",
    
    # Fidelity Funds (3) - Your working ones!
    "FXAIX", "FXNAX", "FZROX",
    
    # Vanguard Funds (3)
    "VTSAX", "VTIAX", "VBTLX",
    
    # Schwab Funds (3)
    "SWPPX", "SWTSX", "SWISX",
    
    # REITs (3)
    "VNQ", "XLRE", "IYR",
    
    # Tech Stocks (3)
    "ORCL", "INTC", "AMD"
]

def get_stock_data_hybrid(symbol, period="5y", api_key=None):
    """
    Get stock data using yfinance first, fallback to Alpha Vantage for mutual funds
    """
    
    # First try yfinance
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if not data.empty and len(data) > 50:  # Need minimum data for ML
            return data, "yfinance"
        else:
            print(f"yfinance returned insufficient data ({len(data) if not data.empty else 0} rows)")
    except Exception as e:
        print(f"yfinance failed: {e}")
    
    # If yfinance fails, try Alpha Vantage
    if api_key and api_key != "YOUR_API_KEY_HERE":
        print(f"Trying Alpha Vantage for {symbol}...")
        
        # Try different time series functions
        functions = [
            ("TIME_SERIES_DAILY", "Time Series (Daily)"),
            ("TIME_SERIES_WEEKLY", "Weekly Time Series"), 
            ("TIME_SERIES_MONTHLY", "Monthly Time Series")
        ]
        
        for function, series_key in functions:
            try:
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': function,
                    'symbol': symbol,
                    'apikey': api_key,
                    'outputsize': 'full'  # Get more historical data
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                
                # Check for errors
                if "Error Message" in data:
                    print(f"Alpha Vantage error: {data['Error Message']}")
                    continue
                elif "Note" in data:
                    print(f"Alpha Vantage rate limit: {data['Note']}")
                    time.sleep(60)  # Wait if rate limited
                    continue
                
                if series_key in data:
                    time_series = data[series_key]
                    
                    # Convert to pandas DataFrame
                    df_data = []
                    for date_str, values in time_series.items():
                        try:
                            df_data.append({
                                'Date': pd.to_datetime(date_str),
                                'Open': float(values['1. open']),
                                'High': float(values['2. high']),
                                'Low': float(values['3. low']),
                                'Close': float(values['4. close']),
                                'Volume': int(float(values.get('5. volume', '0')))
                            })
                        except (ValueError, KeyError) as e:
                            continue  # Skip malformed data points
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.set_index('Date', inplace=True)
                        df.sort_index(inplace=True)  # Sort by date ascending
                        
                        if len(df) > 50:  # Need minimum data for ML
                            return df, "alpha_vantage"
                
                # Rate limiting for Alpha Vantage free tier
                time.sleep(12)  # 5 calls per minute limit
                
            except Exception as e:
                print(f"Alpha Vantage {function} failed: {e}")
                continue
    
    print(f"❌ Could not get data for {symbol} from any source")
    return None, None

def get_stock_data(symbol, period="5y"):
    """Updated version that uses hybrid approach"""
    data, source = get_stock_data_hybrid(symbol, period, ALPHA_VANTAGE_API_KEY)
    
    if data is not None:
        return data
    else:
        print(f"No data found for {symbol}")
        return None

def calculate_features(data):
    """Calculate improved, robust technical indicators"""
    df = data.copy()
    
    # Price-based features (more robust)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_5'] = df['Close'].ewm(span=5).mean()
    df['EMA_10'] = df['Close'].ewm(span=10).mean()
    
    # Normalized price ratios (prevent extreme values)
    df['Price_SMA5_Ratio'] = np.clip(df['Close'] / df['SMA_5'], 0.8, 1.2)
    df['Price_SMA10_Ratio'] = np.clip(df['Close'] / df['SMA_10'], 0.8, 1.2)
    df['Price_EMA5_Ratio'] = np.clip(df['Close'] / df['EMA_5'], 0.8, 1.2)
    
    # Returns (capped to prevent outliers)
    df['Return_1d'] = np.clip(df['Close'].pct_change(1), -0.2, 0.2)
    df['Return_3d'] = np.clip(df['Close'].pct_change(3), -0.3, 0.3)
    df['Return_5d'] = np.clip(df['Close'].pct_change(5), -0.4, 0.4)
    
    # Rolling statistics
    df['Return_Mean_5'] = df['Return_1d'].rolling(window=5).mean()
    df['Return_Std_5'] = df['Return_1d'].rolling(window=5).std()
    df['Return_Mean_10'] = df['Return_1d'].rolling(window=10).mean()
    
    # Volume features (handle mutual funds with zero volume)
    if df['Volume'].sum() > 0:  # Only if volume data exists
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = np.clip(df['Volume'] / df['Volume_SMA_10'], 0.1, 5.0)
    else:
        df['Volume_Ratio'] = 1.0  # Neutral for mutual funds
    
    # Volatility (normalized)
    df['Volatility_5'] = np.clip(df['Return_1d'].rolling(window=5).std(), 0, 0.1)
    df['Volatility_10'] = np.clip(df['Return_1d'].rolling(window=10).std(), 0, 0.1)
    
    # Momentum indicators
    df['Price_Change_5d'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
    df['Price_Change_10d'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
    df['Price_Change_5d'] = np.clip(df['Price_Change_5d'], -0.5, 0.5)
    df['Price_Change_10d'] = np.clip(df['Price_Change_10d'], -0.6, 0.6)
    
    # RSI-like indicator (simplified)
    gains = df['Return_1d'].where(df['Return_1d'] > 0, 0)
    losses = -df['Return_1d'].where(df['Return_1d'] < 0, 0)
    avg_gains = gains.rolling(window=14).mean()
    avg_losses = losses.rolling(window=14).mean()
    rs = avg_gains / (avg_losses + 1e-8)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)  # Neutral RSI
    
    # Replace any remaining infinite values with NaN, then forward fill
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')  # Backfill any remaining NaN at start
    
    return df

def prepare_data(symbol, prediction_days=5):  # Changed to 5 days (1 week)
    """Get data and prepare features with better preprocessing"""
    # Get stock data using hybrid approach
    data = get_stock_data(symbol)
    if data is None:
        return None, None
    
    # Calculate features
    df = calculate_features(data)
    
    # Create target: 1 if NEXT WEEK's price goes up, 0 if down
    df['Target'] = (df['Close'].shift(-prediction_days) > df['Close']).astype(int)
    
    # Select robust features (more features, but properly normalized)
    features = [
        'Price_SMA5_Ratio', 'Price_SMA10_Ratio', 'Price_EMA5_Ratio',
        'Return_1d', 'Return_3d', 'Return_5d',
        'Return_Mean_5', 'Return_Mean_10',
        'Volume_Ratio', 'Volatility_5', 'Volatility_10',
        'Price_Change_5d', 'Price_Change_10d', 'RSI'
    ]
    
    # Clean data
    df_clean = df[features + ['Target']].copy()
    
    # Check data quality
    nan_counts = df_clean[features].isna().sum()
    if nan_counts.sum() > 0:
        print(f"  NaN values per feature:")
        for feature in features:
            if nan_counts[feature] > 0:
                print(f"    {feature}: {nan_counts[feature]} NaN values")
    
    # Drop NaN values and rows where we can't predict (last 5 days)
    df_clean = df_clean.dropna()
    df_clean = df_clean.iloc[:-prediction_days]  # Remove last 5 days (can't predict their future)

    
    if df_clean.empty:
        print("  ❌ No valid data after processing")
        return None, None
    
    if len(df_clean) < 100:  # Increased minimum data requirement
        print(f"  ❌ Not enough clean data: only {len(df_clean)} rows (need at least 100)")
        return None, None
    
    X = df_clean[features]
    y = df_clean['Target']
    
    # Check target distribution
    target_dist = y.value_counts()
    if len(target_dist) < 2:
        print(f"  ❌ Target has only one class: {target_dist}")
        return None, None
    
    # Check if target is too imbalanced
    minority_ratio = min(target_dist) / len(y)
    if minority_ratio < 0.1:  # Less than 10% minority class
        print(f"  ⚠️  Warning: Highly imbalanced target (minority class: {minority_ratio:.1%})")
    
    return X, y

def train_model(symbol):
    """Train a robust prediction model for a specific symbol - WEEKLY PREDICTION"""
    global models, scalers, feature_names
    
    # Get data
    data = get_stock_data(symbol)
    if data is None:
        print(f"❌ Could not get data for {symbol}")
        return False
    
    X, y = prepare_data(symbol, prediction_days=5)  # Predict 1 week ahead
    if X is None:
        print(f"❌ Failed to prepare features for {symbol}")
        return False
    
    # Store feature names (should be same for all symbols)
    if not feature_names:
        feature_names = X.columns.tolist()
    
    try:
        # Time series split (more realistic for financial data)
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Use different train/test split ratio
        split_point = int(len(X) * 0.8)  # 80% train, 20% test
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]
        
        # Use RobustScaler instead of StandardScaler (better for outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Try multiple models and pick the best one
        models_to_try = {
            'rf_conservative': RandomForestClassifier(
                n_estimators=50,      # Reduced from 100
                max_depth=5,          # Limited depth to prevent overfitting
                min_samples_split=20, # Increased to prevent overfitting
                min_samples_leaf=10,  # Increased to prevent overfitting
                max_features='sqrt',  # Reduced feature sampling
                random_state=42,
                class_weight='balanced'  # Handle imbalanced data
            ),
            'rf_moderate': RandomForestClassifier(
                n_estimators=30,
                max_depth=3,
                min_samples_split=30,
                min_samples_leaf=15,
                max_features=0.5,
                random_state=42,
                class_weight='balanced'
            ),
            'logistic': LogisticRegression(
                C=1.0,               # Regularization
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        # Cross-validate each model
        for name, model in models_to_try.items():
            try:
                # Cross-validation with time series split
                cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                          cv=tscv, scoring='accuracy')
                avg_score = cv_scores.mean()
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    best_name = name
                    
            except Exception as e:
                print(f"  {name} failed: {e}")
                continue
        
        if best_model is None:
            print(f"❌ No model worked for {symbol}")
            return False
        
        # Train the best model
        best_model.fit(X_train_scaled, y_train)
        
        # Store the model and scaler
        models[symbol] = best_model
        scalers[symbol] = scaler
        
        # Evaluate
        train_accuracy = best_model.score(X_train_scaled, y_train)
        test_accuracy = best_model.score(X_test_scaled, y_test)
        
        
        # Target distribution in training
        target_dist = y_train.value_counts()
        
        # Feature importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return True
        
    except Exception as e:
        print(f"❌ Error training WEEKLY model for {symbol}: {e}")
        return False

def predict_stock(symbol):
    """Predict using the WEEKLY prediction model"""
    global models, scalers, feature_names
    
    if symbol not in models:
        return {"error": f"No model trained for {symbol}"}
    
    # Get recent data using hybrid approach
    data = get_stock_data(symbol, period="6mo")  # Use 6 months for prediction
    if data is None:
        return {"error": f"Could not get data for {symbol}"}
    
    # Calculate features
    df = calculate_features(data)
    latest = df[feature_names].iloc[-1:].fillna(0)

    
    # Use the symbol-specific model and scaler
    model = models[symbol]
    scaler = scalers[symbol]
    latest_scaled = scaler.transform(latest)
    
    prediction = model.predict(latest_scaled)[0]
    probabilities = model.predict_proba(latest_scaled)[0]
    
    return {
        "symbol": symbol,
        "prediction": "UP" if prediction == 1 else "DOWN",
        "confidence": max(probabilities),
        "probability_up": probabilities[1],
        "probability_down": probabilities[0],
        "current_price": float(data['Close'].iloc[-1]),
        "trained_on": f"{symbol} WEEKLY model",
        "raw_prediction": int(prediction),
        "features": dict(zip(feature_names, latest.iloc[0].values)),
        "prediction_horizon": "1 week"
    }

if __name__ == "__main__":
    # Start logging to file
    sys.stdout = logger

    print ()
    print("WEEKLY STOCK PREDICTION ANALYSIS")
    print ()
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {output_file}")
    print("=" * 80)
    
    # Test with ALL 42 symbols to validate fixes
    TEST_SYMBOLS = ALL_42_SYMBOLS  # Use all 42 symbols!
    print ()
    print("🚀Testing data availability...")
    print("=" * 70)
    
    # Test data availability
    available_symbols = []
    failed_symbols = []
    
    for i, symbol in enumerate(TEST_SYMBOLS, 1):
        data = get_stock_data(symbol)
        if data is not None:
            available_symbols.append(symbol)
        else:
            failed_symbols.append(symbol)
            print(f"❌ {symbol}: No data available")
        
        # Small delay between requests
        if i < len(TEST_SYMBOLS):
            time.sleep(1)
    
    print("DATA AVAILABILITY RESULTS")
    print(f"{'='*70}")
    print(f"✅ Available: {len(available_symbols)}/{len(TEST_SYMBOLS)} ({len(available_symbols)/len(TEST_SYMBOLS)*100:.1f}%)")
    
    if not available_symbols:
        print("❌ No symbols available for training")
        logger.close()
        exit()
    
    print(f"\n🚀Training prediction models for {len(available_symbols)} symbols...")
    print("=" * 70)
    
    successful_models = []
    
    for i, symbol in enumerate(available_symbols, 1):
        success = train_model(symbol)
        if success:
            successful_models.append(symbol)
        
        # Add delay between training
        if i < len(available_symbols):
            time.sleep(3)
    
    if not successful_models:
        print("❌ No FIXED models trained successfully")
        logger.close()
        exit()
    
    print(f"\n🚀Making predictions with models...")
    print("=" * 70)
    
    predictions = []
    for symbol in successful_models:
        try:
            result = predict_stock(symbol)
            if "error" not in result:
                predictions.append(result)
            else:
                print(f"Error with {symbol}: {result['error']}")
        except Exception as e:
            print(f"Error predicting {symbol}: {e}")
    
    # Display results
    up_predictions = []
    down_predictions = []
    
    print("MODEL PREDICTION RESULTS")
    print(f"{'='*70}")
    
    for pred in predictions:
        print(f"\n{pred['symbol']}:")
        print(f"  Prediction: {pred['prediction']}")
        print(f"  Confidence: {pred['confidence']:.1%}")
        print(f"  Current Price: ${pred['current_price']:.2f}")
        print(f"  Probability Up: {pred['probability_up']:.1%}")
        print(f"  Probability Down: {pred['probability_down']:.1%}")
        
        if pred['prediction'] == 'UP':
            up_predictions.append(pred['symbol'])
        else:
            down_predictions.append(pred['symbol'])
    
    # Final Summary
    print(f"\n{'='*70}")
    print("FIXED MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"📊 Total symbols analyzed: {len(predictions)}")
    print(f"📈 Predicted UP: {len(up_predictions)} symbols")
    if up_predictions:
        print(f"   {', '.join(up_predictions)}")
    print(f"📉 Predicted DOWN: {len(down_predictions)} symbols")  
    if down_predictions:
        print(f"   {', '.join(down_predictions)}")
    
    # Statistics
    if predictions:
        avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
        avg_prob_up = sum(p['probability_up'] for p in predictions) / len(predictions)
        avg_prob_down = sum(p['probability_down'] for p in predictions) / len(predictions)
        
        print(f"\n📊 MODEL STATISTICS:")
        print(f"   Average confidence: {avg_confidence:.1%}")
        print(f"   Average probability UP: {avg_prob_up:.1%}")
        print(f"   Average probability DOWN: {avg_prob_down:.1%}")
        print(f"   UP prediction rate: {len(up_predictions)/len(predictions)*100:.1f}%")
        print(f"   DOWN prediction rate: {len(down_predictions)/len(predictions)*100:.1f}%")
        
        # Check if we fixed the universal DOWN problem
    
    print(f"\n✅ Analysis completed and saved to {output_file}!")
    print("=" * 80)
    
    # Close the logger
    sys.stdout = logger.terminal
    logger.close()
