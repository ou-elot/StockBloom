import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set your API key
API_KEY = "FAKELB8GSSKL5EAL"

# Page configuration
st.set_page_config(
    page_title="StockBloom AI",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Custom CSS for warm pink aesthetic
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #FDF2F8 0%, #FCE7F3 50%, #FEF3E2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    .main-title {
        font-size: 5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #BE185D, #DC2626, #EA580C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-align: center;
        display: block;
        width: 100%;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        color: #A16207;
        margin-bottom: 0;
        text-align: center;
        display: block;
        width: 100%;
    }
    
    /* Card styling */
    .prediction-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 32px rgba(190, 24, 93, 0.1);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 48px rgba(190, 24, 93, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(254, 243, 226, 0.8), rgba(252, 231, 243, 0.8));
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.6);
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #BE185D, #DC2626) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(190, 24, 93, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(190, 24, 93, 0.4) !important;
    }
    
    /* Hide sidebar completely */
    .css-1d391kg, .css-1lcbmhc, .css-1y4p8pa {
        display: none !important;
    }
    
    /* Adjust main content to full width */
    .css-18e3th9, .css-1d391kg {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Input styling */
    .stTextArea > div > div > textarea {
        background: linear-gradient(135deg, rgba(254, 243, 226, 0.9), rgba(252, 231, 243, 0.9)) !important;
        border-radius: 12px !important;
        border: 1px solid #FECACA !important;
        color: #7C2D12 !important;
    }
    
    .stTextInput > div > div > input {
        background: linear-gradient(135deg, rgba(254, 243, 226, 0.9), rgba(252, 231, 243, 0.9)) !important;
        border-radius: 12px !important;
        border: 1px solid #FECACA !important;
        color: #7C2D12 !important;
    }
    
    /* Success/Error messages */
    .success-message {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 1rem;
        border-radius: 16px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
        padding: 1rem;
        border-radius: 16px;
        margin: 1rem 0;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #F472B6, #FBBF24) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: 1px solid #FECACA;
    }
    
    /* Multiselect styling */
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: 1px solid #FECACA;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hide header link buttons */
    .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a, .stMarkdown h4 a, .stMarkdown h5 a, .stMarkdown h6 a {
        display: none !important;
    }
    
    /* Hide any anchor links */
    .element-container .stMarkdown a[href^="#"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ALL 42 SYMBOLS from your original code
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
    # Fidelity Funds (3)
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

# S&P 500 symbols (complete list)
SP500_SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "TSLA", "META", "UNH", "JNJ", "JPM",
    "V", "PG", "XOM", "HD", "CVX", "MA", "BAC", "ABBV", "PFE", "AVGO",
    "KO", "MRK", "COST", "DIS", "TMO", "ACN", "DHR", "VZ", "MCD", "ABT",
    "CRM", "ADBE", "NKE", "TXN", "LLY", "NFLX", "QCOM", "WMT", "NEE", "PM",
    "COP", "UPS", "RTX", "LOW", "ORCL", "IBM", "HON", "SPGI", "INTU", "AMD",
    "T", "ELV", "CAT", "AMGN", "GS", "BKNG", "DE", "AXP", "NOW", "PLD",
    "SBUX", "TJX", "BLK", "MDT", "CVS", "ADP", "GILD", "LMT", "VRTX", "AMT",
    "SYK", "TMUS", "C", "SCHW", "MO", "ZTS", "CI", "CB", "MMC", "SO",
    "PYPL", "BSX", "DUK", "MDLZ", "ITW", "EOG", "PNC", "AON", "CL", "CME",
    "FDX", "WM", "ICE", "USB", "NSC", "GM", "SLB", "HUM", "GD", "TGT",
    "FCX", "BDX", "EMR", "MU", "CCI", "MCO", "PGR", "TFC", "REGN", "NOC",
    "APD", "CNC", "PSA", "FIS", "ECL", "MMM", "KLAC", "CARR", "EXC", "COF",
    "NUE", "EW", "AFL", "ORLY", "ROP", "CTSH", "ALL", "WBA", "ADI", "AIG",
    "STZ", "GWW", "KMB", "PAYX", "O", "HLT", "BIIB", "MSI", "CTAS", "KMI",
    "SHW", "FAST", "WELL", "AZO", "PSX", "OTIS", "CMG", "SPG", "EA", "IQV",
    "BK", "IDXX", "PPG", "DD", "PRU", "ETN", "DXCM", "MNST", "KDP", "ILMN",
    "WMB", "RSG", "GLW", "YUM", "A", "KEYS", "VRSK", "DLTR", "GIS", "CTVA",
    "TEL", "ROK", "TSCO", "EBAY", "EXR", "CSGP", "ODFL", "ZBH", "HSY", "VRSN",
    "DAL", "ROST", "AVB", "ANSS", "CPRT", "HPQ", "NTAP", "STE", "ALGN", "EFX",
    "FTV", "MPWR", "MTB", "ESS", "FANG", "VMC", "MLM", "ARE", "LH", "IT",
    "STT", "FITB", "CDW", "WST", "WDC", "HBAN", "CHD", "RMD", "CERN", "UAL",
    "AWK", "BRO", "SIVB", "SBAC", "CF", "SWK", "CMS", "LDOS", "NTRS", "RF",
    "WAB", "KEY", "ULTA", "ANET", "AMCR", "INVH", "MAA", "ZBRA", "ETSY", "EXPD",
    "TTWO", "CBOE", "TDY", "FRC", "STLD", "TER", "CLX", "SWKS", "K", "AKAM",
    "FE", "ENPH", "HOLX", "EXPE", "TROW", "EQR", "PKI", "POOL", "AVY", "WRB",
    "CAH", "GPN", "UDR", "IEX", "PAYC", "JBHT", "NDAQ", "DTE", "TECH", "PEAK",
    "COO", "MRNA", "LYV", "DOV", "TPG", "RJF", "CTLT", "SMCI", "IP", "VTRS",
    "JKHY", "J", "SYF", "BR", "MKC", "EVRG", "CHRW", "CINF", "PFG", "WAT",
    "BALL", "GRMN", "LRCX", "CE", "AEE", "APTV", "LVS", "TYL", "INCY", "FFIV",
    "WY", "LNT", "BBWI", "ROL", "ATO", "DPZ", "ALLE", "EIX", "XEL", "MKTX",
    "PKG", "EMN", "HSIC", "DGX", "FMC", "NTAP", "TXT", "CNP", "TRMB", "CRL",
    "L", "VTRS", "EPAM", "WYNN", "RCL", "KIM", "IRM", "SEDG", "AES", "FBHS",
    "LDOS", "TAP", "MGM", "FOXA", "NCLH", "PNR", "ZION", "RE", "DISH", "LKQ",
    "BWA", "MOS", "NWSA", "TPR", "WRK", "AIZ", "NLSN", "PARA", "DVN", "HII",
    "XRAY", "CPB", "HRL", "DXC", "SJM", "FLT", "MTCH", "NVR", "JNPR", "MCHP",
    "GL", "ABMD", "GNRC", "PHM", "BEN", "HAS", "REG", "VNO", "CZR", "AAP",
    "IVZ", "NRG", "WHR", "BF.B", "PBCT", "SEE", "NDSN", "ALB", "LNC", "FLIR",
    "ALK", "GPS", "UAA", "NWL", "IPG", "NOV", "PVH", "PRGO", "PENN", "NWS",
    "HBI", "VFC", "UHS", "RL", "UA", "FOX", "DISH", "COG", "BXP", "CMA",
    "WU", "AAL", "XRAY", "SLG", "MAC", "NOV", "DVA", "VTR", "NLOK", "FTI",
    "KSS", "AOS", "MHK", "PBCT", "PWR", "FLIR", "HFC", "DISCK", "NLSN", "DISH"
]

# Cache functions for better performance
@st.cache_data
def get_sp500_symbols():
    """Get current S&P 500 symbols from yfinance"""
    try:
        # Get S&P 500 symbols from yfinance
        sp500 = yf.Ticker("^GSPC")
        # Alternative: Use a more reliable method to get S&P 500 constituents
        
        # For now, let's use a direct approach - fetch from a reliable financial data source
        # We'll use the Wikipedia approach but with better error handling
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_df = tables[0]  # First table contains current S&P 500 companies
        
        # Find the symbol column (could be 'Symbol', 'Ticker', etc.)
        symbol_column = None
        for col in sp500_df.columns:
            if 'symbol' in str(col).lower():
                symbol_column = col
                break
        
        if symbol_column is None:
            symbol_column = sp500_df.columns[0]  # Use first column as fallback
            
        symbols = sp500_df[symbol_column].dropna().astype(str).str.strip().tolist()
        
        # Clean symbols - remove any invalid entries
        symbols = [s.replace('.', '-') for s in symbols if s and len(s) <= 6 and s.isalpha()]
        
        return symbols
        
    except Exception as e:
        st.error(f"Error fetching S&P 500 list: {e}")
        # More comprehensive fallback list of current major S&P 500 stocks
        return [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "BRK-B", "UNH",
            "JNJ", "JPM", "V", "PG", "XOM", "HD", "CVX", "MA", "BAC", "ABBV",
            "PFE", "AVGO", "KO", "MRK", "COST", "DIS", "TMO", "ACN", "DHR", "VZ",
            "MCD", "ABT", "CRM", "ADBE", "NKE", "TXN", "LLY", "NFLX", "QCOM", "WMT",
            "NEE", "PM", "COP", "UPS", "RTX", "LOW", "ORCL", "IBM", "HON", "SPGI"
        ]

@st.cache_data
def get_company_name(symbol):
    """Get company name for a given symbol"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return info.get('longName', info.get('shortName', symbol))
    except:
        return symbol

@st.cache_data
def get_stock_data(symbol, period="3y"):
    """Get stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if not data.empty and len(data) > 30:
            return data
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

@st.cache_data
def calculate_features(data):
    """Calculate technical indicators"""
    df = data.copy()
    
    # Price-based features
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_5'] = df['Close'].ewm(span=5).mean()
    df['EMA_10'] = df['Close'].ewm(span=10).mean()
    
    # Normalized price ratios
    df['Price_SMA5_Ratio'] = np.clip(df['Close'] / df['SMA_5'], 0.8, 1.2)
    df['Price_SMA10_Ratio'] = np.clip(df['Close'] / df['SMA_10'], 0.8, 1.2)
    df['Price_EMA5_Ratio'] = np.clip(df['Close'] / df['EMA_5'], 0.8, 1.2)
    
    # Returns
    df['Return_1d'] = np.clip(df['Close'].pct_change(1), -0.2, 0.2)
    df['Return_3d'] = np.clip(df['Close'].pct_change(3), -0.3, 0.3)
    df['Return_5d'] = np.clip(df['Close'].pct_change(5), -0.4, 0.4)
    
    # Rolling statistics
    df['Return_Mean_5'] = df['Return_1d'].rolling(window=5).mean()
    df['Return_Std_5'] = df['Return_1d'].rolling(window=5).std()
    df['Return_Mean_10'] = df['Return_1d'].rolling(window=10).mean()
    
    # Volume features
    if df['Volume'].sum() > 0:
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = np.clip(df['Volume'] / df['Volume_SMA_10'], 0.1, 5.0)
    else:
        df['Volume_Ratio'] = 1.0
    
    # Volatility
    df['Volatility_5'] = np.clip(df['Return_1d'].rolling(window=5).std(), 0, 0.1)
    df['Volatility_10'] = np.clip(df['Return_1d'].rolling(window=10).std(), 0, 0.1)
    
    # Momentum indicators
    df['Price_Change_5d'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
    df['Price_Change_10d'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
    df['Price_Change_5d'] = np.clip(df['Price_Change_5d'], -0.5, 0.5)
    df['Price_Change_10d'] = np.clip(df['Price_Change_10d'], -0.6, 0.6)
    
    # RSI-like indicator
    gains = df['Return_1d'].where(df['Return_1d'] > 0, 0)
    losses = -df['Return_1d'].where(df['Return_1d'] < 0, 0)
    avg_gains = gains.rolling(window=14).mean()
    avg_losses = losses.rolling(window=14).mean()
    rs = avg_gains / (avg_losses + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def prepare_data(symbol, prediction_days=5):
    """Prepare data for training"""
    data = get_stock_data(symbol)
    if data is None:
        return None, None
    
    df = calculate_features(data)
    
    # Create target
    df['Target'] = (df['Close'].shift(-prediction_days) > df['Close']).astype(int)
    
    # Select features
    features = [
        'Price_SMA5_Ratio', 'Price_SMA10_Ratio', 'Price_EMA5_Ratio',
        'Return_1d', 'Return_3d', 'Return_5d',
        'Return_Mean_5', 'Return_Mean_10',
        'Volume_Ratio', 'Volatility_5', 'Volatility_10',
        'Price_Change_5d', 'Price_Change_10d', 'RSI'
    ]
    
    # Clean data
    df_clean = df[features + ['Target']].copy()
    df_clean = df_clean.dropna()
    df_clean = df_clean.iloc[:-prediction_days]
    
    if df_clean.empty or len(df_clean) < 75:
        return None, None
    
    X = df_clean[features]
    y = df_clean['Target']
    
    return X, y

def train_model(symbol, progress_bar=None):
    """Train prediction model"""
    X, y = prepare_data(symbol)
    if X is None:
        return None, None, None
    
    try:
        # Split data
        split_point = int(len(X) * 0.8)
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if progress_bar:
            progress_bar.progress(50)
        
        # Try different models
        models_to_try = {
            'rf_conservative': RandomForestClassifier(
                n_estimators=50, max_depth=5, min_samples_split=20,
                min_samples_leaf=10, max_features='sqrt', random_state=42,
                class_weight='balanced'
            ),
            'logistic': LogisticRegression(
                C=1.0, max_iter=1000, random_state=42, class_weight='balanced'
            )
        }
        
        best_model = None
        best_score = 0
        
        # Cross-validate models
        tscv = TimeSeriesSplit(n_splits=3)
        for name, model in models_to_try.items():
            try:
                scores = []
                for train_idx, val_idx in tscv.split(X_train_scaled):
                    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    model.fit(X_tr, y_tr)
                    score = model.score(X_val, y_val)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
            except:
                continue
        
        if best_model is None:
            return None, None, None
        
        if progress_bar:
            progress_bar.progress(80)
        
        # Train final model
        best_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_accuracy = best_model.score(X_train_scaled, y_train)
        test_accuracy = best_model.score(X_test_scaled, y_test)
        
        if progress_bar:
            progress_bar.progress(100)
        
        return best_model, scaler, {'train_acc': train_accuracy, 'test_acc': test_accuracy}
        
    except Exception as e:
        st.error(f"Error training model for {symbol}: {e}")
        return None, None, None

def predict_stock(symbol, model, scaler):
    """Make prediction for a symbol"""
    data = get_stock_data(symbol, period="3y")
    if data is None:
        return None
    
    df = calculate_features(data)
    
    features = [
        'Price_SMA5_Ratio', 'Price_SMA10_Ratio', 'Price_EMA5_Ratio',
        'Return_1d', 'Return_3d', 'Return_5d',
        'Return_Mean_5', 'Return_Mean_10',
        'Volume_Ratio', 'Volatility_5', 'Volatility_10',
        'Price_Change_5d', 'Price_Change_10d', 'RSI'
    ]
    
    latest = df[features].iloc[-1:].fillna(0)
    latest_scaled = scaler.transform(latest)
    
    prediction = model.predict(latest_scaled)[0]
    probabilities = model.predict_proba(latest_scaled)[0]
    
    return {
        "symbol": symbol,
        "prediction": "UP" if prediction == 1 else "DOWN",
        "confidence": max(probabilities),
        "probability_up": probabilities[1] if len(probabilities) > 1 else 0.5,
        "probability_down": probabilities[0] if len(probabilities) > 1 else 0.5,
        "current_price": float(data['Close'].iloc[-1]),
        "raw_prediction": int(prediction)
    }

def create_prediction_chart(predictions_df):
    """Create visualization of predictions"""
    fig = go.Figure()
    
    # Add bars for confidence levels
    colors = ['#10B981' if pred == 'UP' else '#EF4444' for pred in predictions_df['prediction']]
    
    fig.add_trace(go.Bar(
        x=predictions_df['symbol'],
        y=predictions_df['confidence'],
        marker_color=colors,
        text=[f"{conf:.1%}" for conf in predictions_df['confidence']],
        textposition='auto',
        name='Confidence Level'
    ))
    
    fig.update_layout(
        title="Prediction Confidence Levels",
        xaxis_title="Symbols",
        yaxis_title="Confidence",
        template="plotly_white",
        height=400,
        font=dict(family="Inter", size=12),
        plot_bgcolor='rgba(255,255,255,0.8)',
        paper_bgcolor='rgba(255,255,255,0.8)'
    )
    
    return fig

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">StockBloom 🌸</h1>
        <p class="main-subtitle">Beautiful predictions for your investment journey</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["🌤️ Weekly Garden Forecast", "🍂 Harvest Season"])
    
    with tab1:
        show_garden_overview()
    
    with tab2:
        show_harvest_season()

def show_garden_overview():
    """Weekly Garden Forecast page - current functionality"""
    
    # Main input section with rounded rectangle
    # Create the input container
    with st.container():
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 24px;
            padding: 0.6rem;
            margin: 0.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.8);
            box-shadow: 0 8px 32px rgba(190, 24, 93, 0.1);
        ">
            <h3 style="color: #7C2D12; margin-bottom: 0rem; text-align: center;">
                🌸 Bloom Your Portfolio
            </h3>
            <p style="color: #A16207; font-size: 0.9rem; text-align: center; margin: 0;">
                Plant. Grow. Prosper.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # NEW INSTRUCTIONS BLOCK
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(254, 243, 226, 0.9), rgba(252, 231, 243, 0.9)); backdrop-filter: blur(10px); border-radius: 20px; padding: 1rem; margin: 1rem 0; border: 1px solid rgba(255, 255, 255, 0.8); box-shadow: 0 6px 24px rgba(190, 24, 93, 0.08);">
            <div style="text-align: center; margin-bottom: 0.4rem;">
                <h4 style="color: #7C2D12; margin: 0; font-size: 1.2rem; font-weight: 600;">🌱 How to Get Started</h4>
            </div>
            <div style="text-align: center; margin-bottom: 0.4rem;">
                <h5 style="color: #7C2D12; margin: 0; font-size: 1rem; font-weight: 600;">Either:</h5>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-bottom: 0.4rem;">
                <div style="text-align: center;">
                    <div style="background: rgba(255, 255, 255, 0.7); border-radius: 12px; padding: 0.8rem; height: 100%; border: 1px solid rgba(255, 255, 255, 0.5);">
                        <div style="font-size: 1.4rem; margin-bottom: 0.3rem;">📝</div>
                        <h5 style="color: #7C2D12; margin: 0.3rem 0; font-size: 1rem;">Enter Stock Symbols</h5>
                        <p style="color: #A16207; margin: 0; font-size: 0.9rem; line-height: 1.3;">Type any stock symbols separated by commas, spaces, or semicolons (e.g., AAPL, MSFT, GOOGL)</p>
                    </div>
                </div>
                <div style="text-align: center;">
                    <div style="background: rgba(255, 255, 255, 0.7); border-radius: 12px; padding: 0.8rem; height: 100%; border: 1px solid rgba(255, 255, 255, 0.5);">
                        <div style="font-size: 1.4rem; margin-bottom: 0.3rem;">📄</div>
                        <h5 style="color: #7C2D12; margin: 0.3rem 0; font-size: 1rem;">Upload CSV File</h5>
                        <p style="color: #A16207; margin: 0; font-size: 0.9rem; line-height: 1.3;">Upload a CSV with stock symbols in any column. Perfect for portfolio analysis!</p>
                    </div>
                </div>
            </div>
            <div style="text-align: center; margin-bottom: 0.4rem;">
                <h5 style="color: #7C2D12; margin: 0; font-size: 1rem; font-weight: 600;">Then:</h5>
            </div>
            <div style="text-align: center; margin-bottom: 0.8rem;">
                <div style="background: rgba(255, 255, 255, 0.7); border-radius: 12px; padding: 0.8rem; border: 1px solid rgba(255, 255, 255, 0.5); width: 100%;">
                    <div style="font-size: 1.4rem; margin-bottom: 0.3rem;">🌸</div>
                    <h5 style="color: #7C2D12; margin: 0.3rem 0; font-size: 1rem;">Get Predictions</h5>
                    <p style="color: #A16207; margin: 0; font-size: 0.9rem; line-height: 1.3;">Click "Grow Your Garden" to generate AI-powered predictions for your stocks</p>
                </div>
            </div>
            <div style="text-align: center; padding: 0.6rem; background: rgba(255, 255, 255, 0.5); border-radius: 10px;">
                <p style="color: #A16207; margin: 0; font-size: 0.85rem;"><strong>💡 Pro Tip:</strong> Try our preset buttons for quick analysis of popular stocks and ETFs!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Two main input methods side by side
        col1, col2 = st.columns([1, 1])
        
        # Symbol Entry Box
        with col1:
            st.markdown("""
            <div style="
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 1.5rem;
                margin: 1rem 0;
                border: 1px solid rgba(255, 255, 255, 0.8);
                box-shadow: 0 6px 24px rgba(190, 24, 93, 0.08);
                height: 60px;
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                <h4 style="color: #7C2D12; margin: 0; text-align: center;">
                    📝 Enter Stock Symbols
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            symbol_input = st.text_area(
                "",
                placeholder="Enter symbols separated by commas, spaces, or semicolons (e.g., AAPL MSFT GOOGL or AAPL, MSFT, GOOGL or AAPL; MSFT; GOOGL)",
                height=100,
                help="Enter stock symbols separated by commas, spaces, or semicolons. You can use any valid ticker symbols.",
                key="symbol_input_area",
                on_change=None  # Remove any callback to make it more responsive
            )
            
            # Quick select buttons
            st.markdown("**Or choose a preset:**")
            col_preset1, col_preset2 = st.columns(2)
            
            with col_preset1:
                if st.button("🔥 Popular", use_container_width=True):
                    st.session_state['symbol_input'] = "AAPL, MSFT, GOOGL, AMZN, TSLA, SPY, QQQ"
            
            with col_preset2:
                if st.button("📊 ETFs", use_container_width=True):
                    st.session_state['symbol_input'] = "SPY, QQQ, IWM, VTI, VEA, VWO"
            
            # Add spacing to align with CSV column
            st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
        
        # CSV Upload Box
        with col2:
            st.markdown("""
            <div style="
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 1.5rem;
                margin: 1rem 0;
                border: 1px solid rgba(255, 255, 255, 0.8);
                box-shadow: 0 6px 24px rgba(190, 24, 93, 0.08);
                height: 60px;
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                <h4 style="color: #7C2D12; margin: 0; text-align: center;">
                    📄 Upload CSV File
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "",
                type=['csv'],
                help="Upload a CSV file with stock symbols. The CSV should have a column containing ticker symbols.",
                key="csv_uploader"
            )
            
            if uploaded_file is not None:
                try:
                    # Read the CSV file
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded CSV with {len(df)} rows")
                    
                    # Show column selection
                    st.markdown("**Select the column containing symbols:**")
                    symbol_column = st.selectbox(
                        "",
                        options=df.columns.tolist(),
                        key="symbol_column_select"
                    )
                    
                    if symbol_column and st.button("🔄 Load Symbols from CSV", use_container_width=True):
                        # Extract symbols from the selected column
                        symbols_from_csv = df[symbol_column].dropna().astype(str).str.strip().str.upper().tolist()
                        symbols_from_csv = [s for s in symbols_from_csv if s and len(s) <= 6]  # Basic symbol validation
                        
                        if symbols_from_csv:
                            st.session_state['symbol_input'] = ", ".join(symbols_from_csv)
                            st.success(f"📈 Loaded {len(symbols_from_csv)} symbols from CSV!")
                        else:
                            st.error("No valid symbols found in the selected column.")
                            
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
            
            # Info box for CSV upload
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(254, 243, 226, 0.8), rgba(252, 231, 243, 0.8));
                border-radius: 16px;
                padding: 1rem;
                margin-top: 1rem;
                border: 1px solid rgba(255, 255, 255, 0.6);
            ">
                <p style="color: #A16207; font-size: 0.85rem; margin: 0;">
                    <strong>💡 CSV Tips:</strong><br>
                    • Include symbols in any column<br>
                    • One symbol per row<br>
                    • Headers are needed<br>
                    • Max 6 characters per symbol
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Update symbol_input if preset was clicked or if there's text in the area
        if 'symbol_input' in st.session_state:
            symbol_input = st.session_state['symbol_input']
        
        # Parse symbols with multiple delimiters - this happens automatically on any change
        if symbol_input and symbol_input.strip():
            # Replace semicolons and multiple spaces with commas, then split
            normalized_input = symbol_input.replace(';', ',').replace(' ', ',')
            # Split by comma and clean up
            symbols = [s.strip().upper() for s in normalized_input.split(',') if s.strip()]
            symbols = [s for s in symbols if s and len(s) <= 10]  # Remove empty strings and limit length
        else:
            symbols = []
    
    # Show selected symbols
    if symbols:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(254, 243, 226, 0.8), rgba(252, 231, 243, 0.8));
            border-radius: 16px;
            padding: 1rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.6);
        ">
            <p style="color: #7C2D12; margin: 0;">
                <strong>📈 Ready to analyze:</strong> {', '.join(symbols)} ({len(symbols)} symbols)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate Predictions Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🪴 Grow Your Garden 🪴", use_container_width=True, type="primary"):
                
                st.markdown("### 🌱 Growing Your Predictions...")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                predictions = []
                models = {}
                scalers = {}
                
                total_symbols = len(symbols)
                
                for i, symbol in enumerate(symbols):
                    status_text.text(f"Training model for {symbol}... 🌸")
                    progress_bar.progress((i + 1) / total_symbols)
                    
                    # Train model
                    model, scaler, metrics = train_model(symbol)
                    
                    if model is not None:
                        models[symbol] = model
                        scalers[symbol] = scaler
                        
                        # Make prediction
                        pred = predict_stock(symbol, model, scaler)
                        if pred:
                            predictions.append(pred)
                    else:
                        st.warning(f"Could not create model for {symbol}")
                    
                    time.sleep(0.1)  # Small delay for UX
                
                progress_bar.empty()
                status_text.empty()
                
                if predictions:
                    st.success("🌺 Your Garden is Bloomed!")
                    
                    # Store in session state
                    st.session_state['predictions'] = predictions
                    st.session_state['models_trained'] = True
    
    # Display Results
    if 'predictions' in st.session_state and st.session_state['predictions']:
        predictions = st.session_state['predictions']
        
        # Individual predictions
        st.markdown("### 🌻 Your Beautiful Garden")
        
        cols = st.columns(3)
        for i, pred in enumerate(predictions):
            with cols[i % 3]:
                trend_color = "#10B981" if pred['prediction'] == "UP" else "#8B5A3C"
                trend_emoji = "🌺" if pred['prediction'] == "UP" else "🥀"
                trend_text = "Blooming" if pred['prediction'] == "UP" else "Wilting"
                
                # Get company name
                company_name = get_company_name(pred['symbol'])
                
                st.markdown(f"""
                <div class="prediction-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                        <div style="flex: 1; min-height: 40px;">
                            <h3 style="color: #7C2D12; margin: 0; font-size: 1rem; font-weight: 600; line-height: 1.2;">{pred['symbol']}</h3>
                            <p style="color: #A16207; margin: 0; font-size: 0.8rem; line-height: 1.2;">{company_name}</p>
                        </div>
                        <span style="background: {trend_color}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin-left: 8px; flex-shrink: 0;">
                            {trend_emoji} {trend_text}
                        </span>
                    </div>
                    <p style="font-size: 1.1rem; font-weight: 600; color: #7C2D12; margin-bottom: 1rem;">
                        ${pred['current_price']:.2f}
                    </p>
                    <div style="margin-bottom: 0.5rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                            <span style="font-size: 0.8rem; color: #A16207; font-weight: 500;">Growth Probability</span>
                            <span style="font-size: 0.8rem; color: #7C2D12; font-weight: 600;">{pred['probability_up']:.1%}</span>
                        </div>
                        <div style="background: #F3F4F6; border-radius: 10px; height: 8px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #F472B6, #FBBF24); height: 100%; width: {pred['probability_up']*100:.1f}%; border-radius: 10px; transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Download option
        st.markdown("### 📥 Download Your Predictions")
        
        # Create downloadable data
        download_data = pd.DataFrame(predictions)
        csv = download_data.to_csv(index=False)
        
        st.download_button(
            label="🌸 Download Predictions as CSV",
            data=csv,
            file_name=f"stockbloom_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Disclaimer
        st.markdown("""
        ---
        **🌸 Gentle Reminder:** These predictions are created with care using advanced algorithms for educational purposes. 
        Investing involves risk, and past performance doesn't guarantee future results. Please make thoughtful decisions 
        and consider professional advice.
        """)

def analyze_sp500_stocks():
    """Analyze S&P 500 stocks for Action Items"""
    if 'sp500_analysis' in st.session_state:
        return st.session_state['sp500_analysis']
    
    st.markdown("### 🔍 Analyzing S&P 500 Stocks...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    predictions = []
    total_symbols = len(SP500_SYMBOLS)
    batch_size = 10  # Process in smaller batches
    
    for batch_start in range(0, total_symbols, batch_size):  # Process all S&P 500 stocks
        batch_end = min(batch_start + batch_size, total_symbols)
        batch_symbols = SP500_SYMBOLS[batch_start:batch_end]
        
        for i, symbol in enumerate(batch_symbols):
            current_progress = (batch_start + i + 1) / total_symbols
            status_text.text(f"Analyzing {symbol}... ({batch_start + i + 1}/{total_symbols})")
            progress_bar.progress(current_progress)
            
            try:
                # Train model for this symbol
                model, scaler, metrics = train_model(symbol)
                
                if model is not None:
                    # Make prediction
                    pred = predict_stock(symbol, model, scaler)
                    if pred:
                        predictions.append(pred)
                
            except Exception as e:
                continue  # Skip symbols that fail
            
            time.sleep(0.05)  # Small delay to prevent overwhelming APIs
    
    progress_bar.empty()
    status_text.empty()
    
    # Cache results
    st.session_state['sp500_analysis'] = predictions
    st.session_state['sp500_analysis_time'] = datetime.now()
    
    return predictions

def show_harvest_season():
    """Harvest Season page with Plant Seeds and Pull Weeds sections using S&P 500"""
    
    # Check if we need to refresh analysis (cache for 1 hour)
    should_analyze = True
    if 'sp500_analysis' in st.session_state and 'sp500_analysis_time' in st.session_state:
        time_diff = datetime.now() - st.session_state['sp500_analysis_time']
        if time_diff.total_seconds() < 3600:  # 1 hour cache
            should_analyze = False
    
    if should_analyze:
        # Show refresh button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Analyze S&P 500 Stocks", use_container_width=True, type="primary"):
                if 'sp500_analysis' in st.session_state:
                    del st.session_state['sp500_analysis']
                predictions = analyze_sp500_stocks()
            else:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, rgba(254, 243, 226, 0.8), rgba(252, 231, 243, 0.8));
                    border-radius: 16px;
                    padding: 2rem;
                    margin: 2rem 0;
                    border: 1px solid rgba(255, 255, 255, 0.6);
                    text-align: center;
                ">
                    <h3 style="color: #7C2D12; margin-bottom: 1rem;">🍂 Ready for Harvest Season?</h3>
                    <p style="color: #A16207; margin: 0;">
                        Click the button above to analyze S&P 500 stocks and discover which ones to plant and which weeds to pull!
                    </p>
                </div>
                """, unsafe_allow_html=True)
                return
    else:
        predictions = st.session_state['sp500_analysis']
    
    if not predictions:
        st.error("No predictions available. Please try the analysis again.")
        return
    
    # Filter predictions
    plant_seeds = [p for p in predictions if p['probability_up'] > 0.55]
    pull_weeds = [p for p in predictions if p['probability_down'] > 0.55]
    
    # Summary stats
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(254, 243, 226, 0.8), rgba(252, 231, 243, 0.8));
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.6);
        text-align: center;
    ">
        <h3 style="color: #7C2D12; margin-bottom: 1rem;">📊 S&P 500 Market Analysis</h3>
        <div style="display: flex; justify-content: space-around;">
            <div>
                <h4 style="color: #10B981; margin: 0;">{len(plant_seeds)}</h4>
                <p style="color: #A16207; margin: 0; font-size: 0.9rem;">Seeds to Plant</p>
            </div>
            <div>
                <h4 style="color: #8B5A3C; margin: 0;">{len(pull_weeds)}</h4>
                <p style="color: #A16207; margin: 0; font-size: 0.9rem;">Weeds to Pull</p>
            </div>
            <div>
                <h4 style="color: #BE185D; margin: 0;">{len(predictions)}</h4>
                <p style="color: #A16207; margin: 0; font-size: 0.9rem;">Total Analyzed</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Plant These Seeds Section
    st.markdown("### 🌱 Plant These Seeds")
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(16, 185, 129, 0.2);
    ">
        <p style="color: #047857; margin: 0; font-size: 0.9rem;">
            <strong>🌟 High Growth Potential:</strong> These stocks have greater than 55% probability of increasing. Consider adding to your portfolio.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if plant_seeds:
        cols = st.columns(3)
        for i, pred in enumerate(plant_seeds):
            with cols[i % 3]:
                company_name = get_company_name(pred['symbol'])
                st.markdown(f"""
                <div style="
                    background: rgba(255, 255, 255, 0.9);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 1.5rem;
                    border: 2px solid #10B981;
                    box-shadow: 0 8px 32px rgba(16, 185, 129, 0.15);
                    margin-bottom: 1rem;
                ">
                    <div style="text-align: center; margin-bottom: 0.5rem;">
                        <span style="background: #10B981; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                            🌺 {pred['probability_up']:.1%} Probability
                        </span>
                    </div>
                    <h4 style="color: #7C2D12; margin: 0.5rem 0; font-size: 1.3rem; text-align: center; font-weight: 700;">{pred['symbol']}</h4>
                    <p style="color: #A16207; margin: 0; font-size: 0.8rem; text-align: center;">{company_name}</p>
                    <p style="font-size: 1.1rem; font-weight: 600; color: #047857; margin: 0.5rem 0; text-align: center;">
                        ${pred['current_price']:.2f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.9);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.6);
        ">
            <p style="color: #A16207; margin: 0;">
                🌿 No high-confidence growth opportunities found in your current analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Pull the Weeds Section
    st.markdown("### 🥀 Pull the Weeds")
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05));
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(239, 68, 68, 0.2);
    ">
        <p style="color: #B91C1C; margin: 0; font-size: 0.9rem;">
            <strong>⚠️ High Decline Risk:</strong> These stocks have greater than 55% probability of decreasing. Consider reducing exposure.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if pull_weeds:
        cols = st.columns(3)
        for i, pred in enumerate(pull_weeds):
            with cols[i % 3]:
                company_name = get_company_name(pred['symbol'])
                st.markdown(f"""
                <div style="
                    background: rgba(255, 255, 255, 0.9);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 1.5rem;
                    border: 2px solid #8B5A3C;
                    box-shadow: 0 8px 32px rgba(239, 68, 68, 0.15);
                    margin-bottom: 1rem;
                ">
                    <div style="text-align: center; margin-bottom: 0.5rem;">
                        <span style="background: #8B5A3C; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                            🥀 {pred['probability_down']:.1%} Probability
                        </span>
                    </div>
                    <h4 style="color: #7C2D12; margin: 0.5rem 0; font-size: 1.3rem; text-align: center; font-weight: 700;">{pred['symbol']}</h4>
                    <p style="color: #A16207; margin: 0; font-size: 0.8rem; text-align: center;">{company_name}</p>
                    <p style="font-size: 1.1rem; font-weight: 600; color: #B91C1C; margin: 0.5rem 0; text-align: center;">
                        ${pred['current_price']:.2f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.9);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.6);
        ">
            <p style="color: #A16207; margin: 0;">
                🌺 Great news! No high-risk declining stocks found in your current analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
