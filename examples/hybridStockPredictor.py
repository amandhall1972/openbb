import numpy as np
import pandas as pd
import yfinance as yf
import unittest
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class HybridStockPredictor:
    """
    Implements a dual-layered quantitative trading architecture combining
    historical technical indicators with aggregated sentiment tracking for AAPL.
    """
    def __init__(self, ticker: str = "AAPL"):
        self.ticker = ticker
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10)
        self.feature_columns = []

    def fetch_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetches underlying raw price metrics from Yahoo Finance API."""
        df = yf.download(self.ticker, start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No market data returned for ticker: {self.ticker}")
        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates explicit momentum, trend, and volatility feature spaces."""
        df = df.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']

        # 1. Bollinger Bands (20-day SMA, 2 Standard Deviations)
        df['SMA_20'] = close.rolling(window=20).mean()
        df['STD_20'] = close.rolling(window=20).std()
        df['BB_Upper'] = df['SMA_20'] + (2 * df['STD_20'])
        df['BB_Lower'] = df['SMA_20'] - (2 * df['STD_20'])

        # 2. MACD Calculation
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # 3. Relative Strength Index (RSI - 14 Days)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))

        # 4. Average Directional Index (ADX - 14 Days)
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        plus_di = 100 * (pd.Series(plus_dm, index=high.index).rolling(window=14).mean() / (atr + 1e-9)).values
        minus_di = 100 * (pd.Series(minus_dm, index=high.index).rolling(window=14).mean() / (atr + 1e-9)).values
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        df['ADX'] = pd.Series(dx).rolling(window=14).mean().values

        # Additional complementary technical features to complete the 15-indicator footprint
        df['SMA_50'] = close.rolling(window=50).mean()
        df['EMA_10'] = close.ewm(span=10, adjust=False).mean()
        df['Momentum'] = close.diff(periods=5)
        df['ROC'] = ((close - close.shift(10)) / (close.shift(10) + 1e-9)) * 100
        df['Stochastic_K'] = ((close - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min() + 1e-9)) * 100
        df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
        df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
        df['Price_Dist_BB_Upper'] = df['BB_Upper'] - close
        df['Price_Dist_BB_Lower'] = close - df['BB_Lower']

        return df.dropna()

    def generate_synthetic_sentiment_stream(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Generates simulated daily aggregated Twitter/social sentiment data
        bounded mathematically between -1 (strongly bearish) and +1 (strongly bullish).
        """
        np.random.seed(42)
        # Generate raw daily scores with a slight positive structural bias matching modern tech equities
        daily_scores = np.random.normal(loc=0.08, scale=0.35, size=len(index))
        daily_scores = np.clip(daily_scores, -1.0, 1.0)

        sentiment_df = pd.DataFrame(data={'Social_Sentiment': daily_scores}, index=index)
        # Inject structural moving averages to capture persisting media narrative regimes
        sentiment_df['Sentiment_MA3'] = sentiment_df['Social_Sentiment'].rolling(window=3, min_periods=1).mean()
        return sentiment_df

    def process_data_pipeline(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Combines structural market metrics and sentiment inputs into unified datasets."""
        market_df = self.fetch_market_data(start_date, end_date)
        features_df = self.compute_technical_indicators(market_df)
        sentiment_df = self.generate_synthetic_sentiment_stream(features_df.index)

        merged_df = pd.concat([features_df, sentiment_df], axis=1)

        # Formulate directional target paradigm based on next-day Close price movement
        # Define: Buy (1) if Return > 0.5%, Sell (-1) if Return < -0.5%, Else Hold (0)
        next_day_return = merged_df['Close'].shift(-1) / merged_df['Close'] - 1
        merged_df['Target'] = np.where(next_day_return > 0.005, 1, np.where(next_day_return < -0.005, -1, 0))

        # Discard final row because its forward prediction target is undefined
        return merged_df.dropna()

    def train_engine(self, df: pd.DataFrame) -> dict:
        """Trains the prediction matrix and saves evaluation metrics."""
        self.feature_columns = [
            'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal', 'RSI', 'ADX',
            'SMA_50', 'EMA_10', 'Momentum', 'ROC', 'Stochastic_K', 'Stochastic_D',
            'Volume_SMA', 'Price_Dist_BB_Upper', 'Price_Dist_BB_Lower',
            'Social_Sentiment', 'Sentiment_MA3'
        ]

        X = df[self.feature_columns]
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        predictions = self.model.predict(X_test_scaled)

        return classification_report(y_test, predictions, output_dict=True, zero_division=0)

    def generate_signal(self, current_features: pd.DataFrame) -> dict:
        """Generates real-time standalone trade signals based on the latest feature vector."""
        if not self.feature_columns:
            raise RuntimeError("Model execution requires a preliminary training sequence.")

        input_data = current_features[self.feature_columns].tail(1)
        scaled_input = self.scaler.transform(input_data)
        prediction_code = self.model.predict(scaled_input)[0]
        probabilities = self.model.predict_proba(scaled_input)[0]

        signal_mapping = {1: "BUY", 0: "HOLD", -1: "SELL"}
        return {
            "Timestamp": input_data.index[0].strftime("%Y-%m-%d"),
            "Signal": signal_mapping[prediction_code],
            "Confidence_Metrics": {
                "Sell_Prob": round(probabilities[0], 4),
                "Hold_Prob": round(probabilities[1], 4),
                "Buy_Prob": round(probabilities[2], 4) if len(probabilities) > 2 else 0.0
            }
        }

# --- UNIT TESTING FRAMEWORK ---

class TestHybridSignalEngine(unittest.TestCase):
    """Rigorous unit testing covering technical feature calculation logic and matrix bounds."""

    def setUp(self):
        # Generate isolated deterministic test frames
        np.random.seed(10)
        dates = pd.date_range(start="2026-01-01", periods=60, freq="D")
        self.test_df = pd.DataFrame(index=dates)
        self.test_df['Close'] = 100.0 + np.cumsum(np.random.normal(0, 1.5, size=60))
        self.test_df['High'] = self.test_df['Close'] + np.random.uniform(0.5, 2.0, size=60)
        self.test_df['Low'] = self.test_df['Close'] - np.random.uniform(0.5, 2.0, size=60)
        self.test_df['Open'] = self.test_df['Close'].shift(1).fillna(100.0)
        self.test_df['Volume'] = np.random.randint(1000000, 5000000, size=60)
        self.predictor = HybridStockPredictor(ticker="AAPL")

    def test_technical_indicator_dimensions(self):
        processed_df = self.predictor.compute_technical_indicators(self.test_df)
        self.assertIn('RSI', processed_df.columns)
        self.assertIn('MACD', processed_df.columns)
        self.assertIn('BB_Upper', processed_df.columns)
        self.assertFalse(processed_df.isna().any().any(), "Processed data contains unhandled NaN values.")

    def test_rsi_bounds(self):
        processed_df = self.predictor.compute_technical_indicators(self.test_df)
        self.assertTrue((processed_df['RSI'] >= 0).all() and (processed_df['RSI'] <= 100).all(),
                        "RSI calculations broke mathematical logic boundaries [0, 100].")

    def test_sentiment_generation_bounds(self):
        dates = pd.date_range(start="2026-01-01", periods=10, freq="D")
        sentiment_df = self.predictor.generate_synthetic_sentiment_stream(dates)
        self.assertTrue((sentiment_df['Social_Sentiment'] >= -1.0).all() and (sentiment_df['Social_Sentiment'] <= 1.0).all(),
                        "Aggregated sentiment values overflow allowed boundaries.")

# --- PRODUCTION RUNNER ---

if __name__ == "__main__":
    print("[*] Initiating internal test execution suite...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHybridSignalEngine)
    test_result = unittest.TextTestRunner(verbosity=1).run(suite)

    if test_result.wasSuccessful():
        print("[+] Verification testing completed cleanly. Initiating production training pipeline...")
        engine = HybridStockPredictor(ticker="AAPL")

        # Process historical training window
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365)

        data_matrix = engine.process_data_pipeline(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        metrics = engine.train_engine(data_matrix)

        print(f"\n[+] Hybrid Model Training Success Details (Accuracy: {metrics.get('accuracy', 0.0):.4f})")
        print("-----------------------------------------------------------------------------")

        # Extract live real-time output signal
        live_signal = engine.generate_signal(data_matrix)
        print("\n[+] LIVE HYBRID DIRECTIONAL SIGNAL OUTPUT:")
        print(f"    Target Asset      : {engine.ticker}")
        print(f"    Execution Window  : {live_signal['Timestamp']}")
        print(f"    Signaling Verdict : {live_signal['Signal']}")
        print(f"    Model Confidence  : {live_signal['Confidence_Metrics']}")
