import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import requests
import json
import time
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Smart Stock Predictor", layout="wide", page_icon="📈")


class DataHandler:
    """Handles all data loading and processing operations"""

    @staticmethod
    @st.cache_data
    def load_stock_data(ticker, period="2y"):
        """Load historical stock data using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if data.empty:
                st.error(f"No data found for ticker {ticker}")
                return None, None

            # Get company info
            info = stock.info
            return data, info
        except Exception as e:
            st.error(f"Error loading data for {ticker}: {str(e)}")
            return None, None

    @staticmethod
    @st.cache_data
    def get_company_news(ticker):
        """Get recent news for the company."""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news[:5]  # Get top 5 news items
            return news
        except:
            return []

    @staticmethod
    @st.cache_data
    def get_insider_trading(ticker):
        """Get insider trading information."""
        try:
            stock = yf.Ticker(ticker)
            insider_purchases = stock.insider_purchases
            insider_transactions = stock.insider_transactions
            return insider_purchases, insider_transactions
        except:
            return None, None


class TechnicalIndicators:
    """Calculate various technical indicators"""

    @staticmethod
    def calculate_rsi(prices, window=14):
        """
        Calculate Relative Strength Index (RSI)
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        if len(prices) < window + 1:
            return pd.Series([50] * len(prices), index=prices.index)

        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses using exponential moving average
        avg_gains = gains.ewm(span=window).mean()
        avg_losses = losses.ewm(span=window).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_moving_averages(prices):
        """Calculate various moving averages"""
        return {
            'SMA_20': prices.rolling(window=20).mean(),
            'SMA_50': prices.rolling(window=50).mean(),
            'EMA_12': prices.ewm(span=12).mean(),
            'EMA_26': prices.ewm(span=26).mean()
        }

    @staticmethod
    def calculate_macd(prices):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line

        return {
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        }

    @staticmethod
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        return {
            'Upper': sma + (std * num_std),
            'Middle': sma,
            'Lower': sma - (std * num_std)
        }


class ModelTrainer:
    """Handles LSTM model training and predictions"""

    @staticmethod
    def preprocess_data(data, lookback_window=60, test_size=0.2):
        """Preprocess data for LSTM model."""
        prices = data['Close'].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(prices)

        X, y = [], []
        for i in range(lookback_window, len(scaled_data)):
            X.append(scaled_data[i - lookback_window:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, y_train, X_test, y_test, scaler

    @staticmethod
    def build_lstm_model(input_shape):
        """Build and compile LSTM model."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    @staticmethod
    @st.cache_data
    def train_lstm_model(X_train, y_train, X_test, y_test):
        """Train LSTM model with caching."""
        model = ModelTrainer.build_lstm_model((X_train.shape[1], 1))

        progress_placeholder = st.empty()

        class StreamlitCallback:
            def __init__(self, total_epochs, placeholder):
                self.total_epochs = total_epochs
                self.placeholder = placeholder
                self.current_epoch = 0

            def on_epoch_end(self, epoch, logs=None):
                self.current_epoch = epoch + 1
                progress = self.current_epoch / self.total_epochs
                self.placeholder.progress(progress,
                                          f"🧠 AI Learning... {self.current_epoch}/{self.total_epochs} rounds completed")

        callback = StreamlitCallback(30, progress_placeholder)

        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=30,
            validation_data=(X_test, y_test),
            verbose=0
        )

        progress_placeholder.empty()
        return model, history

    @staticmethod
    def predict_future_prices(model, scaler, recent_data, days_ahead=7):
        """Predict future stock prices for the next few days."""
        try:
            # Get the last 60 days of data for prediction
            last_60_days = recent_data['Close'].tail(60).values.reshape(-1, 1)
            last_60_days_scaled = scaler.transform(last_60_days)

            # Reshape for LSTM input
            X_future = last_60_days_scaled.reshape(1, 60, 1)

            # Predict next few days
            future_predictions = []
            current_input = X_future.copy()

            for _ in range(days_ahead):
                # Predict next day
                next_pred_scaled = model.predict(current_input, verbose=0)
                next_pred = scaler.inverse_transform(next_pred_scaled)[0][0]
                future_predictions.append(next_pred)

                # Update input for next prediction (rolling window)
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0] = next_pred_scaled[0][0]

            return future_predictions
        except Exception as e:
            return []


class AnalysisEngine:
    """Handles recommendation generation and analysis"""

    @staticmethod
    def generate_recommendation(current_price, predicted_prices, recent_data, company_info, rsi_current):
        """Generate buy/sell/hold recommendation based on predictions and technical analysis."""

        if not predicted_prices:
            return "HOLD", "Neutral", "Insufficient data for reliable recommendation", 0

        # Calculate prediction trend
        avg_predicted = np.mean(predicted_prices)
        price_change_pct = ((avg_predicted - current_price) / current_price) * 100

        # Technical indicators
        sma_20 = recent_data['Close'].rolling(20).mean().iloc[-1]
        sma_50 = recent_data['Close'].rolling(50).mean().iloc[-1]

        # Volume analysis
        volume_avg = recent_data['Volume'].tail(10).mean()
        current_volume = recent_data['Volume'].iloc[-1]
        volume_ratio = current_volume / volume_avg

        # Recent momentum
        recent_momentum = recent_data['Close'].pct_change().tail(5).mean() * 100

        # Volatility (risk factor)
        volatility = recent_data['Close'].pct_change().tail(20).std() * 100

        # Market cap consideration (larger companies are generally more stable)
        market_cap = company_info.get('marketCap', 0)
        is_large_cap = market_cap > 10e9  # > $10B

        # Scoring system
        score = 0
        reasons = []

        # RSI-based scoring
        if rsi_current < 30:
            score += 1.5
            reasons.append(f"RSI ({rsi_current:.1f}) indicates oversold conditions")
        elif rsi_current > 70:
            score -= 1.5
            reasons.append(f"RSI ({rsi_current:.1f}) indicates overbought conditions")
        elif 40 <= rsi_current <= 60:
            score += 0.5
            reasons.append(f"RSI ({rsi_current:.1f}) shows balanced momentum")

        # Prediction-based scoring
        if price_change_pct > 5:
            score += 2
            reasons.append(f"AI predicts {price_change_pct:.1f}% price increase")
        elif price_change_pct > 2:
            score += 1
            reasons.append(f"AI predicts {price_change_pct:.1f}% modest increase")
        elif price_change_pct < -5:
            score -= 2
            reasons.append(f"AI predicts {price_change_pct:.1f}% price decrease")
        elif price_change_pct < -2:
            score -= 1
            reasons.append(f"AI predicts {price_change_pct:.1f}% modest decrease")

        # Technical analysis scoring
        if current_price > sma_20 and sma_20 > sma_50:
            score += 1
            reasons.append("Price trending above moving averages")
        elif current_price < sma_20 and sma_20 < sma_50:
            score -= 1
            reasons.append("Price trending below moving averages")

        # Volume scoring
        if volume_ratio > 1.5:
            score += 0.5
            reasons.append("High trading volume shows strong interest")
        elif volume_ratio < 0.5:
            score -= 0.5
            reasons.append("Low trading volume shows weak interest")

        # Momentum scoring
        if recent_momentum > 1:
            score += 0.5
            reasons.append("Recent positive momentum")
        elif recent_momentum < -1:
            score -= 0.5
            reasons.append("Recent negative momentum")

        # Risk adjustment for volatility
        if volatility > 4 and not is_large_cap:
            score -= 0.5
            reasons.append("High volatility increases risk")

        # Generate recommendation
        if score >= 2.5:
            recommendation = "STRONG BUY"
            confidence = "High"
        elif score >= 1.5:
            recommendation = "BUY"
            confidence = "Strong"
        elif score >= 0.5:
            recommendation = "BUY"
            confidence = "Moderate"
        elif score <= -2.5:
            recommendation = "STRONG SELL"
            confidence = "High"
        elif score <= -1.5:
            recommendation = "SELL"
            confidence = "Strong"
        elif score <= -0.5:
            recommendation = "SELL"
            confidence = "Moderate"
        else:
            recommendation = "HOLD"
            confidence = "Neutral"

        reason_text = "; ".join(reasons[:3])  # Top 3 reasons

        return recommendation, confidence, reason_text, price_change_pct

    @staticmethod
    def check_ollama_status():
        """Check if Ollama is running and get available models."""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                return True, model_names
            else:
                return False, []
        except:
            return False, []

    @staticmethod
    def generate_enhanced_analysis(ticker, company_info, recent_data, news, insider_data,
                                   predictions, actual_prices, future_predictions=None,
                                   recommendation_data=None, technical_indicators=None):
        """Generate comprehensive analysis with technical indicators."""

        ollama_running, available_models = AnalysisEngine.check_ollama_status()

        if not ollama_running:
            return AnalysisEngine.generate_fallback_analysis(
                ticker, company_info, recent_data, news, predictions, actual_prices,
                future_predictions, recommendation_data, technical_indicators
            )

        # Enhanced Ollama analysis would go here
        # For now, return fallback analysis
        return AnalysisEngine.generate_fallback_analysis(
            ticker, company_info, recent_data, news, predictions, actual_prices,
            future_predictions, recommendation_data, technical_indicators
        )

    @staticmethod
    def generate_fallback_analysis(ticker, company_info, recent_data, news, predictions,
                                   actual_prices, future_predictions=None, recommendation_data=None,
                                   technical_indicators=None):
        """Generate enhanced analysis without Ollama."""

        # Calculate basic metrics
        latest_price = recent_data['Close'].iloc[-1]
        prev_price = recent_data['Close'].iloc[-2]
        price_change = ((latest_price - prev_price) / prev_price) * 100

        # Get technical indicators
        rsi_current = 50  # Default
        if technical_indicators and 'RSI' in technical_indicators:
            rsi_current = technical_indicators['RSI'].iloc[-1]

        # Get recommendation
        if recommendation_data:
            recommendation, confidence, reason, expected_change = recommendation_data

            # Enhanced verdict with RSI
            if rsi_current < 30:
                rsi_signal = "📉 Oversold (Good buying opportunity)"
            elif rsi_current > 70:
                rsi_signal = "📈 Overbought (Consider selling)"
            else:
                rsi_signal = f"📊 Neutral RSI ({rsi_current:.1f})"

            verdict = f"🎯 **{recommendation}** ({confidence} confidence)\n\n**RSI Signal:** {rsi_signal}\n\n**Analysis:** {reason}"
        else:
            verdict = "Analysis pending..."

        # Enhanced analysis with technical indicators
        analysis = f"""
✅ **BUILT-IN ANALYSIS ACTIVE**

{verdict}

📊 **TECHNICAL INDICATORS:**
• **RSI:** {rsi_current:.1f} {"(Oversold - Buy Signal)" if rsi_current < 30 else "(Overbought - Sell Signal)" if rsi_current > 70 else "(Neutral)"}
• **Price vs 20-day average:** {'Above' if latest_price > recent_data['Close'].rolling(20).mean().iloc[-1] else 'Below'}
• **Recent momentum:** {'Positive' if price_change > 0 else 'Negative'} ({price_change:.1f}%)

📈 **PRICE PREDICTION:**
"""

        if future_predictions:
            avg_future = np.mean(future_predictions)
            expected_return = ((avg_future - latest_price) / latest_price) * 100
            analysis += f"Expected price in 7 days: ${avg_future:.2f} ({expected_return:+.1f}%)"
        else:
            analysis += "Price prediction unavailable"

        analysis += f"""

⚠️ **RISK ASSESSMENT:**
Market volatility: {recent_data['Close'].pct_change().tail(20).std() * 100:.1f}%

💡 **BOTTOM LINE:**
Based on technical analysis and AI predictions, consider the RSI levels and recent price action before making decisions.

🔧 **Pro Tip:** Install Ollama for more detailed AI insights: `ollama serve` then `ollama pull llama3.2`
"""

        return analysis


class Visualizer:
    """Handles all chart creation and visualization"""

    @staticmethod
    def create_enhanced_chart(data, predictions, y_test_actual, ticker, technical_indicators=None):
        """Create comprehensive chart with technical indicators."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Main price chart with predictions
        recent_dates = data.index[-60:]
        recent_prices = data['Close'].iloc[-60:]
        test_dates = data.index[-len(predictions):]

        ax1.plot(recent_dates, recent_prices, label='📈 Actual Price', color='#2E86AB', linewidth=2.5)
        ax1.plot(test_dates, predictions.flatten(), label='🤖 AI Prediction',
                 color='#F24236', linewidth=2.5, linestyle='--', alpha=0.8)

        # Add moving averages if available
        if technical_indicators and 'SMA_20' in technical_indicators:
            sma_20 = technical_indicators['SMA_20'].iloc[-60:]
            ax1.plot(recent_dates, sma_20, label='20-day MA', color='orange', alpha=0.7)

        ax1.set_title(f'📊 {ticker.upper()} Price & Predictions', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # RSI Chart
        if technical_indicators and 'RSI' in technical_indicators:
            rsi_data = technical_indicators['RSI'].iloc[-60:]
            ax2.plot(recent_dates, rsi_data, color='purple', linewidth=2)
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
            ax2.fill_between(recent_dates, 70, 100, alpha=0.1, color='red')
            ax2.fill_between(recent_dates, 0, 30, alpha=0.1, color='green')
            ax2.set_title(f'📈 RSI Indicator', fontsize=12, fontweight='bold')
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'RSI Data\nNot Available', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=12)
            ax2.set_title('📈 RSI Indicator', fontsize=12, fontweight='bold')

        # Volume Chart
        volume_data = data['Volume'].iloc[-60:]
        ax3.bar(recent_dates, volume_data, alpha=0.7, color='lightblue')
        ax3.set_title(f'📊 Trading Volume', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Volume')
        ax3.grid(True, alpha=0.3)

        # MACD Chart (if available)
        if technical_indicators and 'MACD' in technical_indicators:
            macd_data = technical_indicators['MACD'].iloc[-60:]
            signal_data = technical_indicators['Signal'].iloc[-60:]
            ax4.plot(recent_dates, macd_data, label='MACD', color='blue')
            ax4.plot(recent_dates, signal_data, label='Signal', color='red')
            ax4.bar(recent_dates, technical_indicators['Histogram'].iloc[-60:],
                    alpha=0.3, color='gray', label='Histogram')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_title(f'📈 MACD', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'MACD Data\nNot Available', ha='center', va='center',
                     transform=ax4.transAxes, fontsize=12)
            ax4.set_title('📈 MACD', fontsize=12, fontweight='bold')

        # Format dates for all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        return fig

    @staticmethod
    def create_rsi_gauge(rsi_value):
        """Create an RSI gauge visualization."""
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(projection='polar'))

        # Set up the gauge
        theta = np.linspace(0, np.pi, 100)

        # RSI zones
        oversold_theta = np.linspace(0, np.pi * 0.3, 30)
        neutral_theta = np.linspace(np.pi * 0.3, np.pi * 0.7, 40)
        overbought_theta = np.linspace(np.pi * 0.7, np.pi, 30)

        # Plot zones
        ax.fill_between(oversold_theta, 0, 1, alpha=0.3, color='green', label='Oversold (<30)')
        ax.fill_between(neutral_theta, 0, 1, alpha=0.3, color='yellow', label='Neutral (30-70)')
        ax.fill_between(overbought_theta, 0, 1, alpha=0.3, color='red', label='Overbought (>70)')

        # RSI needle
        rsi_theta = (rsi_value / 100) * np.pi
        ax.plot([rsi_theta, rsi_theta], [0, 0.8], color='black', linewidth=4, label=f'RSI: {rsi_value:.1f}')

        # Formatting
        ax.set_ylim(0, 1)
        ax.set_theta_zero_location('W')
        ax.set_theta_direction(1)
        ax.set_thetagrids([0, 30, 70, 100], ['0', '30', '70', '100'])
        ax.set_title(f'RSI Gauge: {rsi_value:.1f}', pad=20, fontsize=14, fontweight='bold')

        return fig


def main():
    st.title("📈 Smart Stock Predictor")
    st.markdown("### *Advanced AI-powered stock analysis with technical indicators*")

    # Info box for beginners
    with st.expander("🔰 New to stocks? Click here first!"):
        st.markdown("""
        **What is a stock?** A stock represents ownership in a company. When the company does well, your stock value goes up!

        **Technical Indicators Explained:**
        - **RSI (Relative Strength Index):** Shows if a stock is overbought (>70) or oversold (<30)
        - **Moving Averages:** Show the average price over time to identify trends
        - **MACD:** Shows momentum changes in the stock price
        - **Volume:** Shows how many shares are being traded

        **Remember:** This is for educational purposes. Never invest money you can't afford to lose!
        """)

    # Sidebar
    st.sidebar.header("🎯 Stock Analysis Settings")

    # Stock ticker input with popular suggestions
    st.sidebar.markdown("**Choose a stock to analyze:**")
    popular_stocks = {
        "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL",
        "Amazon": "AMZN", "Tesla": "TSLA", "Netflix": "NFLX",
        "Meta (Facebook)": "META", "Disney": "DIS", "NVIDIA": "NVDA"
    }

    stock_choice = st.sidebar.selectbox("Popular stocks:", list(popular_stocks.keys()))
    ticker = st.sidebar.text_input("Or enter any stock symbol:", popular_stocks[stock_choice])

    period = st.sidebar.selectbox("Analysis period:", ["1 year", "2 years", "5 years"], index=1)
    period_map = {"1 year": "1y", "2 years": "2y", "5 years": "5y"}

    # Analysis options
    show_rsi = st.sidebar.checkbox("Show RSI Analysis", value=True)
    show_technical = st.sidebar.checkbox("Show Technical Indicators", value=True)

    if st.sidebar.button("🚀 Analyze This Stock!", type="primary"):
        if not ticker:
            st.error("Please enter a stock symbol first!")
            return

        # Main analysis area
        st.subheader(f"🔍 Analyzing {ticker.upper()}")

        # Progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

        try:
            # Step 1: Load basic data
            status_text.text("📊 Getting stock data...")
            data, company_info = DataHandler.load_stock_data(ticker, period_map[period])

            if data is None:
                st.error("Couldn't find data for this stock. Check the symbol and try again!")
                return

            progress_bar.progress(10)

            # Step 2: Calculate technical indicators
            status_text.text("📈 Calculating technical indicators...")
            tech_indicators = {}

            if show_rsi or show_technical:
                # Calculate RSI
                tech_indicators['RSI'] = TechnicalIndicators.calculate_rsi(data['Close'])

                # Calculate other technical indicators
                moving_averages = TechnicalIndicators.calculate_moving_averages(data['Close'])
                tech_indicators.update(moving_averages)

                # Calculate MACD
                macd_data = TechnicalIndicators.calculate_macd(data['Close'])
                tech_indicators.update(macd_data)

                # Calculate Bollinger Bands
                bollinger_data = TechnicalIndicators.calculate_bollinger_bands(data['Close'])
                tech_indicators.update(bollinger_data)

            progress_bar.progress(25)

            # Step 3: Get news and insider info
            status_text.text("📰 Checking latest news...")
            news = DataHandler.get_company_news(ticker)
            insider_purchases, insider_transactions = DataHandler.get_insider_trading(ticker)
            progress_bar.progress(35)

            # Step 4: Prepare data for AI
            status_text.text("🔧 Preparing data for AI analysis...")
            X_train, y_train, X_test, y_test, scaler = ModelTrainer.preprocess_data(data, 60, 0.2)
            progress_bar.progress(50)

            # Step 5: Train AI model
            status_text.text("🧠 Training AI model...")
            model, history = ModelTrainer.train_lstm_model(X_train, y_train, X_test, y_test)
            progress_bar.progress(70)

            # Step 6: Make predictions
            status_text.text("🔮 Making predictions...")
            predictions_scaled = model.predict(X_test, verbose=0)
            predictions = scaler.inverse_transform(predictions_scaled)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            progress_bar.progress(80)

            # Step 7: Future price predictions
            status_text.text("🔮 Predicting future prices...")
            future_predictions = ModelTrainer.predict_future_prices(model, scaler, data, days_ahead=7)
            progress_bar.progress(90)

            # Step 8: Generate recommendation
            current_price = data['Close'].iloc[-1]
            rsi_current = tech_indicators.get('RSI', pd.Series([50])).iloc[-1]
            recommendation_data = AnalysisEngine.generate_recommendation(
                current_price, future_predictions, data, company_info, rsi_current
            )

            # Step 9: Get comprehensive analysis
            status_text.text("🤖 Getting AI insights...")
            analysis = AnalysisEngine.generate_enhanced_analysis(
                ticker, company_info, data.tail(50), news,
                (insider_purchases, insider_transactions),
                predictions.flatten(), y_test_actual.flatten(),
                future_predictions, recommendation_data, tech_indicators
            )
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")

            # Clear progress indicators
            time.sleep(1)
            progress_container.empty()

            # Display results
            col1, col2 = st.columns([2, 1])

            with col1:
                # Company overview
                st.subheader(f"📋 {company_info.get('longName', ticker)}")

                col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                with col_info1:
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2]
                    price_change = current_price - prev_price
                    price_change_pct = (price_change / prev_price) * 100

                    st.metric(
                        "💰 Current Price",
                        f"${current_price:.2f}",
                        delta=f"{price_change_pct:.1f}%"
                    )

                with col_info2:
                    market_cap = company_info.get('marketCap', 0)
                    st.metric("🏢 Market Cap", f"${market_cap / 1e9:.1f}B")

                with col_info3:
                    sector = company_info.get('sector', 'Unknown')
                    st.metric("🏭 Sector", sector)

                with col_info4:
                    if 'RSI' in tech_indicators:
                        rsi_val = tech_indicators['RSI'].iloc[-1]
                        rsi_status = "🟢 Oversold" if rsi_val < 30 else "🔴 Overbought" if rsi_val > 70 else "🟡 Neutral"
                        st.metric("📊 RSI Status", rsi_status, f"{rsi_val:.1f}")
                    else:
                        st.metric("📊 RSI", "N/A")

                # Enhanced Chart
                st.subheader("📊 Technical Analysis Charts")
                if show_technical:
                    fig = Visualizer.create_enhanced_chart(data, predictions, y_test_actual, ticker, tech_indicators)
                    st.pyplot(fig)
                else:
                    # Simple chart fallback
                    fig, ax = plt.subplots(figsize=(12, 6))
                    recent_dates = data.index[-60:]
                    recent_prices = data['Close'].iloc[-60:]
                    test_dates = data.index[-len(predictions):]

                    ax.plot(recent_dates, recent_prices, label='📈 Actual Price', color='#2E86AB', linewidth=2.5)
                    ax.plot(test_dates, predictions.flatten(), label='🤖 AI Prediction',
                            color='#F24236', linewidth=2.5, linestyle='--', alpha=0.8)

                    ax.set_title(f'📊 {ticker.upper()} Stock Price vs AI Predictions', fontsize=16, fontweight='bold')
                    ax.set_ylabel('Price ($)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                # RSI Gauge (if enabled)
                if show_rsi and 'RSI' in tech_indicators:
                    st.subheader("🎯 RSI Momentum Gauge")
                    col_rsi1, col_rsi2 = st.columns([1, 1])

                    with col_rsi1:
                        rsi_current = tech_indicators['RSI'].iloc[-1]
                        rsi_fig = Visualizer.create_rsi_gauge(rsi_current)
                        st.pyplot(rsi_fig)

                    with col_rsi2:
                        st.markdown("### RSI Interpretation")
                        if rsi_current < 30:
                            st.success("🟢 **OVERSOLD CONDITION**")
                            st.write(
                                "The stock may be undervalued and could bounce back soon. This is often considered a buying opportunity.")
                        elif rsi_current > 70:
                            st.error("🔴 **OVERBOUGHT CONDITION**")
                            st.write(
                                "The stock may be overvalued and could face a pullback. Consider taking profits or waiting.")
                        else:
                            st.info("🟡 **NEUTRAL MOMENTUM**")
                            st.write(
                                "The stock is in a balanced state. Look for other indicators to guide your decision.")

                        # RSI trend analysis
                        rsi_week_ago = tech_indicators['RSI'].iloc[-7] if len(
                            tech_indicators['RSI']) >= 7 else rsi_current
                        rsi_trend = rsi_current - rsi_week_ago

                        if abs(rsi_trend) > 5:
                            trend_direction = "increasing" if rsi_trend > 0 else "decreasing"
                            st.write(
                                f"📈 **RSI Trend:** {trend_direction} by {abs(rsi_trend):.1f} points over the past week")

                # Model Performance
                st.subheader("🎯 AI Model Performance")
                rmse = np.sqrt(mean_squared_error(y_test_actual.flatten(), predictions.flatten()))
                mae = mean_absolute_error(y_test_actual.flatten(), predictions.flatten())
                mape = np.mean(
                    np.abs((y_test_actual.flatten() - predictions.flatten()) / y_test_actual.flatten())) * 100

                col_perf1, col_perf2, col_perf3 = st.columns(3)
                with col_perf1:
                    st.metric("🎯 Accuracy", f"{100 - mape:.1f}%", help="Higher is better")
                with col_perf2:
                    st.metric("📊 Avg Error", f"${mae:.2f}", help="Lower is better")
                with col_perf3:
                    st.metric("📈 RMSE", f"${rmse:.2f}", help="Root Mean Square Error")

                # Recent News
                if news:
                    st.subheader("📰 Recent News & Market Impact")
                    for i, article in enumerate(news[:3]):
                        with st.expander(f"📄 {article.get('title', 'News Article')}"):
                            st.write(
                                f"**Published:** {datetime.fromtimestamp(article.get('providerPublishTime', 0)).strftime('%B %d, %Y at %I:%M %p')}")
                            st.write(article.get('summary', 'No summary available'))
                            if article.get('link'):
                                st.markdown(f"[📖 Read full article]({article['link']})")

            with col2:
                st.subheader("🤖 AI Analysis & Recommendations")

                # Ollama Status
                ollama_running, available_models = AnalysisEngine.check_ollama_status()
                if ollama_running and available_models:
                    st.success(f"🤖 Enhanced AI Ready")
                    st.caption(f"Model: {available_models[0][:20]}...")
                else:
                    st.info("🤖 Built-in Analysis Active")
                    st.caption("Install Ollama for enhanced AI insights")

                # Display comprehensive analysis
                st.markdown(analysis)

                # Trading Recommendation Box
                if recommendation_data:
                    rec, confidence, reason, expected_change = recommendation_data

                    if "STRONG BUY" in rec:
                        st.success(f"🚀 **{rec}**")
                    elif "BUY" in rec:
                        st.success(f"📈 **{rec}**")
                    elif "STRONG SELL" in rec:
                        st.error(f"📉 **{rec}**")
                    elif "SELL" in rec:
                        st.error(f"📉 **{rec}**")
                    else:
                        st.warning(f"⚪ **{rec}**")

                    st.write(f"**Confidence:** {confidence}")
                    st.write(f"**Expected 7-day change:** {expected_change:+.1f}%")
                    st.caption(f"Reason: {reason}")

                # Future Predictions
                if future_predictions:
                    st.subheader("🔮 7-Day Price Forecast")

                    avg_future = np.mean(future_predictions)
                    min_future = min(future_predictions)
                    max_future = max(future_predictions)

                    col_pred1, col_pred2 = st.columns(2)
                    with col_pred1:
                        st.metric("📊 Expected", f"${avg_future:.2f}")
                        st.metric("📉 Low Est.", f"${min_future:.2f}")
                    with col_pred2:
                        potential_gain = ((avg_future - current_price) / current_price) * 100
                        st.metric("📈 High Est.", f"${max_future:.2f}")
                        st.metric("🎯 Potential", f"{potential_gain:+.1f}%")

                    # Daily breakdown
                    with st.expander("📅 Daily Predictions"):
                        for i, price in enumerate(future_predictions, 1):
                            change_from_current = ((price - current_price) / current_price) * 100
                            date_str = (datetime.now() + timedelta(days=i)).strftime('%b %d')
                            st.write(f"**{date_str}:** ${price:.2f} ({change_from_current:+.1f}%)")

                # Technical Indicators Summary
                if show_technical and tech_indicators:
                    st.subheader("📊 Technical Indicators")

                    # Moving Averages
                    if 'SMA_20' in tech_indicators and 'SMA_50' in tech_indicators:
                        sma_20 = tech_indicators['SMA_20'].iloc[-1]
                        sma_50 = tech_indicators['SMA_50'].iloc[-1]

                        st.write("**Moving Averages:**")
                        st.write(f"• 20-day: ${sma_20:.2f}")
                        st.write(f"• 50-day: ${sma_50:.2f}")

                        if current_price > sma_20 > sma_50:
                            st.success("📈 Bullish trend (price above averages)")
                        elif current_price < sma_20 < sma_50:
                            st.error("📉 Bearish trend (price below averages)")
                        else:
                            st.info("📊 Mixed signals")

                    # MACD
                    if 'MACD' in tech_indicators:
                        macd_current = tech_indicators['MACD'].iloc[-1]
                        signal_current = tech_indicators['Signal'].iloc[-1]

                        st.write("**MACD:**")
                        if macd_current > signal_current:
                            st.success("📈 Bullish momentum")
                        else:
                            st.error("📉 Bearish momentum")

                # Risk Assessment
                st.subheader("⚠️ Risk Assessment")

                volatility = data['Close'].pct_change().tail(20).std() * 100
                volume_avg = data['Volume'].tail(20).mean()
                current_volume = data['Volume'].iloc[-1]

                risk_score = 0
                risk_factors = []

                if volatility > 4:
                    risk_score += 1
                    risk_factors.append(f"High volatility ({volatility:.1f}%)")

                if current_volume < volume_avg * 0.5:
                    risk_score += 1
                    risk_factors.append("Low trading volume")

                if market_cap < 1e9:
                    risk_score += 1
                    risk_factors.append("Small company size")

                if risk_score >= 2:
                    st.error("🔴 **HIGH RISK**")
                elif risk_score == 1:
                    st.warning("🟡 **MODERATE RISK**")
                else:
                    st.success("🟢 **LOW RISK**")

                if risk_factors:
                    st.write("**Risk factors:**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")

                # Quick Stats
                st.subheader("📈 Quick Stats")

                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    # Trading activity
                    volume_ratio = current_volume / volume_avg
                    if volume_ratio > 1.5:
                        st.metric("📊 Volume", "🔥 High")
                    elif volume_ratio < 0.5:
                        st.metric("📊 Volume", "😴 Low")
                    else:
                        st.metric("📊 Volume", "📊 Normal")

                    # Recent trend
                    recent_trend = data['Close'].pct_change().tail(5).mean() * 100
                    if recent_trend > 1:
                        st.metric("📈 Trend", "📈 Rising")
                    elif recent_trend < -1:
                        st.metric("📈 Trend", "📉 Falling")
                    else:
                        st.metric("📈 Trend", "➡️ Stable")

                with col_stat2:
                    # Beta (if available)
                    beta = company_info.get('beta', 'N/A')
                    if beta != 'N/A':
                        st.metric("📊 Beta", f"{beta:.2f}")
                    else:
                        st.metric("📊 Beta", "N/A")

                    # P/E Ratio (if available)
                    pe_ratio = company_info.get('forwardPE', company_info.get('trailingPE', 'N/A'))
                    if pe_ratio != 'N/A':
                        st.metric("💰 P/E Ratio", f"{pe_ratio:.1f}")
                    else:
                        st.metric("💰 P/E Ratio", "N/A")

                # Disclaimer
                st.markdown("---")
                st.warning(
                    "⚠️ **Disclaimer:** This analysis is for educational purposes only. Always consult with financial advisors and conduct your own research before making investment decisions. Past performance does not guarantee future results.")

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info(
                "💡 Try refreshing the page or selecting a different stock. If the problem persists, the stock symbol might be invalid or data might be temporarily unavailable.")

            # Show debug info in expander
            with st.expander("🔧 Debug Information"):
                st.code(f"Error details: {str(e)}")
                st.write("**Troubleshooting tips:**")
                st.write("1. Check if the stock symbol is correct")
                st.write("2. Try a different time period")
                st.write("3. Ensure stable internet connection")
                st.write("4. Some stocks may have limited historical data")


if __name__ == "__main__":
    main()
