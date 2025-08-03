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


@st.cache_data
def load_data(ticker, period="2y"):
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


@st.cache_data
def get_company_news(ticker):
    """Get recent news for the company."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:5]  # Get top 5 news items
        return news
    except:
        return []


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


@st.cache_data
def get_sec_filings(ticker):
    """Get recent SEC filings."""
    try:
        stock = yf.Ticker(ticker)
        # Get quarterly and annual reports
        quarterly = stock.quarterly_financials
        annual = stock.financials
        return quarterly, annual
    except:
        return None, None


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


@st.cache_data
def train_lstm_model(X_train, y_train, X_test, y_test):
    """Train LSTM model with caching."""
    model = build_lstm_model((X_train.shape[1], 1))

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


def check_ollama_status():
    """Check if Ollama is running and get available models."""
    try:
        # Check if Ollama is running
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            return True, model_names
        else:
            return False, []
    except:
        return False, []


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


def generate_recommendation(current_price, predicted_prices, recent_data, company_info):
    """Generate buy/sell/hold recommendation based on predictions and analysis."""

    if not predicted_prices:
        return "HOLD", "Insufficient data for reliable recommendation"

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
    if score >= 2:
        recommendation = "BUY"
        confidence = "Strong"
    elif score >= 1:
        recommendation = "BUY"
        confidence = "Moderate"
    elif score <= -2:
        recommendation = "SELL"
        confidence = "Strong"
    elif score <= -1:
        recommendation = "SELL"
        confidence = "Moderate"
    else:
        recommendation = "HOLD"
        confidence = "Neutral"

    reason_text = "; ".join(reasons[:3])  # Top 3 reasons

    return recommendation, confidence, reason_text, price_change_pct


def get_comprehensive_analysis(ticker, company_info, recent_data, news, insider_data, predictions, actual_prices,
                               future_predictions=None, recommendation_data=None):
    """Get comprehensive analysis using Ollama with all available data."""

    # First check if Ollama is available
    ollama_running, available_models = check_ollama_status()

    if not ollama_running:
        return generate_fallback_analysis(ticker, company_info, recent_data, news, predictions, actual_prices)

    # Choose the best available model
    preferred_models = ['llama3.2', 'llama3.1', 'llama3', 'llama2', 'phi3', 'mistral']
    selected_model = None

    for model in preferred_models:
        if any(model in available_model for available_model in available_models):
            selected_model = next(available_model for available_model in available_models if model in available_model)
            break

    if not selected_model:
        return generate_fallback_analysis(ticker, company_info, recent_data, news, predictions, actual_prices)

    try:
        # Prepare comprehensive context
        latest_price = recent_data['Close'].iloc[-1]
        prev_price = recent_data['Close'].iloc[-2]
        price_change = ((latest_price - prev_price) / prev_price) * 100

        # Calculate technical indicators
        sma_20 = recent_data['Close'].rolling(20).mean().iloc[-1]
        sma_50 = recent_data['Close'].rolling(50).mean().iloc[-1]
        volume_avg = recent_data['Volume'].tail(10).mean() if len(recent_data) >= 10 else recent_data['Volume'].mean()
        current_volume = recent_data['Volume'].iloc[-1]

        # Get prediction accuracy
        if len(predictions) > 0 and len(actual_prices) > 0:
            accuracy = 100 - (np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100)
        else:
            accuracy = 0

        # Prepare news summary
        news_summary = "No recent news available"
        if news:
            news_titles = [item.get('title', '') for item in news[:3]]
            news_summary = " | ".join(news_titles)

        # Company basics
        company_name = company_info.get('longName', ticker)
        sector = company_info.get('sector', 'Unknown')
        market_cap = company_info.get('marketCap', 0)

        # Future price predictions and recommendation
        future_price_text = "No price prediction available"
        recommendation_text = "Analysis pending"

        if future_predictions:
            avg_future_price = np.mean(future_predictions)
            future_price_text = f"Predicted 7-day average: ${avg_future_price:.2f}"

        if recommendation_data:
            rec, confidence, reason, change_pct = recommendation_data
            recommendation_text = f"{rec} ({confidence} confidence) - Expected {change_pct:.1f}% change"

        prompt = f"""
        You are a friendly financial advisor explaining to someone who knows nothing about stocks. 

        Company: {company_name} ({ticker})
        Sector: {sector}
        Current Stock Price: ${latest_price:.2f}
        Yesterday's Change: {price_change:.1f}%
        Market Value: ${market_cap / 1e9:.1f} billion

        Technical Analysis:
        - 20-day average price: ${sma_20:.2f}
        - 50-day average price: ${sma_50:.2f}
        - Trading volume today vs average: {(current_volume / volume_avg) * 100:.0f}%

        AI Prediction Accuracy: {accuracy:.1f}%
        {future_price_text}
        AI Recommendation: {recommendation_text}

        Recent News Headlines: {news_summary}

        Please provide a simple, easy-to-understand analysis in exactly this format:

        🎯 SIMPLE VERDICT: [Clear BUY/SELL/HOLD recommendation with target price if buying]

        📊 WHAT'S HAPPENING: [Explain the recent price movement in simple terms, 2-3 sentences]

        🔮 PRICE PREDICTION: [What price do we expect in the next week? Be specific with numbers]

        📰 NEWS IMPACT: [How might recent news affect the stock? Keep it simple]

        ⚠️ RISKS TO CONSIDER: [What could go wrong? Simple warning in 1-2 sentences]

        💡 BOTTOM LINE: [Final advice with specific action - buy below X price, sell above Y price, or hold]

        Use simple language like you're talking to a friend who's never invested before.
        """

        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': selected_model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'top_p': 0.9
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            analysis_text = result.get('response', '')
            if analysis_text:
                return analysis_text
            else:
                return generate_fallback_analysis(ticker, company_info, recent_data, news, predictions, actual_prices)
        else:
            return generate_fallback_analysis(ticker, company_info, recent_data, news, predictions, actual_prices)

    except requests.exceptions.ConnectionError:
        return generate_fallback_analysis(ticker, company_info, recent_data, news, predictions, actual_prices)
    except requests.exceptions.Timeout:
        return generate_fallback_analysis(ticker, company_info, recent_data, news, predictions, actual_prices)
    except Exception as e:
        return generate_fallback_analysis(ticker, company_info, recent_data, news, predictions, actual_prices)


def generate_fallback_analysis(ticker, company_info, recent_data, news, predictions, actual_prices,
                               future_predictions=None, recommendation_data=None):
    """Generate analysis without Ollama when it's not available."""

    # Calculate basic metrics
    latest_price = recent_data['Close'].iloc[-1]
    prev_price = recent_data['Close'].iloc[-2]
    price_change = ((latest_price - prev_price) / prev_price) * 100

    # Calculate moving averages
    sma_20 = recent_data['Close'].rolling(20).mean().iloc[-1]
    sma_50 = recent_data['Close'].rolling(50).mean().iloc[-1]

    # Volume analysis
    volume_avg = recent_data['Volume'].tail(10).mean() if len(recent_data) >= 10 else recent_data['Volume'].mean()
    current_volume = recent_data['Volume'].iloc[-1]
    volume_ratio = current_volume / volume_avg

    # Prediction accuracy
    if len(predictions) > 0 and len(actual_prices) > 0:
        accuracy = 100 - (np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100)
    else:
        accuracy = 0

    # Generate simple analysis
    company_name = company_info.get('longName', ticker)

    # Get recommendation
    if recommendation_data:
        recommendation, confidence, reason, expected_change = recommendation_data
        verdict = f"📈 **{recommendation}** ({confidence} confidence) - {reason}"
    else:
        # Simple fallback recommendation logic
        if latest_price > sma_20 and price_change > 0:
            recommendation = "BUY"
            verdict = f"📈 **{recommendation}** - {company_name} is showing positive momentum with price above recent averages."
        elif latest_price < sma_20 and price_change < -2:
            recommendation = "SELL"
            verdict = f"📉 **{recommendation}** - {company_name} is experiencing weakness with price below recent averages."
        else:
            recommendation = "HOLD"
            verdict = f"➡️ **{recommendation}** - {company_name} is in a neutral position - wait for clearer signals."

    # What's happening
    if abs(price_change) > 2:
        happening = f"The stock moved {abs(price_change):.1f}% {'up' if price_change > 0 else 'down'} recently, which is a significant move."
    else:
        happening = f"The stock has been relatively stable with only a {abs(price_change):.1f}% change recently."

    # Volume insight
    if volume_ratio > 1.5:
        happening += " Trading activity is higher than usual, suggesting increased investor interest."
    elif volume_ratio < 0.5:
        happening += " Trading activity is lower than usual, suggesting less investor interest."

    # Price prediction
    if future_predictions:
        avg_future = np.mean(future_predictions)
        min_future = min(future_predictions)
        max_future = max(future_predictions)
        prediction_text = f"🔮 **PRICE PREDICTION:** Our AI expects the stock to trade between ${min_future:.2f} - ${max_future:.2f} next week, with an average around ${avg_future:.2f}."

        # Calculate expected return
        expected_return = ((avg_future - latest_price) / latest_price) * 100
        if expected_return > 0:
            prediction_text += f" That's a potential {expected_return:.1f}% gain!"
        else:
            prediction_text += f" That's a potential {abs(expected_return):.1f}% decline."
    else:
        prediction_text = "🔮 **PRICE PREDICTION:** Unable to generate reliable price predictions at this time."

    # News impact
    news_impact = "📰 Recent news may be affecting the stock price - check the news section below for details." if news else "📰 No major recent news found that might significantly impact this stock."

    # Risks
    sector = company_info.get('sector', 'this sector')
    risks = f"⚠️ **RISKS TO CONSIDER:** Like all stocks, {ticker} can be affected by market conditions, company performance, and {sector.lower()} industry changes."

    # Bottom line with specific action
    if recommendation == "BUY":
        target_price = latest_price * 0.95  # Suggest buying 5% below current price
        bottom_line = f"💡 **BOTTOM LINE:** Consider buying {ticker} if it drops to ${target_price:.2f} or below for better value."
    elif recommendation == "SELL":
        target_price = latest_price * 1.05  # Suggest selling 5% above current price
        bottom_line = f"💡 **BOTTOM LINE:** Consider selling {ticker} if it rises to ${target_price:.2f} or above to lock in gains."
    else:
        bottom_line = f"💡 **BOTTOM LINE:** Hold your position in {ticker} and wait for clearer market signals before making moves."

    fallback_analysis = f"""
🤖 **AI Assistant Status:** Currently using built-in analysis (Ollama AI not available)

🎯 **SIMPLE VERDICT:** {verdict}

📊 **WHAT'S HAPPENING:** {happening}

{prediction_text}

{news_impact}

{risks}

{bottom_line}

💡 **Tip:** To get more detailed AI insights, install and run Ollama with: `ollama serve` then `ollama pull llama3.2`
"""

    return fallback_analysis


def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape


def explain_metrics_simply(rmse, mae, mape, avg_price):
    """Explain metrics in simple terms."""
    explanations = []

    # RMSE explanation
    rmse_pct = (rmse / avg_price) * 100
    if rmse_pct < 3:
        explanations.append(
            f"✅ **Very Accurate**: Our predictions are typically within ${rmse:.2f} of the actual price")
    elif rmse_pct < 5:
        explanations.append(f"✅ **Pretty Good**: Our predictions are usually within ${rmse:.2f} of the actual price")
    else:
        explanations.append(f"⚠️ **Moderate Accuracy**: Our predictions can be off by about ${rmse:.2f} on average")

    # MAPE explanation
    if mape < 3:
        explanations.append(f"🎯 **Highly Reliable**: We're accurate about {100 - mape:.1f}% of the time")
    elif mape < 5:
        explanations.append(f"📊 **Good Reliability**: We're accurate about {100 - mape:.1f}% of the time")
    else:
        explanations.append(f"📈 **Fair Reliability**: We're accurate about {100 - mape:.1f}% of the time")

    return explanations


def create_simple_chart(data, predictions, y_test_actual, ticker):
    """Create an easy-to-understand chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot recent actual prices
    recent_dates = data.index[-60:]  # Last 60 days
    recent_prices = data['Close'].iloc[-60:]

    # Plot prediction period
    test_dates = data.index[-len(predictions):]

    # Actual price line
    ax.plot(recent_dates, recent_prices, label='📈 Actual Stock Price',
            color='#2E86AB', linewidth=2.5)

    # Prediction line
    ax.plot(test_dates, predictions.flatten(), label='🤖 AI Prediction',
            color='#F24236', linewidth=2.5, linestyle='--', alpha=0.8)

    # Add shaded area for prediction period
    ax.axvspan(test_dates[0], test_dates[-1], alpha=0.1, color='red',
               label='Prediction Period')

    ax.set_title(f'📊 {ticker.upper()} Stock Price: What Actually Happened vs What AI Predicted',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Stock Price ($)', fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Format dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)

    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def main():
    st.title("📈 Smart Stock Predictor")
    st.markdown("### *Get simple, AI-powered insights about any stock - no financial experience needed!*")

    # Info box for beginners
    with st.expander("🔰 New to stocks? Click here first!"):
        st.markdown("""
        **What is a stock?** A stock represents a tiny piece of ownership in a company. When the company does well, your stock value goes up!

        **What does this app do?** 
        - 🤖 Uses AI to predict where the stock price might go
        - 📰 Checks recent news that might affect the stock
        - 📊 Explains everything in simple terms

        **Remember:** This is for educational purposes. Never invest money you can't afford to lose!
        """)

    # Sidebar
    st.sidebar.header("🎯 Stock Analysis Settings")

    # Stock ticker input with popular suggestions
    st.sidebar.markdown("**Choose a stock to analyze:**")
    popular_stocks = {
        "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL",
        "Amazon": "AMZN", "Tesla": "TSLA", "Netflix": "NFLX",
        "Meta (Facebook)": "META", "Disney": "DIS"
    }

    stock_choice = st.sidebar.selectbox("Popular stocks:", list(popular_stocks.keys()))
    ticker = st.sidebar.text_input("Or enter any stock symbol:", popular_stocks[stock_choice])

    period = st.sidebar.selectbox("How far back to analyze:",
                                  ["1 year", "2 years", "5 years"],
                                  index=1)
    period_map = {"1 year": "1y", "2 years": "2y", "5 years": "5y"}

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
            data, company_info = load_data(ticker, period_map[period])

            if data is None:
                st.error("Couldn't find data for this stock. Check the symbol and try again!")
                return

            progress_bar.progress(15)

            # Step 2: Get news and insider info
            status_text.text("📰 Checking latest news...")
            news = get_company_news(ticker)
            insider_purchases, insider_transactions = get_insider_trading(ticker)
            progress_bar.progress(30)

            # Step 3: Prepare data for AI
            status_text.text("🔧 Preparing data for AI analysis...")
            X_train, y_train, X_test, y_test, scaler = preprocess_data(data, 60, 0.2)
            progress_bar.progress(45)

            # Step 4: Train AI model
            status_text.text("🧠 Training AI model...")
            model, history = train_lstm_model(X_train, y_train, X_test, y_test)
            progress_bar.progress(70)

            # Step 5: Make predictions
            status_text.text("🔮 Making predictions...")
            predictions_scaled = model.predict(X_test, verbose=0)
            predictions = scaler.inverse_transform(predictions_scaled)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            progress_bar.progress(85)

            # Step 6: Future price predictions
            status_text.text("🔮 Predicting future prices...")
            future_predictions = predict_future_prices(model, scaler, data, days_ahead=7)
            progress_bar.progress(90)

            # Step 7: Generate buy/sell/hold recommendation
            current_price = data['Close'].iloc[-1]
            recommendation_data = generate_recommendation(current_price, future_predictions, data, company_info)

            # Step 8: Get comprehensive analysis
            status_text.text("🤖 Getting AI insights...")
            analysis = get_comprehensive_analysis(
                ticker, company_info, data.tail(50), news,
                (insider_purchases, insider_transactions),
                predictions.flatten(), y_test_actual.flatten(),
                future_predictions, recommendation_data
            )
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")

            # Clear progress indicators
            time.sleep(1)
            progress_container.empty()

            # Display results in columns
            col1, col2 = st.columns([2, 1])

            with col1:
                # Company overview
                st.subheader(f"📋 About {company_info.get('longName', ticker)}")

                col_info1, col_info2, col_info3 = st.columns(3)
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
                    st.metric("🏢 Company Size", f"${market_cap / 1e9:.1f}B")

                with col_info3:
                    sector = company_info.get('sector', 'Unknown')
                    st.metric("🏭 Business Type", sector)

                # Chart
                st.subheader("📊 Price Prediction Chart")
                fig = create_simple_chart(data, predictions, y_test_actual, ticker)
                st.pyplot(fig)

                # Model performance in simple terms
                rmse, mae, mape = calculate_metrics(y_test_actual.flatten(), predictions.flatten())
                avg_price = np.mean(y_test_actual.flatten())

                st.subheader("🎯 How Good Is Our Prediction?")
                explanations = explain_metrics_simply(rmse, mae, mape, avg_price)
                for explanation in explanations:
                    st.markdown(explanation)

                # Recent news
                if news:
                    st.subheader("📰 Recent News")
                    for i, article in enumerate(news[:3]):
                        with st.expander(f"📄 {article.get('title', 'News Article')}"):
                            st.write(
                                f"**Published:** {datetime.fromtimestamp(article.get('providerPublishTime', 0)).strftime('%B %d, %Y')}")
                            st.write(article.get('summary', 'No summary available'))
                            if article.get('link'):
                                st.markdown(f"[Read full article]({article['link']})")

            with col2:
                st.subheader("🤖 AI Analysis")

                # Check Ollama status first
                ollama_running, available_models = check_ollama_status()

                if ollama_running and available_models:
                    st.success(f"🤖 AI Ready - Using model: {available_models[0] if available_models else 'Unknown'}")
                else:
                    st.info("🤖 Using built-in analysis (Ollama AI not available)")

                # Display the comprehensive analysis
                if "🎯 SIMPLE VERDICT:" in analysis or "🎯 **SIMPLE VERDICT:**" in analysis:
                    st.markdown(analysis)
                else:
                    st.info(analysis)

                # Quick stats
                st.subheader("📊 Quick Stats")

                # Display recommendation prominently
                if recommendation_data:
                    rec, confidence, reason, expected_change = recommendation_data

                    # Color code the recommendation
                    if rec == "BUY":
                        st.success(f"🚀 **RECOMMENDATION: {rec}**")
                        st.write(f"**Confidence:** {confidence}")
                        st.write(f"**Expected Change:** {expected_change:.1f}%")
                    elif rec == "SELL":
                        st.error(f"📉 **RECOMMENDATION: {rec}**")
                        st.write(f"**Confidence:** {confidence}")
                        st.write(f"**Expected Change:** {expected_change:.1f}%")
                    else:
                        st.warning(f"⚪ **RECOMMENDATION: {rec}**")
                        st.write(f"**Confidence:** {confidence}")
                        st.write(f"**Expected Change:** {expected_change:.1f}%")

                    st.write(f"**Reason:** {reason}")

                    # Future price predictions
                    if future_predictions:
                        st.subheader("🔮 7-Day Price Forecast")
                        avg_future = np.mean(future_predictions)
                        min_future = min(future_predictions)
                        max_future = max(future_predictions)

                        col_pred1, col_pred2, col_pred3 = st.columns(3)
                        with col_pred1:
                            st.metric("Expected Price", f"${avg_future:.2f}")
                        with col_pred2:
                            st.metric("Low Estimate", f"${min_future:.2f}")
                        with col_pred3:
                            st.metric("High Estimate", f"${max_future:.2f}")

                        # Show daily predictions
                        st.write("**Daily Predictions:**")
                        for i, price in enumerate(future_predictions, 1):
                            change_from_current = ((price - current_price) / current_price) * 100
                            st.write(f"Day {i}: ${price:.2f} ({change_from_current:+.1f}%)")

                st.markdown("---")

                # Volume analysis
                volume_avg = data['Volume'].tail(20).mean()
                current_volume = data['Volume'].iloc[-1]
                volume_ratio = current_volume / volume_avg

                if volume_ratio > 1.5:
                    volume_status = "🔥 High activity"
                elif volume_ratio < 0.5:
                    volume_status = "😴 Low activity"
                else:
                    volume_status = "📊 Normal activity"

                st.metric("Trading Activity", volume_status)

                # Price trend
                recent_trend = data['Close'].pct_change().tail(5).mean() * 100
                if recent_trend > 1:
                    trend_status = "📈 Rising"
                elif recent_trend < -1:
                    trend_status = "📉 Falling"
                else:
                    trend_status = "➡️ Stable"

                st.metric("Recent Trend", trend_status)

                # Volatility
                volatility = data['Close'].pct_change().tail(20).std() * 100
                if volatility > 3:
                    vol_status = "🎢 Very volatile"
                elif volatility > 1.5:
                    vol_status = "📊 Moderately volatile"
                else:
                    vol_status = "😌 Stable"

                st.metric("Price Stability", vol_status)

                # Disclaimer
                st.markdown("---")
                st.warning(
                    "⚠️ **Important:** This is educational analysis only. Always do your own research and consult financial advisors before investing!")

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
            st.info("Try refreshing the page or choosing a different stock symbol.")


if __name__ == "__main__":
    main()
