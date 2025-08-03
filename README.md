# lstm_predict_stock

## Stock Prediction App (LSTM)

## Overview

This project is a stock price prediction app built using **Long Short-Term Memory (LSTM)** networks, powered by **Streamlit**. The app fetches historical stock and ETF data using the **yfinance** library, trains an LSTM model to predict future stock prices, and visualizes the predictions with **matplotlib**.

The goal of this project is to provide a simple, interactive web interface for users to visualize stock predictions and make informed decisions based on historical data trends.

---

## Features

- **Stock/ETF Prediction**: Predict future stock prices using LSTM (a deep learning technique).
- **Interactive Interface**: Built with Streamlit for an easy-to-use and real-time interface.
- **Visualization**: Display stock data, predictions, and evaluation graphs (actual vs. predicted prices).
- **Real-Time Search**: Users can input a stock symbol (ticker) and view predictions for that specific stock.
- **Model Evaluation**: See performance metrics like RMSE (Root Mean Squared Error) for model evaluation.

---

## Requirements

The app requires **Python 3.12** or higher and the following libraries:

- **Streamlit**: For creating the web interface.
- **yfinance**: To download historical stock and ETF data.
- **Keras/TensorFlow**: For building and training the LSTM model.
- **scikit-learn**: For additional machine learning functionalities.
- **matplotlib**: For visualizations.
- **pandas**: For data manipulation.
- **numpy**: For numerical operations.

To install the dependencies, you can use the `requirements.txt` provided.

---

## Installation

### 1. **Clone the Repository**

First, clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/stock-prediction-app.git
cd stock-prediction-app
