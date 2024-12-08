
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tkinter as tk
from tkinter import messagebox

# Function to download stock data
def download_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Function to preprocess data
def preprocess_data(data):
    data_close = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_close)
    return scaled_data, scaler

# Function to create dataset for LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Function to build LSTM model
def build_lstm_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to run the prediction
def predict_stock():
    stock_symbol = stock_entry.get()
    start_date = start_entry.get()
    end_date = end_entry.get()

    try:
        # Download stock data
        data = download_stock_data(stock_symbol, start_date, end_date)

        # Preprocess the data
        scaled_data, scaler = preprocess_data(data)

        # Create dataset
        time_step = 60
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Split the data into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build and train the model
        model = build_lstm_model(X_train)
        model.fit(X_train, y_train, batch_size=64, epochs=5)

        # Predict and plot results
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(actual_prices, color='blue', label='Actual Prices')
        plt.plot(predictions, color='red', label='Predicted Prices')
        plt.title(f'Stock Price Prediction for {stock_symbol}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the GUI application
root = tk.Tk()
root.title("Stock Price Predictor")

# GUI components
tk.Label(root, text="Stock Symbol (e.g., AAPL)").grid(row=0, column=0)
stock_entry = tk.Entry(root)
stock_entry.grid(row=0, column=1)

tk.Label(root, text="Start Date (YYYY-MM-DD)").grid(row=1, column=0)
start_entry = tk.Entry(root)
start_entry.grid(row=1, column=1)

tk.Label(root, text="End Date (YYYY-MM-DD)").grid(row=2, column=0)
end_entry = tk.Entry(root)
end_entry.grid(row=2, column=1)

predict_button = tk.Button(root, text="Predict", command=predict_stock)
predict_button.grid(row=3, column=0, columnspan=2)

# Start the GUI event loop
root.mainloop()
