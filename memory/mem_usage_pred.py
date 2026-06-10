import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

def predict_server_memory(file_path, server_label):
    # Load dataset
    data = pd.read_csv(file_path)
    data.drop(columns=["hostId", "hostName"], inplace=True, errors='ignore')
    
    # Prepare dataframe for Prophet
    df = data[['timestamp', 'memoryUsagePct']].copy()
    df.rename(columns={'timestamp': 'ds', 'memoryUsagePct': 'y'}, inplace=True)
    
    # Convert timestamp to datetime and remove timezone information
    df['ds'] = pd.to_datetime(df['ds'], utc=True).dt.tz_localize(None)
    
    # Sort by date
    df.sort_values(by='ds', inplace=True)
    
    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df)
    
    # Create a dataframe for future predictions (e.g., next 7 days)
    future = model.make_future_dataframe(periods=7, freq='D')
    forecast = model.predict(future)
    
    # Plot the forecast
    fig = model.plot(forecast)
    plt.title(f"Memory Usage Prediction - {server_label}")
    plt.xlabel("Time")
    plt.ylabel("Memory Usage (%)")
    plt.show()

# File paths
file1 = r"C:\server_data\DataCenter Last month data\SERVER WISE DATA\MEMORY DATA\superadmin-memory-custom (1).csv"
file2 = r"C:\server_data\DataCenter Last month data\SERVER WISE DATA\MEMORY DATA\superadmin-memory-custom (2).csv"
file3 = r"C:\server_data\DataCenter Last month data\SERVER WISE DATA\MEMORY DATA\superadmin-memory-custom.csv"

# Predict and plot for each server
predict_server_memory(file1, "Server 1")
predict_server_memory(file2, "Server 2")
predict_server_memory(file3, "Server 3")
