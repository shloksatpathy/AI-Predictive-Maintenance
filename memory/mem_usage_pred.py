import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

def predict_server_memory(file_path, server_label):
    # Load dataset
    data = pd.read_csv(file_path)
    data.drop(columns=["host_id", "cpu_usage", "status", "power_kw"], inplace=True, errors='ignore')
    
    # Prepare dataframe for Prophet
    df = data[['ts', 'memory_usage_pct']].copy()
    df.rename(columns={'ts': 'ds', 'memory_usage_pct': 'y'}, inplace=True)
    
    # Convert timestamp to datetime and remove timezone information
    df['ds'] = pd.to_datetime(df['ds'], format='mixed', utc=True).dt.tz_localize(None)
    
    # Sort by date
    df.sort_values(by='ds', inplace=True)
    
    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df)
    
    # Create a dataframe for future predictions (e.g., next 7 days)
    future = model.make_future_dataframe(periods=7, freq='D')
    forecast = model.predict(future)
    print(forecast)
    # Plot the forecast
    fig = model.plot(forecast)
    plt.title(f"Memory Usage Prediction - {server_label}")
    plt.xlabel("Time")
    plt.ylabel("Memory Usage (%)")
    plt.show()

# File paths
file1 = r"C:\server_data\Datacenter Datas\SERVER WISE DATA\host_metrics_host1.csv"
file2 = r"C:\server_data\Datacenter Datas\SERVER WISE DATA\host_metrics_host2.csv"
file3 = r"C:\server_data\Datacenter Datas\SERVER WISE DATA\host_metrics_host3.csv"

 #Predict and plot for each server
predict_server_memory(file1, "Server 1")
predict_server_memory(file2, "Server 2")
predict_server_memory(file3, "Server 3")
# data = pd.read_csv(r"C:\Users\Shlok\Downloads\host_metrics 1.csv")
# data_filt = data[~data['host_id'].isin([2, 3])].reset_index(drop=True)
# print(data.columns)
# data_filt.drop(columns=["id", "host_id", "cpu_usage_pct", "power_kw", "temperature_c", "status"], inplace=True)
# df = data_filt[['ts', 'memory_usage_pct']].copy()
# df.rename(columns={'ts':'ds', 'memory_usage_pct':'y'}, inplace=True)
# df['ds'] = pd.to_datetime(df['ds'], format='mixed', utc=True).dt.tz_localize(None)
# df.sort_values(by='ds', inplace=True)

# model=Prophet()
# model.fit(df)

# future = model.make_future_dataframe(periods=30, freq='D')
# forecast = model.predict(future)

# fig = model.plot(forecast)
# plt.title("Memory Usage Prediction - Host Metrics")
# plt.xlabel("Time")
# plt.ylabel("Memory Usage (%)")
# plt.show()