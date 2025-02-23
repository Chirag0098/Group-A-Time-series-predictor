import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Streamlit Page Configuration

st.set_page_config(page_title="Time Series Forecasting", layout="wide")
st.title("Time Series Forecasting & Analysis")
st.image(os.path.join(os.getcwd(),"static","time series.png"), width=1000)
st.markdown(
    """
    <style>
    /* Sidebar: Keep Blue Background */
    [data-testid="stSidebar"] {
        background-color: #1E3A8A; /* Dark Blue */
        color: white;
    }

    /* Main Content: Black Background & Light Blue Borders */
    [data-testid="stAppViewContainer"] {
        background-color: black; /* Main Black Background */
        padding: 20px;
    }

    /* Add Light Blue Borders Around Content */
    [data-testid="stAppViewContainer"] > div {
        border: 4px solid #4FC3F7; /* Light Blue Border */
        border-radius: 15px;
        padding: 25px;
        box-shadow: 5px 5px 15px rgba(79, 195, 247, 0.5);
    }

    /* Clock Icon Background */
    body {
        background-image: url('https://img.icons8.com/clouds/512/apple-clock.png');
        background-size: 100px;
        background-repeat: repeat;
        opacity: 0.99;
    }

    /* Adjust Sidebar Text Color */
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Adjust Main Content Text Color */
    [data-testid="stAppViewContainer"] * {
        color: white !important;
    }

    /* Customize Headers */
    h1, h2, h3 {
        color: #4FC3F7 !important; /* Light Blue Headers */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Main Content: Black Background */
    [data-testid="stAppViewContainer"] {
        background-color: black;
        padding: 20px;
    }

    /* Add Blurred Sky Blue Borders */
    [data-testid="stAppViewContainer"] > div {
        border: 4px solid rgba(135, 206, 235, 0.6); /* Sky Blue Border with Transparency */
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0px 0px 20px 10px rgba(135, 206, 235, 0.5); /* Blurred Effect */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Introduction to Time Series Data
st.markdown("This app performs advanced time series forecasting using ARIMA/SARIMA models.")
st.header("📌 Real-Life Applications of Time Series Forecasting")
st.write("""
1. **Stock Market Analysis**: Investors use time series forecasting to predict stock price movements, helping them make informed decisions on buying or selling stocks.
2. **Weather Forecasting**: Meteorologists analyze past temperature, humidity, and pressure trends to make short- and long-term weather predictions.
3. **Sales & Demand Forecasting**: Businesses use historical sales data to forecast demand, helping with inventory management and supply chain optimization.
4. **Healthcare & Disease Prediction**: Hospitals and governments analyze disease trends to predict outbreaks, enabling better resource allocation and policy-making.
""")
# Introduction to Time Series Data
st.write("""
### Understanding Time Series Data
Time series data consists of observations recorded sequentially over time. It is characterized by:
- **Trend**: The long-term movement of the data.
- **Seasonality**: Repeating patterns at fixed intervals.
- **Cyclic Behavior**: Fluctuations that are not of fixed frequency.
- **Irregular Components**: Random noise or anomalies.

A good time series dataset should:
- Have a **datetime index** or a timestamp column.
- Be **regularly spaced** (daily, monthly, yearly, etc.).
- Contain **enough data points** to capture meaningful patterns.
- Be checked for **missing values** and handled appropriately.

Common Applications:
- **Stock Market Analysis**
- **Weather Forecasting**
- **Economic Trends & GDP Analysis**
- **Sales Predictions**
""")
# Function to check if data is a time series
def check_timeseries(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.iloc[:, 0], format='mixed', errors='coerce')
            df.set_index(df.index, inplace=True)
            df = df.iloc[:, 1]
            return df.dropna()
        except Exception as e:
            return str(e)
    return df.dropna()

# Function to test stationarity using ADF and KPSS tests
def test_stationarity(ts):
    adf_result = adfuller(ts.dropna())
    kpss_result = kpss(ts.dropna(), regression='c')
    
    st.write("### Augmented Dickey-Fuller Test (ADF)")
    st.write("*Hypothesis:* H₀: The time series has a unit root (non-stationary), H₁: The time series is stationary.")
    st.write(f"Test Statistic: {adf_result[0]}")
    st.write(f"p-value: {adf_result[1]}")
    st.write(f"Critical Values: {adf_result[4]}")
    if adf_result[1] < 0.05:
        st.write("*Decision:* Reject H₀. The series is stationary.")
    else:
        st.write("*Decision:* Fail to reject H₀. The series is non-stationary.")
    
    st.write("### KPSS Test")
    st.write("*Hypothesis:* H₀: The time series is stationary, H₁: The time series has a unit root (non-stationary).")
    st.write(f"Test Statistic: {kpss_result[0]}")
    st.write(f"p-value: {kpss_result[1]}")
    st.write(f"Critical Values: {kpss_result[3]}")
    if kpss_result[1] < 0.05:
        st.write("*Decision:* Reject H₀. The series is non-stationary.")
    else:
        st.write("*Decision:* Fail to reject H₀. The series is stationary.")
    
    return adf_result[1], kpss_result[1]

# Function to apply Ljung-Box test for residual autocorrelation
def ljung_box_test(residuals, lags=10):
    result = acorr_ljungbox(residuals.dropna(), lags=[lags], return_df=True)
    st.write("### Ljung-Box Test: Checking for Residual Autocorrelation")
    st.write("*Hypothesis:* H₀: Residuals are white noise (no autocorrelation). H₁: Residuals show autocorrelation.")
    st.write(result)
    if result['lb_pvalue'].values[0] < 0.05:
        st.write("*Decision:* Reject H₀. Residuals show autocorrelation, indicating a poor model fit.")
    else:
        st.write("*Decision:* Fail to reject H₀. Residuals are white noise, indicating a good model fit.")

# Upload and analyze data
uploaded_file = st.file_uploader("📂 Upload CSV File (Time Series Data)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = check_timeseries(df)
    if isinstance(df, str):
        st.error("Error in data: " + df)
    else:
        st.subheader("📈 Data Overview & Trends")
        st.line_chart(df)
        
        diff_order = st.slider("Select Number of Differencing", min_value=0, max_value=3, value=1)
        df = df.diff(periods=diff_order).dropna()
        st.subheader("📊 Stationarity Tests")
        test_stationarity(df)
        
        st.subheader("📉 ACF and PACF Plots")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_acf(df, ax=axes[0])
        plot_pacf(df, ax=axes[1])
        axes[0].set_title("Autocorrelation Function (ACF)")
        axes[1].set_title("Partial Autocorrelation Function (PACF)")
        st.pyplot(fig)
        
        seasonal = st.radio("Does your data have seasonality?", [True, False])
        st.subheader("🤖 Best ARIMA/SARIMA Model")
        model = auto_arima(df, seasonal=seasonal, stepwise=True, suppress_warnings=True)
        st.write(model.summary())
        
        forecast_period = st.slider("Select Forecasting Period", min_value=1, max_value=len(df)//2, value=12)
        forecast_values, conf_int = model.predict(n_periods=forecast_period, return_conf_int=True)
        
        st.subheader("📈 Forecasted Values")
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df, label='Actual Data')
        plt.plot(pd.date_range(df.index[-1], periods=forecast_period, freq='D'), forecast_values, label='Forecast', linestyle='dashed')
        plt.legend()
        st.pyplot(plt)
        
        st.subheader("🔍 Ljung-Box Test Results")
        ljung_box_test(model.resid())
        
        st.subheader("📊 Error Metrics")
        mse = mean_squared_error(df[-forecast_period:], forecast_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(df[-forecast_period:], forecast_values)
        
        st.write(f"MSE: {mse} (Lower is better)")
        st.write(f"RMSE: {rmse} (Lower is better)")
        st.write(f"MAE: {mae} (Lower is better)")
