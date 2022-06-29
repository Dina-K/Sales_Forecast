#import libraries
#import streamlit
import streamlit as st
#import NumPy and Pandas for data manipulation
import pandas as pd
import io
import requests
import numpy as np
#import Prophet
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
#for encoding binary data to printable ASCII characters and decoding it back to binary form
import base64

#define title to be displayed on the top of UI
st.title('Time Series Forecasting')

#instructions to import the data
#st.write('Import data')
#st.write('Import the time series CSV file. It should have two columns labelled as "ds" and "y". The "ds" column should be of DateTime format by Pandas. The "y" column must be numeric representing the measurement to be forecasted.')

#Insert a file uploader widget
df = st.file_uploader('Upload here data here', type='csv')

st.info( f"""
    👆 Upload a .csv file first. Sample to try: [peyton_manning_wiki_ts.csv](https://raw.githubusercontent.com/zachrenwick/streamlit_forecasting_app/master/example_data/example_wp_log_peyton_manning.csv)
    """)

#read the data
if df is not None:
    data = pd.read_csv(df)
    #data=df
    st.write(data)
    data['ds'] = pd.to_datetime(data['ds'], errors='coerce')
    #display the data
    st.write(df)
    #latest date in the data
    """From: """
    min_date= data['ds'].min()
    st.write(min_date)
    """To: """
    max_date = data['ds'].max()
    st.write(max_date)

#Choose the forecast horizon/forecast period
st.write('SELECT FORECAST PERIOD')

st.info('Forecasts become less accurate with larger forecast horizons')

#Insert a numeric input widget
periods_input = st.number_input('How many days would you like to forecast into the future?', min_value=1, max_value=365)

#Fit the time series data for making forecast
if df is not None:
    #instantiate model
    model = Prophet(interval_width=0.98)
    #fit the data
    model.fit(data)

#Visualize the forecasted data
st.info('"yhat" is the predicted value, and the upper and lower limits are (by default) 95% confidence intervals.')

if df is not None:
    future = model.make_future_dataframe(periods=periods_input)

    forecast = model.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered = fcst[fcst['ds'] > max_date]

    st.write(fcst_filtered)

    """
    The next visual shows the actual (black dots) and predicted (blue line) values over time.
    """

    fig1= model.plot(forecast, xlabel='Date')
    st.write(fig1)

    """
    The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.
    """

    fig2=model.plot_components(forecast)
    st.write(fig2)

#Download the Forecast Data
"""
The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)