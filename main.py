import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet 
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.title('Stock Forecast App')

#defining starting date and ending date of stock tickers
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

#creating drop down list for selected stocks
stocks = ('BTC-USD', 'ETH-USD', 'GOOG', 'AAPL', 'MSFT') 
selected_stock = st.selectbox("Select dataset for prediction", stocks) 

#number of years in the future to be forecasted
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

#downloading ticker data using yfinance
@st.cache 
def load_data(ticker):
    data = yf.download(ticker, START, TODAY) #downloads data from start date to today
    data.reset_index(inplace=True) #put date in the first column
    return data

data_load_state = st.text("Loading Data...")
data = load_data(selected_stock) #load data for selected stock
data_load_state.text("Data Loaded Successfully!")

#tail table of data
st.subheader('Raw data') 
st.write(data.tail())

#plotly graph function
def plot_raw_data(): 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = 'stock_open'))
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = 'stock_close'))
    fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting

#changing column names for fbprophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)

st.subheader('Forecast data') 
st.write(forecast.tail())

st.write('Forecast data')
fig2 = plot_plotly(m, forecast)
st.plotly_chart(fig2) 

st.write('Forecast components')
fig3 = m.plot_components(forecast)
st.write(fig3)
