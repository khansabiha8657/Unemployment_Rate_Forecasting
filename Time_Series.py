import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit

def read_unemp(file='C:/Users/Mohd Sajid Khan/Documents/Sabiha_915/Time series/output.csv',index=range(1, 885548)):
    # Read the CSV file into a DataFrame
    unemp_data = pd.read_csv(file)
    
    # Remove rows with missing values (NaN)
    unemp_data = unemp_data.dropna()
    
    return unemp_data

# Call the function to read the data
unemp_data = read_unemp()
unemp_data.index = pd.RangeIndex(start=1, stop=len(unemp_data) + 1)
print(unemp_data)

# Create a dictionary of DataFrames for each year
yearly_data = {}
for year in range(1990, 2017):
    yearly_data[year] = unemp_data[unemp_data['Year'] == year]

# Create an empty dictionary to store the mean monthly rates for each year
mean_monthly_rates = {}

# Define the sequence of months
months_in_sequence = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Loop through each year from 1990 to 2016
for year in range(1990, 2017):
    # Filter data for the current year
    year_data = unemp_data[unemp_data['Year'] == year]
    
    # Calculate the mean monthly rates for the current year and arrange them in the specified order
    mean_rates = [year_data[year_data['Month'] == month]['Rate'].mean() for month in months_in_sequence]
    
    # Create a dictionary where keys are months and values are mean rates, then store it
    mean_monthly_rates[year] = dict(zip(months_in_sequence, mean_rates))

# Create a list of years from 1990 to 2016, repeated for 12 months each
years = [year for year in range(1990, 2017) for _ in range(12)]

# Create a list of month names repeated for each year
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] * 27

# Flatten the mean_monthly_rates dictionary into a list
mean_rates = [rate for year in range(1990, 2017) for rate in mean_monthly_rates[year].values()]

# Create a DataFrame from the lists
mean_monthly_rate_df = pd.DataFrame({'year': years, 'month': months, 'rate': mean_rates})
mean_monthly_rate_df
# Now, mean_monthly_rate_df is a DataFrame with columns 'year', 'month', and 'rate', containing the mean monthly rates by year.

# Create a new column 'Date' by combining 'year' and 'month' columns
mean_monthly_rate_df['Date'] = pd.to_datetime(mean_monthly_rate_df['year'].astype(str) + '-' + mean_monthly_rate_df['month'].astype(str) + '-01', format='%Y-%b-%d')

# Select the columns you want to keep (excluding 'year' and 'month')
df = mean_monthly_rate_df.drop(columns=['year', 'month'])
df

df["Date"]= pd.to_datetime(df["Date"])

df.set_index("Date", inplace=True)
df
df.isna().sum()
df_copy = df.copy()

plt.boxplot(df)
# Add labels and a title
plt.xlabel('Data')
plt.ylabel('Values')
plt.title('Box Plot Example')

# Display the plot
plt.show()

def replace_outliers_with_mean(data, threshold=1.5):
    data = np.array(data)  # Convert input to a NumPy array for easier manipulation
    quartiles = np.percentile(data, [25, 75])  # Calculate 1st and 3rd quartiles
    iqr = quartiles[1] - quartiles[0]  # Calculate interquartile range (IQR)
    
    lower_bound = quartiles[0] - threshold * iqr  # Calculate lower bound for outliers
    upper_bound = quartiles[1] + threshold * iqr  # Calculate upper bound for outliers
    
    # Replace outliers with the mean of non-outlier data points
    data[(data < lower_bound) | (data > upper_bound)] = np.mean(data[(data >= lower_bound) & (data <= upper_bound)])
    
    return data

data = replace_outliers_with_mean(df)
data

df3 = pd.DataFrame(data, columns=["Rate"], index = df.index)
df3.info()

# Min-Max scaling
def normalize_data(data):
    min_vals = data.min()
    max_vals = data.max()
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data
df3.iloc[:,0] = normalize_data(df3.iloc[:,0])
df3

df_copy.plot( figsize = (15,7), ylabel = "Unemployment Rate", title=  "Time Series Plot of US Unemployment Rate")
df_2.plot( figsize = (15,7), xlabel = 'year',ylabel = "AverageTemperature", title=  "Temperature change over the years")

# Perform seasonal decomposition
decomposition = seasonal_decompose(df3["Rate"].dropna(), model='additive', period = 4) 

# Plot the decomposed components
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(decomposition.observed)
plt.title('Observed Data')
plt.subplot(4, 1, 2)
plt.plot(decomposition.trend)
plt.title('Trend')
plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal)
plt.title('Seasonality')
plt.subplot(4, 1, 4)
plt.plot(decomposition.resid)
plt.title('Residuals')
plt.tight_layout()
plt.show()

# ADF test for stationarity
result = adfuller(df3["Rate"])
print("ADF Test Results:")
print(f"ADF Statistic: {result[0]}")
print(f"P-value: {result[1]}")
print("Critical Values:")

for key, value in result[4].items():
    print(f"{key}: {value}")

# first difference
df3["differenced_rate"] = df3["Rate"].diff(1)
df3["differenced_rate"].dropna().plot(figsize = (14,6))

# ADF test for stationarity
result = adfuller(df3["differenced_rate"].dropna())
print("ADF Test Results:")
print(f"ADF Statistic: {result[0]}")
print(f"P-value: {result[1]}")
print("Critical Values:")

for key, value in result[4].items():
    print(f"{key}: {value}")

# for seasonal 
df3["seasonal_rate"] = df3["Rate"].diff(12)
df3

result = adfuller(df3["seasonal_rate"].dropna())
print("ADF Test Results:")
print(f"ADF Statistic: {result[0]}")
print(f"P-value: {result[1]}")
print("Critical Values:")
for key, value in result[4].items():
    print(f"{key}: {value}")

df3["seasonal_rate_differenced"] = df3["seasonal_rate"].diff()

result = adfuller(df3["seasonal_rate_differenced"].dropna())
print("ADF Test Results:")
print(f"ADF Statistic: {result[0]}")
print(f"P-value: {result[1]}")
print("Critical Values:")
for key, value in result[4].items():
    print(f"{key}: {value}")

# Plotting PACF and PACF function of first difference for getting the non seasonal parameters i.e. (p,d,q)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(df3["differenced_rate"].dropna(), lags=40, ax=ax1)
plot_pacf(df3["differenced_rate"].dropna(), lags=40, ax=ax2)
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(df3["seasonal_rate_differenced"].dropna(), lags=60, ax=ax1)
plot_pacf(df3["seasonal_rate_differenced"].dropna(), lags=60, ax=ax2)
plt.show()
df3.shape

# training data 
train_data1 = df3.iloc[:300,:]
train_data1

model = pm.auto_arima(train_data1["Rate"], m = 12, seasonal =True, start_p =0, d=1, start_q =0,max_p=3
                      ,max_q =9,max_order = None, test = "adf", stepwise=True, trace = True,D = 0,
                      max_P =4, max_Q =1)
model.summary()


## Forecast future values using the fitted model
forecast_steps = 24

# Extract the test data for comparison
test_data1 = df3["Rate"][-forecast_steps:]
test_data1

forecast1 = model.predict(n_periods=24)
forecast1

# Calculate evaluation metrics
mae = mean_absolute_error(test_data1, forecast1)
mse = mean_squared_error(test_data1, forecast1)
rmse = np.sqrt(mse)
msle = mean_squared_log_error(test_data1, forecast1)
mape = np.mean(np.abs((test_data1 - forecast1) / test_data1)) * 100

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Squared Log Error (MSLE): {msle}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")

# plot the second model's forecast 
plt.figure(figsize=(15, 7))
plt.plot(forecast1.index, forecast1, label='SARIMAX Forecast', color='blue')
plt.plot(df3.index, df3["Rate"], label='Actual Data', color='green')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.title('SARIMAX Forecast vs Actual Data')
plt.show()

# Fit an SARIMAX model to the time series data
model1 = SARIMAX(train_data1["Rate"], order=(0,1,0),seasonal_order=(1,1,1,12))
fitted_model = model1.fit()
print(fitted_model.summary())

diag1 = model.plot_diagnostics(figsize=(15, 7))  # this only show the plot of ACF
plt.show()

# PACF 
res = model.resid()
fig, ax1  = plt.subplots(1, 1, figsize=(10, 4))
plot_pacf(res, lags=10, ax=ax1)
plt.show()

# PACF 
res = model.resid()
fig, ax1  = plt.subplots(1, 1, figsize=(10, 4))
plot_acf(res, lags=10, ax=ax1)
plt.show()

# Fit an SARIMAX model to the time series data
model1 = SARIMAX(train_data1["Rate"], order=(1,1,1),seasonal_order=(1,0,1,12))
fitted_model = model1.fit()
print(fitted_model.summary())

## Forecast future values using the fitted model
forecast_steps = 24

# Extract the test data for comparison
test_data1 = df3["Rate"][-forecast_steps:]
test_data1

forecast_results = fitted_model.get_forecast(steps=forecast_steps)
forecast = forecast_results.predicted_mean
forecast

# Calculate evaluation metrics
mae = mean_absolute_error(test_data1, forecast)
mse = mean_squared_error(test_data1, forecast)
rmse = np.sqrt(mse)
msle = mean_squared_log_error(test_data1, forecast)
mape = np.mean(np.abs((test_data1 - forecast) / test_data1)) * 100

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Squared Log Error (MSLE): {msle}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")

# plot the second model's forecast 
plt.figure(figsize=(15, 7))
plt.plot(forecast.index, forecast, label='SARIMAX Forecast', color='blue')
plt.plot(df3.index, df3["Rate"], label='Actual Data', color='green')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.title('SARIMAX Forecast vs Actual Data')
plt.show()

# Fit an SARIMAX model to the time series data
model2 = SARIMAX(df3["Rate"], order=(1,1,1),seasonal_order=(1,0,1,12))
fitted_model2 = model2.fit()

print(fitted_model2.summary())

# Get the forecast and confidence intervals
forecast_results2 = fitted_model2.get_forecast(steps=24)
forecast2 = forecast_results2.predicted_mean
conf_int = forecast_results2.conf_int(alpha = 0.1)
forecast2

# plot the forecasted values and 90% confidence interval
plt.figure(figsize=(15, 7))
plt.plot(forecast2.index, forecast2, label='SARIMAX Forecast', color='blue')
plt.plot(df3.index, df3["Rate"], label='Actual Data', color='green')
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='gray', alpha=0.1, label='Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.title('SARIMAX Forecast vs Actual Data with Confidence Intervals')
plt.show()
