import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
import warnings

warnings.filterwarnings("ignore")
# Load your data
df = pd.read_csv("2_updated_data(Sheet1).csv")
df['year-month'] = pd.to_datetime(df['year-month'])
df = df.rename(columns={'Annual load cost per month': 'Annual_Load'})
df = df.sort_values(['Employee_ID', 'year-month'])

# Forecast per employee
results = []

for emp_id, group in df.groupby('Employee_ID'):
    ts = group.set_index('year-month').asfreq('MS')['Annual_Load']

    try:
        model = pm.auto_arima(ts, seasonal=True, m=12, suppress_warnings=True, error_action='ignore')
        forecast = model.predict(n_periods=6)
        forecast_dates = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(), periods=6, freq='MS')

        forecast_df = pd.DataFrame({
            'Employee_ID': emp_id,
            'Forecast_Month': forecast_dates,
            'Forecast_Load': forecast
        })
        results.append(forecast_df)
    except Exception as e:
        print(f"Skipping Employee {emp_id} due to error: {e}")

# Combine all forecasts
forecast_df_all = pd.concat(results, ignore_index=True)

# Save or display the result
forecast_df_all.to_csv("forecasted_employee_load.csv", index=False)
print(forecast_df_all.head(12))

# Optional: Plot forecast for one example employee
emp_to_plot = df['Employee_ID'].iloc[0]  # first employee in dataset
actual = df[df['Employee_ID'] == emp_to_plot]
forecast = forecast_df_all[forecast_df_all['Employee_ID'] == emp_to_plot]

plt.figure(figsize=(12, 6))
plt.plot(actual['year-month'], actual['Annual_Load'], label='Actual Load')
plt.plot(forecast['Forecast_Month'], forecast['Forecast_Load'], label='Forecast Load', linestyle='--')
plt.title(f"Forecast for Employee {emp_to_plot}")
plt.xlabel("Date")
plt.ylabel("Annual Load per Month")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
