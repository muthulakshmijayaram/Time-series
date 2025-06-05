import pandas as pd

# Step 1: File paths
actual_file = "/content/Emp(in) (2).csv"  # Change this if needed
forecast_file = "/content/forecasted_employee_load_sarimax.csv"  # Change this if needed

# Step 2: Load actual data
actual_df = pd.read_csv(actual_file)
actual_df.columns = actual_df.columns.str.strip()

# Step 3: Prepare actual data
actual_df['year-month'] = pd.to_datetime(actual_df['year-month']).dt.to_period('M').astype(str)
actual_df = actual_df.rename(columns={
    'year-month': 'Month',
    'Monthly FLC': 'Actual_Load'
})

# Step 4: Load forecast data
forecast_df = pd.read_csv(forecast_file)
forecast_df.columns = forecast_df.columns.str.strip()

forecast_df['Forecast_Month'] = pd.to_datetime(forecast_df['Forecast_Month']).dt.to_period('M').astype(str)
forecast_df = forecast_df.rename(columns={
    'Forecast_Month': 'Month',
    'Forecast_Load': 'Predicted_Load'
})

# Step 5: Merge on Employee_ID and Month
merged_df = pd.merge(actual_df, forecast_df, on=['Employee_ID', 'Month'])

# Step 6: Calculate differences
merged_df['Difference'] = merged_df['Actual_Load'] - merged_df['Predicted_Load']
merged_df['%_Error'] = (merged_df['Difference'] / merged_df['Actual_Load']) * 100

# Step 7: Round values
merged_df[['Actual_Load', 'Predicted_Load', 'Difference', '%_Error']] = merged_df[[
    'Actual_Load', 'Predicted_Load', 'Difference', '%_Error'
]].round(2)

# Step 8: Final comparison table
final_df = merged_df[['Employee_ID', 'Month', 'Actual_Load', 'Predicted_Load', 'Difference', '%_Error']]

# Step 9: Export result
final_df.to_csv("/content/final_forecast_comparison.csv", index=False)
print("✅ Comparison file saved as: final_forecast_comparison.csv")
print(final_df.head(10))
