import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)
N_SAMPLES = 8000

# 1. Simulate IMD Monsoon Synoptic Regimes
# Regimes: 0=Break Monsoon (Dry), 1=Normal Monsoon, 2=Active Monsoon (Heavy), 3=Depression / Low Pressure System
regimes = np.random.choice([0, 1, 2, 3], size=N_SAMPLES, p=[0.25, 0.40, 0.25, 0.10])

# Atmospheric variables
temperatures = np.random.uniform(24.0, 36.0, size=N_SAMPLES)
relative_humidity = np.clip(
    np.where(regimes == 0, np.random.normal(55, 10, N_SAMPLES),
    np.where(regimes == 1, np.random.normal(78, 8, N_SAMPLES),
    np.where(regimes == 2, np.random.normal(88, 5, N_SAMPLES),
                           np.random.normal(95, 3, N_SAMPLES)))),
    30, 100
)

wind_speed_knots = np.clip(
    np.where(regimes == 0, np.random.normal(8, 3, N_SAMPLES),
    np.where(regimes == 1, np.random.normal(15, 4, N_SAMPLES),
    np.where(regimes == 2, np.random.normal(25, 6, N_SAMPLES),
                           np.random.normal(38, 8, N_SAMPLES)))),
    2, 60
)

# Raw Numerical Weather Prediction (NWP) model forecast (e.g. GFS / NCMRWF raw output)
# Note: Raw NWP has known systematic bias: underpredicts extreme active/depression rain and overpredicts light rain
true_rainfall = np.clip(
    np.where(regimes == 0, np.random.exponential(1.5, N_SAMPLES),
    np.where(regimes == 1, np.random.gamma(4, 3, N_SAMPLES),
    np.where(regimes == 2, np.random.gamma(8, 6, N_SAMPLES),
                           np.random.gamma(12, 10, N_SAMPLES)))),
    0, 250
)

# Simulate systematic NWP model bias
nwp_forecast = true_rainfall * np.where(regimes == 0, 1.8, # overpredicts break regime
                               np.where(regimes == 1, 1.1,
                               np.where(regimes == 2, 0.65, # underpredicts active regime
                                                      0.45))) + np.random.normal(0, 3, N_SAMPLES) # underpredicts extreme depression
nwp_forecast = np.clip(nwp_forecast, 0, 200)

df = pd.DataFrame({
    'nwp_raw_forecast_mm': nwp_forecast,
    'humidity_pct': relative_humidity,
    'temperature_c': temperatures,
    'wind_speed_knots': wind_speed_knots,
    'synoptic_regime': regimes,
    'ground_truth_rainfall_mm': true_rainfall
})

# Split Train / Test
split = int(0.8 * N_SAMPLES)
train_df, test_df = df.iloc[:split], df.iloc[split:]

X_cols = ['nwp_raw_forecast_mm', 'humidity_pct', 'temperature_c', 'wind_speed_knots', 'synoptic_regime']
X_train, y_train = train_df[X_cols], train_df['ground_truth_rainfall_mm']
X_test, y_test = test_df[X_cols], test_df['ground_truth_rainfall_mm']

print("Training Regime-Aware AI Post-Processing Model (Gradient Boosting)...")
model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
raw_rmse = np.sqrt(mean_squared_error(y_test, test_df['nwp_raw_forecast_mm']))
ai_rmse = np.sqrt(mean_squared_error(y_test, preds))
raw_mae = mean_absolute_error(y_test, test_df['nwp_raw_forecast_mm'])
ai_mae = mean_absolute_error(y_test, preds)
ai_r2 = r2_score(y_test, preds)

print("\n" + "="*50)
print("SIH26080 VALIDATION RESULTS: RAW NWP VS REGIME-AWARE AI")
print("="*50)
print(f"Raw NWP Forecast RMSE:      {raw_rmse:.2f} mm")
print(f"Regime-Aware AI RMSE:       {ai_rmse:.2f} mm  (⬇ {((raw_rmse-ai_rmse)/raw_rmse)*100:.1f}% Error Reduction!)")
print(f"Raw NWP Forecast MAE:       {raw_mae:.2f} mm")
print(f"Regime-Aware AI MAE:        {ai_mae:.2f} mm  (⬇ {((raw_mae-ai_mae)/raw_mae)*100:.1f}% Error Reduction!)")
print(f"AI Post-Processing R² Score: {ai_r2:.3f}")
print("="*50)

out_dir = r"C:\Users\vansh\.gemini\antigravity\scratch\sih-26080-monsoon-ai"
joblib.dump(model, os.path.join(out_dir, "monsoon_postprocessor.pkl"))
test_df.to_csv(os.path.join(out_dir, "test_validation_data.csv"), index=False)
print("Model and validation data saved successfully!")
