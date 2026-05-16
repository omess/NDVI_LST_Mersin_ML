import rasterio
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.ndimage import uniform_filter, distance_transform_edt
import matplotlib.pyplot as plt

print("🏗️ Engineering Advanced Spatial Features for Portfolio...")

# 1. LOAD RAW GEOTIFFS
with rasterio.open("../Data/Mersin_NDVI_2025.tif") as src:
    ndvi = src.read(1)
with rasterio.open("../Data/Mersin_LST_2025.tif") as src:
    lst = src.read(1)

# 2. FEATURE ENGINEERING (Advanced GIS Variables)
# Feature 1: Distance to Sea (Our geographic baseline)
land_mask = np.where(ndvi > 0, 1, 0)
dist_km = (distance_transform_edt(land_mask) * 30) / 1000

# Feature 2: Neighborhood Vegetation (5x5 pixel average ~ 150-meter canopy impact)
ndvi_neighborhood = uniform_filter(ndvi, size=5)

# 3. MASK AND CLEAN
mask = (ndvi > 0.1) & np.isfinite(lst)

# Define clean arrays so they can be referenced easily later
ndvi_clean = ndvi[mask]
lst_clean = lst[mask]
dist_clean = dist_km[mask]
neigh_clean = ndvi_neighborhood[mask]

# Build the feature matrix (X) with 3 variables
X = np.column_stack((ndvi_clean, dist_clean, neigh_clean))
y = lst_clean

# Split into Training and Testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN ML MODEL
print("🧠 Training Portfolio-Grade Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. EVALUATE RESULTS
score = model.score(X_test, y_test)
print(f"\n🚀 UPGRADED MODEL RESULTS 🚀")
print(f"Model Accuracy (R² Score): {score:.4f}")

# 6. PRINT FEATURE IMPORTANCES
importances = model.feature_importances_
features = ["Target Pixel NDVI", "Distance to Sea (km)", "Neighborhood NDVI (150m Area)"]

print("\n📊 Driver Dominance Analysis:")
for name, imp in zip(features, importances):
    print(f" - {name}: {imp*100:.1f}%")

# -----------------------------
# 7. GENERATE PORTFOLIO PLOT
# -----------------------------
print("\n📊 Generating final Machine Learning trend plot...")

# Create a clean range of NDVI values for plotting the trend line using defined variables
x_line = np.linspace(ndvi_clean.min(), ndvi_clean.max(), 100)
dist_median = np.median(dist_clean)
neigh_median = np.median(neigh_clean)

# Stack features to match the exact shape the model expects (3 columns)
X_dummy = np.column_stack((x_line, np.full_like(x_line, dist_median), np.full_like(x_line, neigh_median)))
y_line_predict = model.predict(X_dummy)

plt.figure(figsize=(10,7))

# Plot a clean 1% sample of the original data points
plt.scatter(ndvi_clean[::100], lst_clean[::100], s=5, alpha=0.1, color='gray', label='Data Points (1% Sample)')

# Plot the ML prediction trendline
plt.plot(x_line, y_line_predict, color='red', linewidth=3, label='ML Predictive Trend')

plt.xlabel("NDVI (Vegetation Index)")
plt.ylabel("Land Surface Temperature (°C)")
plt.title(f"Mersin: Machine Learning Predictive Trend (R²: {score:.4f})")
plt.legend(loc='upper right')
plt.grid(alpha=0.3)

# Save the missing figure to your maps folder
plt.savefig("../Maps/NDVI_LST_ML_Trend.png", dpi=300)
plt.show()

print("🎯 Success! 'NDVI_LST_ML_Trend.png' has been saved to your Maps folder.")
print("Analysis complete. You are ready to push to GitHub!")