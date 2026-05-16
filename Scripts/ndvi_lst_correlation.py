import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

print("Starting Professional NDVI-LST Analysis...")

# 1. LOAD DATA
ndvi_src = rasterio.open("../Data/Mersin_NDVI_2025.tif")
ndvi = ndvi_src.read(1)

lst_src = rasterio.open("../Data/Mersin_LST_2025.tif")
lst = lst_src.read(1)

# 2. CREATE LAND MASK (ndvi > 0.1 excludes water)
mask = (ndvi > 0.1) & (np.isfinite(ndvi)) & (np.isfinite(lst))
ndvi_clean = ndvi[mask]
lst_clean = lst[mask]

# 3. CALCULATE CORRELATION
correlation, p_value = pearsonr(ndvi_clean, lst_clean)
print(f"Correlation: {correlation:.4f}")

from sklearn.linear_model import LinearRegression

# -----------------------------
# 4. MACHINE LEARNING REGRESSION
# -----------------------------
# Reshape data for the ML model
X = ndvi_clean.reshape(-1, 1)
y = lst_clean

# Create and "Train" the model
model = LinearRegression()
model.fit(X, y)

# Get the real slope (m) and intercept (b)
m = model.coef_[0]
b = model.intercept_

# Create the prediction line
x_range = np.array([[ndvi_clean.min()], [ndvi_clean.max()]])
y_predict = model.predict(x_range)

print(f"Corrected Slope (m): {m}")

# -----------------------------
# 5. SCIENTIFIC PLOT (Updated)
# -----------------------------
plt.figure(figsize=(10,7))

# Plot 1% sample to keep the script fast
plt.scatter(ndvi_clean[::100], lst_clean[::100], s=5, alpha=0.2, color='gray', label='Data (1% Sample)')

# Use the prediction points from the Scikit-Learn model
plt.plot(x_range, y_predict, color='red', linewidth=3, label=f'Trend: y={m:.2f}x + {b:.2f}')

plt.xlabel("NDVI (Vegetation Index)")
plt.ylabel("LST (Land Surface Temperature °C)")
plt.title(f"Mersin: The Cooling Effect (Slope: {m:.2f})")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("../Maps/NDVI_LST_ML_Trend.png", dpi=300)
plt.show()

# 6. SAVE AND SHOW
plt.savefig("../Maps/NDVI_LST_Fixed_Trend.png", dpi=300)
plt.show()

print("Analysis complete. Look for 'NDVI_LST_Fixed_Trend.png' in your Maps folder!")