import rasterio
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("🚀 Initializing Mersin Machine Learning Model...")

# 1. LOAD DATA
with rasterio.open("../Data/Mersin_NDVI_2025.tif") as src:
    ndvi = src.read(1)
with rasterio.open("../Data/Mersin_LST_2025.tif") as src:
    lst = src.read(1)

# 2. FEATURE ENGINEERING (Distance to Sea)
from scipy.ndimage import distance_transform_edt
land_mask = np.where(ndvi > 0, 1, 0)
dist_km = (distance_transform_edt(land_mask) * 30) / 1000

# 3. PREPARE DATA FOR ML
mask = (ndvi > 0.1) & np.isfinite(lst)
# Features (X): NDVI and Distance
X = np.column_stack((ndvi[mask], dist_km[mask]))
# Target (y): Temperature
y = lst[mask]

# Split data: 80% to train the AI, 20% to test it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN THE RANDOM FOREST
# We use 50 trees (n_estimators) to keep it fast for now
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
print("🧠 Training the model (this may take a minute)...")
model.fit(X_train, y_train)

# 5. EVALUATE
score = model.score(X_test, y_test)
print(f"✅ Model Accuracy (R² Score): {score:.4f}")

# 6. FEATURE IMPORTANCE
importances = model.feature_importances_
print(f"📊 Importance - NDVI: {importances[0]:.2f}, Distance: {importances[1]:.2f}")