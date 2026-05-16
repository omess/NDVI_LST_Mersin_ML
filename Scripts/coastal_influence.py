import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

print("Calculating Coastal Influence...")

# 1. LOAD NDVI TO FIND WATER
with rasterio.open("../Data/Mersin_NDVI_2025.tif") as src:
    ndvi = src.read(1)
    affine = src.transform # This tells us the size of 1 pixel (usually 30m)

# 2. CREATE A LAND/WATER MASK
# Water usually has NDVI < 0
land_mask = np.where(ndvi > 0, 1, 0) 

# 3. CALCULATE DISTANCE FROM WATER
# This calculates distance in PIXELS
dist_px = distance_transform_edt(land_mask)

# Convert pixel distance to Kilometers (assuming 30m pixels)
dist_km = (dist_px * 30) / 1000

# 4. LOAD LST TO COMPARE
with rasterio.open("../Data/Mersin_LST_2025.tif") as src:
    lst = src.read(1)

# 5. CLEAN DATA (Focus only on Land)
mask = (ndvi > 0.1) & np.isfinite(lst)
dist_clean = dist_km[mask]
lst_clean = lst[mask]

# 6. CALCULATE THE "SEA BREEZE" EFFECT (Regression)
m, b = np.polyfit(dist_clean, lst_clean, 1)

print(f"For every 1km you move inland, temperature changes by {m:.2f}°C")

# 7. PLOT THE GRADIENT
plt.figure(figsize=(10,6))
plt.scatter(dist_clean[::500], lst_clean[::500], alpha=0.1, s=1, color='blue')
plt.plot(dist_clean, m*dist_clean + b, color='red', label=f'Trend: {m:.2f}°C per km')
plt.xlabel("Distance from Sea (km)")
plt.ylabel("LST (°C)")
plt.title("Mersin: How the Sea Cools the Land")
plt.legend()
plt.show()