import rasterio
import numpy as np

# 1. LOAD THE DATA
with rasterio.open("../Data/Mersin_NDVI_2025.tif") as ndvi_src:
    ndvi = ndvi_src.read(1)
    meta = ndvi_src.meta  # We save the metadata (size, location) to use for our new file

with rasterio.open("../Data/Mersin_LST_2025.tif") as lst_src:
    lst = lst_src.read(1)

# 2. PERFORM CLASSIFICATION
# Create an empty grid filled with zeros
hotspots = np.zeros(ndvi.shape, dtype=np.uint8)

# Apply our Scientific Rules
# Rule: If NDVI is very low AND Temperature is very high, mark as 1
hotspot_mask = (ndvi > 0) & (ndvi < 0.2) & (lst > 45)
hotspots[hotspot_mask] = 1

print(f"Hotspots identified: {np.sum(hotspots == 1)} pixels")

# 3. SAVE THE NEW GEOTIFF
# Update metadata to tell QGIS this is an 8-bit integer file (classes 0 and 1)
meta.update(dtype=rasterio.uint8, count=1)

with rasterio.open("../Data/Mersin_Hotspots_2025.tif", 'w', **meta) as dst:
    dst.write(hotspots, 1)

print("Success! 'Mersin_Hotspots_2025.tif' created. Open it in QGIS now!")