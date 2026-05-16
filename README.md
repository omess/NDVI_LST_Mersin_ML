# NDVI and Land Surface Temperature Analysis of Mersin Using NASA Landsat Data

## 🌍 Project Overview
This project presents an end-to-end Geospatial Data Science workflow evaluating the relationship between urban vegetation cover and land surface temperatures (LST) across the coastal region of Mersin, Türkiye. Utilizing satellite remote sensing, automated data pipelines, and Machine Learning regression techniques, this study quantifies the macro-geographical and micro-urban drivers of the local climate to provide actionable insights for urban heat mitigation.

![Mersin NDVI vs LST Spatial Layout](maps/NDVI_vs_LST_Comparison.png)

### 🚀 Key Breakthroughs & Core Skills Demonstrated:
* **Cloud-Scale Remote Sensing:** Ingested and preprocessed NASA/USGS Landsat 8 Collection 2 (Level 2) surface radiance data using Google Earth Engine (GEE).
* **Geospatial Data Engineering (Python):** Developed robust spatial data pipelines using `rasterio` and `numpy` to clean 7.9+ million land pixels, completely neutralizing data contamination from the Mediterranean Sea.
* **Advanced Feature Engineering:** Extracted neighborhood canopy density using focal operations (`scipy.ndimage.uniform_filter`) to calculate localized spatial impacts.
* **Predictive Machine Learning:** Trained a Scikit-Learn Random Forest Regressor to isolate and rank ecological and geographical drivers of urban temperature variations.

---

## 🛠️ Project Structure
```text
project/
│
├── data/              # Raw and processed GeoTIFF rasters (NDVI, LST, Hotspots)
├── maps/              # Exported map layouts, scatter plots, and portfolio graphics
├── scripts/           # Production Python execution scripts (.py)
└── notebooks/         # Exploratory data science notebooks (.ipynb)