# Seasonal forecasting of concurrent droughts and heatwaves in India

Code and derived data for the manuscript "Linking Climate Drivers to Seasonal Forecasts of Concurrent Droughts and Heatwaves in India" (submitted to the Bulletin of the American Meteorological Society).

## Overview
The framework estimates the probability of a concurrent drought and heatwave (CDHW) for each 1° grid cell and month over India at lead times of 0–3 months.

- CDHW: a month with SPI-3 < −1 (drought) and monthly SHI > 1 (heatwave) in the same grid cell.
- Forecasts: CFSv2 reforecasts (1982–2011) and real-time forecasts (2012–2025), ensemble mean of 24 members, regridded to 1°, bias-corrected with Quantile Delta Mapping (QDM).
- Model: XGBoost with probability calibration, evaluated with blocked five-fold cross-validation over 1982–2025.

## Repository structure
- `code/`: [list the scripts in order, e.g. 01_compute_spi_shi.py, 02_qdm_bias_correction.py, 03_train_xgboost.py, 04_plots.py]
- `data/`: derived data produced in this study [list files]
- `figures/`: scripts or outputs for the manuscript figures

## Input data (not included, open access)
- Rainfall (daily, 0.25°, 1951–2025): India Meteorological Department (IMD), Pai et al. (2014), https://doi.org/10.54302/mausam.v65i1.851
- Maximum temperature (daily, 1°, 1951–2025): IMD, Srivastava et al. (2009), https://doi.org/10.1002/asl.232
- Climate indices (NAO, PDO, ONI, DMI): NOAA National Centers for Environmental Information, https://www.ncei.noaa.gov/access/monitoring/products/
- Precipitation and maximum temperature forecasts (CFSv2 reforecast 1982–2011 and real-time 2012–2025, ensemble mean of 24 members): NCEP CFSv2, Saha et al. (2014), https://doi.org/10.1175/JCLI-D-12-00823.1, accessed through the IRI Data Library [add the IRI dataset link you used]
- Köppen–Geiger climate classes (1°): Beck et al. (2023), https://doi.org/10.1038/s41597-023-02549-6
- Population (2011 Census, sub-district level): National Water Informatics Centre, India-WRIS, https://indiawris.gov.in/wris/#/geoSpatialData 

## Contact
Vemuri Harini, Indian Institute of Science, Bangalore, [vemuriharini@iisc.ac.in]
