# Dataset

`Steel_industry_data.csv` is the **Steel Industry Energy Consumption** dataset from DAEWOO Steel Co. (Gwangyang, South Korea):
- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/851
- Kaggle mirror: https://www.kaggle.com/datasets/joebeachcapital/steel-industry-energy-consumption

35,040 rows of 15-minute readings for 2018: energy use (kWh), lagging/leading reactive power,
power factor, CO2, week status, day of week and load type.

Loader notes (`src/data.py`):
- The file starts with a UTF-8 byte-order mark, so it is read with `utf-8-sig`.
- Each day's `00:00` row closes that day (it follows 23:45), so it is moved to the next date,
  giving a continuous 15-minute series from 2018-01-01 00:15 to 2019-01-01 00:00.

If the file is missing, the app generates synthetic data with the same schema so it still runs.
