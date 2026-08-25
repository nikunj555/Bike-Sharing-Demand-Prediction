# 🚲 BoomBikes — Bike Sharing Demand Prediction

Multiple linear regression model that predicts daily bike rental demand for BoomBikes, a US-based bike-sharing company, to help them plan capacity and marketing strategy as demand recovers post-pandemic.

## Problem Statement

BoomBikes wants to understand **what factors drive daily bike rental demand** and **by how much**, so they can:
- Manipulate business strategy to meet demand levels
- Understand demand dynamics of a new market

This is a regression problem — the target variable `cnt` (total daily rental count) is continuous.

## Dataset

- **730 daily records** spanning 2018–2019
- Features: season, year, month, holiday, weekday, working day, weather situation, temperature, humidity, windspeed
- Target: `cnt` — total daily bike rentals (casual + registered users)

## Approach

| Step | What was done |
|---|---|
| **1. Data Understanding** | Checked shape, dtypes, missing values, statistical summary |
| **2. Data Visualization** | Target distribution, categorical vs. target bar charts, continuous vs. target scatter plots, correlation heatmap, year-over-year monthly trend |
| **3. Data Preparation** | Dropped leaky/irrelevant columns (`casual`, `registered`, `instant`, `dteday`), dropped `atemp` (r ≈ 0.99 with `temp`), one-hot encoded categoricals, 80/20 train-test split, MinMax scaling |
| **4. Model Building** | RFE for initial feature selection → iterative refinement using OLS p-values and VIF (statsmodels) to reach a stable, interpretable feature set |
| **5. Evaluation** | R², RMSE on train/test, residual analysis (distribution + Q-Q plot), actual vs. predicted plot, feature coefficient interpretation |

## Key Results

| Metric | Train | Test |
|---|---|---|
| **R² Score** | 0.837 | **0.855** |
| **RMSE** | 789.7 | 703.8 |

- Train–test R² gap of **1.9%** → model generalizes well, no overfitting.
- Final model retained **15 features**, all statistically significant (p < 0.05).

### Top Demand Drivers (by coefficient magnitude)

| Feature | Effect on Demand |
|---|---|
| `temp` | 📈 Strongest positive driver — warmer days mean significantly more rentals |
| `yr` | 📈 Demand grew substantially from 2018 → 2019 |
| `weathersit` (Light Snow/Rain) | 📉 Sharp drop in rentals during poor weather |
| `hum`, `windspeed` | 📉 High humidity and wind both suppress demand |
| `season` (Spring) | 📉 Lowest-demand season relative to baseline |

## Business Insights

1. **Seasonality matters** — fall and summer months consistently outperform spring; BoomBikes should scale fleet availability and staffing accordingly.
2. **Weather-sensitive demand** — mist and light snow/rain days see a marked drop; dynamic pricing or promotions could help smooth utilization on bad-weather days.
3. **Strong year-over-year growth** — 2019 demand significantly exceeded 2018, suggesting the business should plan for continued expansion.
4. **Working vs. holiday patterns** — holidays show a measurable dip in rentals, useful for operational planning.

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `Statsmodels`

## Repository Structure

```
├── BoomBikes.csv                          # Dataset
├── Bike_sharing_demand.ipynb              # Full analysis notebook
└── README.md
```

## How to Run

```bash
git clone <repo-url>
cd boombikes-demand-prediction
pip install -r requirements.txt
jupyter notebook Bike_sharing_demand.ipynb
```

**Requirements:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`

---

*Part of my machine learning project portfolio — built to demonstrate end-to-end regression modeling: data cleaning, feature selection, multicollinearity diagnostics, and translating model coefficients into business recommendations.*
