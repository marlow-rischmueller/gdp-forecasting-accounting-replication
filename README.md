# Replication Study: Is Accounting Useful for Forecasting GDP Growth?
### A Machine Learning Perspective

> Course Project · Machine Learning in Finance · University of Bremen · 2024  
> Developed as part of a 2-person team · Replication of: Datar, Jain, Wang & Zhang (2020)

Replication and extension of a machine learning approach to GDP growth forecasting using corporate accounting data, professional forecasts, and market variables - evaluated with Elastic Net regression, Random Forest, and a Neural Network.

---

## Motivation

Conventional GDP forecasts (e.g. the Survey of Professional Forecasters) largely ignore firm-level accounting data despite evidence that aggregate earnings and profitability metrics carry predictive signal for macroeconomic growth. Datar et al. (2020) address this gap using Elastic Net regression on a high-dimensional dataset of 341 accounting and market variables.

This project replicates their approach, documents the challenges encountered in reproducing the original results, and extends the analysis with two additional models (Random Forest, Neural Network) and a walk-forward validation strategy that better accounts for the time-series structure of the data.

**Research question:** Does the inclusion of accounting data improve GDP growth forecasts - and if so, at which forecasting horizons?

---

## Methodology

### Data Sources

| Dataset | Source | Description |
|---|---|---|
| Accounting & market data | Compustat (`fundq_NA_202308`) | Quarterly firm-level fundamentals, US companies, 1966–2023 |
| GDP forecasts (SPF) | Federal Reserve Bank of Philadelphia | Mean & median quarterly GDP growth estimates |
| BEA GDP estimates | Federal Reserve Bank of Philadelphia | "Advanced" (preliminary) and "final" GDP growth figures |

> Stock return data (1, 3, 6, 12, 24-month returns) from CRSP were unavailable and are excluded from this replication, resulting in a reduced Feature Set 2 (24 variables instead of 28).

### Feature Sets

| Feature Set              | Variables | Contents |
|--------------------------|---|---|
| FS1 - Survey & Estimates | 21 | SPF forecasts, BEA advanced estimate for prior quarter |
| FS2 - Market             | 24 | Stock price variables from Compustat (CRSP returns excluded) |
| FS3 - Accounting         | 283 | Accounting variables from Compustat (9 excluded due to apparent duplication) |

All models are trained and evaluated on combinations of these feature sets to isolate the contribution of each data type.

### Data Preparation

Following Datar et al. (2020):
1. Filter to US companies; remove duplicates via GV-Key and date
2. Convert year-to-date cumulative accounting variables to quarterly flow variables
3. Derive six growth measures per variable (quarter-over-quarter, year-over-year, scaled by total assets and revenues)
4. Aggregate cross-sectionally per quarter: equal-weighted mean, market-cap-weighted mean, median, standard deviation
5. Remove features with >15% missing values or near-zero variance
6. Winsorise at the 2nd and 98th percentile
7. Remove features with pairwise correlation >90%
8. Standardise to zero mean and unit variance

**Replication challenge:** Datar et al. report 69,993 features after preprocessing; our implementation yields ~26,000. Despite direct correspondence with one of the original authors (Apurv Jain, January 2024), the discrepancy could not be fully resolved. Our analysis therefore proceeds from the feature list provided in the paper's appendix (Datar et al., 2020, p. 34).

### Models

**Elastic Net** (primary model, following Datar et al.):  
Combines L1 (Lasso) and L2 (Ridge) regularisation, enabling variable selection even when the number of features exceeds the number of observations ("short-fat data"). Hyperparameters α ∈ {0, 0.1, ..., 1} and λ ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10} were tuned via walk-forward validation.

**Random Forest** and **Neural Network** (extension):  
Added following Goulet Coulombe et al. (2022), who argue that non-linear models are better suited for long-horizon macroeconomic forecasting.

**Baseline:**  
Exponentially weighted mean of historical GDP growth rates (most recent quarter weighted at 100%, each prior quarter at 90% of the previous weight).

### Validation Strategy

Unlike Datar et al.'s fixed train/validation/test split, we use an **expanding walk-forward validation** over the training set (Q1 1990 – Q3 2014):
- Initial training window: 20 quarters
- Validation window: 20 quarters (fixed size, moving forward)
- The training window expands by one quarter at each step

Final model evaluation uses the holdout test set: Q4 2014 – Q4 2019 (21 quarters).

**Target variable:** BEA "final" GDP growth estimate for the current quarter (Q) and the next four quarters (Q+1 through Q+4).

---

## Results

### Elastic Net (own walk-forward validation)

MSE on holdout test set. Lower is better. Baseline = exponentially weighted historical mean.

| Horizon | FS1 only | FS1 + FS2 | FS1 + FS3 | All Data | Baseline |
|---|---|---|---|---|---|
| Q | **2.079** | 2.186 | 2.269 | 2.215 | 2.502 |
| Q+1 | **2.689** | 2.953 | 2.976 | 3.017 | 2.526 |
| Q+2 | **1.728** | 1.743 | 1.812 | 1.812 | 2.437 |
| Q+3 | 1.796 | 1.796 | **1.730** | **1.730** | 1.829 |
| Q+4 | 1.825 | 1.825 | **1.613** | **1.613** | 1.969 |

### Random Forest

| Horizon | FS1 only | FS1 + FS2 | FS1 + FS3 | All Data | Baseline |
|---|---|---|---|---|---|
| Q | **2.093** | 2.286 | 2.216 | 2.198 | 2.502 |
| Q+1 | 2.536 | 2.476 | 2.352 | **2.203** | 2.526 |
| Q+2 | 2.644 | 1.754 | 1.758 | **1.567** | 2.437 |
| Q+3 | 1.670 | 1.457 | 1.423 | **1.006** | 1.829 |
| Q+4 | 1.871 | 1.529 | 1.263 | **1.187** | 1.969 |

### Neural Network

The neural network underperforms the baseline at all horizons and with all feature set combinations. Results deteriorate substantially when accounting or market data are added. This is likely attributable to the limited architecture (one hidden layer) and the constrained hyperparameter search space inherited from Datar et al.

### Key Findings

**Accounting data adds value for long-horizon forecasts (Q+3, Q+4):**
- Elastic Net with accounting data beats the baseline from Q+3 onwards
- Random Forest with all data beats the baseline from Q+1 onwards and achieves the lowest MSE of any model at Q+3 (1.006) and Q+4 (1.187)
- For short-horizon forecasts (Q, Q+1), professional forecasts alone (FS1) are sufficient - adding accounting or market data increases MSE

**Feature importance (Elastic Net, All Data):**
- At Q: professional forecasts dominate (69% of coefficient weight); Profit category accounts for 17%
- At Q+3 and Q+4: professional forecasts and market data contribute 0%; only accounting categories remain (Investments, Profit, Liabilities, Capital)
- This confirms the original finding that accounting data matters specifically for longer time horizons

**Replication assessment:** Core findings of Datar et al. (2020) are confirmed - accounting data improves long-horizon GDP forecasts. The Random Forest outperforms both the Elastic Net replication and Datar et al.'s original model at Q+3 and Q+4, consistent with Goulet Coulombe et al. (2022).

---

## Repository Structure

```
.
├── project.ipynb               # Data loading & preprocessing (Compustat fundq → Feature Sets 2 & 3)
├── Loading_Data.ipynb          # Feature Set 1 construction (SPF & BEA data)
├── EN_Training.ipynb           # Model training & evaluation (Elastic Net, Random Forest, Neural Net)
├── Zuteilung.xlsx              # Mapping of accounting features to categories (Capital, Investments, etc.)
├── ReadMe.txt                  # Original readme (German)
└── data/
    ├── fundq_NA_202308.txt     # Placeholder - replace with Compustat fundq dataset (see Setup)
    ├── fundq_description.xlsx  # Variable descriptions for the Compustat fundq dataset
    ├── lag_and_target.xlsx     # BEA GDP estimates (advanced & final)
    ├── MeanMedian_NGDP_Growth.xlsx  # SPF nominal GDP growth forecasts
    ├── MeanMedian_RGDP_Growth.xlsx  # SPF real GDP growth forecasts
    ├── target.csv              # Preprocessed target variable
    └── feature_sets/           # Preprocessed feature sets (generated by notebooks)
        ├── feature_set_1_lag.csv
        ├── feature_set_1_without_lag.csv
        ├── feature_set_2_calculated.csv
        └── feature_set_3_calculated.csv
```

### Notebook execution order

1. **`project.ipynb`** - Load and preprocess the Compustat `fundq` dataset; compute Feature Sets 2 and 3; save to `data/feature_sets/`
2. **`Loading_Data.ipynb`** - Load SPF and BEA data; construct Feature Set 1; save to `data/feature_sets/`
3. **`EN_Training.ipynb`** - Load all feature sets; run walk-forward validation; train final models; evaluate on holdout set; compute feature importance

After step 2, the raw Compustat dataset is no longer required - all subsequent work uses the preprocessed CSVs in `data/feature_sets/`.

---

## Technologies

| Category | Libraries |
|---|---|
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn (ElasticNet, RandomForestRegressor, MLPRegressor) |
| Utilities | tqdm, openpyxl (required by pandas for reading `.xlsx` files) |

---

## Setup & Usage

### 1 · Install dependencies

```bash
pip install pandas numpy scikit-learn tqdm openpyxl
```

### 2 · Obtain proprietary datasets

This project requires the **Compustat fundq dataset** (via WRDS - Wharton Research Data Services). Academic access is typically available through university libraries.

**`fundq_NA_202308`** - Quarterly fundamentals:
- Access via WRDS → Compustat → North America → Fundamentals Quarterly
- Filter: all US companies, 1966–2023
- Save as `data/fundq_NA_202308.csv` (replace the placeholder `.txt` file)

**SPF and BEA data** - Available publicly:
- SPF: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters
- BEA: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-for-macroeconomists
- The required files are already included in `data/` (`lag_and_target.xlsx`, `MeanMedian_NGDP_Growth.xlsx`, `MeanMedian_RGDP_Growth.xlsx`)

### 3 · Run notebooks in order

See [Notebook execution order](#notebook-execution-order) above. The preprocessed feature sets in `data/feature_sets/` are already included, so steps 1 and 2 can be skipped if you only want to reproduce the model training and evaluation.

---

## Reference Paper

```
Datar, S., Jain, A., Wang, C. C. Y., & Zhang, S. (2020).
Is Accounting Useful for Forecasting GDP Growth? A Machine Learning Perspective.
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3827510
```

**Further references:**
- Goulet Coulombe, P., Leroux, M., Stevanovic, D., & Surprenant, S. (2022). How is machine learning useful for macroeconomic forecasting?. Journal of Applied Econometrics, 37(5), 920-964.
- Konchitchki, Y., & Patatoukas, P. N. (2014). Accounting earnings and gross domestic product. Journal of Accounting and Economics, 57(1), 76-88.


---

## License

This project is licensed under the [MIT License](LICENSE).  
The Compustat datasets used in this project are proprietary and not included in this repository. Access requires a WRDS subscription.
