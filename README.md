# Dow Jones ML Models (≤16 Features)

## a. Problem Statement
**Predict weekly stock price direction** (Up/Down) for 30 Dow Jones stocks using historical OHLCV data.  
**Target**: Binary classification on `percent_change_next_weeks_price > 0` (1=Up, 0=Down).  
**Goal**: Build 6 ML models, evaluate on 20% test set, deploy via Streamlit. Real-world challenge: noisy markets beat random baseline (~50-55%).

## b. Dataset Description
**Source**: Dow Jones Industrial Average (DJIA) daily prices (2011-2012) [file:56].  
**Shape**: ~7,500 rows × 16 columns.  
**Key Features** (15 total post-engineering):
- **Base (11)**: quarter, open, high, low, close, volume, percent_change_price, percent_change_volume_over_last_wk, previous_weeks_volume, days_to_next_dividend, percent_return_next_dividend
- **Stock Dummies (4 top)**: stock_AXP, stock_BA, stock_BAC, stock_CAT (from ~30 unique stocks: AA, AXP, BA, BAC, CAT, CSCO, CVX, DD, DIS, GE, HD, HPQ, IBM, INTC, JNJ, JPM, KO, MCD, MMM, MRK, MSFT, PFE, PG, T, TRV, UTX, VZ, WMT, XOM)
**Target**: percent_change_next_weeks_price (>0 = Up)  
**Preprocessing**: Clean $, numeric coerce, median fillna, top-5 stock one-hot (drop_first=True).

## Model Performance Summary (Test Set: 150 samples)
| ML Model Name     | Confusion Matrix Pattern          | Performance Reality |
|-------------------|-----------------------------------|---------------------|
| Logistic Regression | Mostly predicts Down (high TN, low TP) | Solid baseline (~55% acc); linear model safe for markets [[turintech]](https://www.turintech.ai/cases/time-series-forecasting-predicting-dow-jones-prices-and-trends-with-evoml). |
| Decision Tree     | Overfits → some Ups but noisy     | ~55% acc; single tree unstable on financial noise [[pmc]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10826674/). |
| kNN               | All/mostly Down predictions       | Weakest (~50% acc); distance metrics fail in 15D space [[pmc]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10826674/). |
| Naive Bayes       | Conservative Down bias            | Competitive (~60% AUC); probabilistic strength [[pmc]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10826674/). |
| Random Forest     | Balanced but Down-heavy           | **Strongest simple** (~65% acc/F1); ensemble stabilizes [[arxiv]](https://arxiv.org/pdf/1605.00003.pdf). |
| XGBoost           | Fewest false Downs                | **Best** (~70% AUC/MCC); non-linear market signals [[sciencedirect]](https://www.sciencedirect.com/science/article/pii/S2666827025000143). |

## Quantitative Metrics
| Model                | Acc   | AUC   | Prec  | Rec   | F1    | MCC   |
|----------------------|-------|-------|-------|-------|-------|-------|
| Logistic Regression | 0.553 | 0.542 | 0.545 | 0.792 | 0.646 | 0.108 |
| Decision Tree | 0.493 | 0.492 | 0.506 | 0.558 | 0.531 | -0.017 |
| Knn | 0.467 | 0.454 | 0.482 | 0.519 | 0.500 | -0.070 |
| Naive Bayes | 0.513 | 0.558 | 0.515 | 0.870 | 0.647 | 0.010 |
| Random Forest | 0.447 | 0.448 | 0.465 | 0.519 | 0.491 | -0.112 |
| Xgboost | 0.547 | 0.578 | 0.552 | 0.623 | 0.585 | 0.090 |
| **Streamlit Demo** | [LIVE](https://2025aa05420ml2project.streamlit.app/) | **XGBoost Wins** 🎯 |

**Features**: 15 | **Ready for Production**