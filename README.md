#  Bike Sharing Demand Prediction using AutoGluon

##  Overview
This project predicts bike rental demand using the Kaggle Bike Sharing Demand dataset. The goal is to build a regression model that estimates the number of bike rentals based on historical time and weather data.

The solution is built using **AutoGluon TabularPredictor**, demonstrating an end-to-end machine learning pipeline including data preprocessing, feature engineering, model training, and hyperparameter optimization.

---

##  Problem Statement
Bike-sharing systems require accurate demand forecasting to:
- Optimize bike availability
- Improve operational efficiency
- Reduce shortages and over-supply
- Enhance user experience

---

##  Dataset
The dataset includes:
- datetime
- season
- holiday
- workingday
- weather
- temp
- atemp
- humidity
- windspeed
- count (target variable)

---

##  Feature Engineering
New features were extracted from the datetime column:
- hour
- day
- month
- year

These features helped capture time-based demand patterns and improved model performance.

---

##  Model Training
The model was trained using:

- AutoGluon TabularPredictor
- Problem Type: Regression
- Evaluation Metric: RMSE
- Preset: best_quality
- Time Limit: 600–700 seconds

Three modeling stages:
1. Baseline model (raw features)
2. Feature-engineered model
3. Hyperparameter-tuned model

---

##  Hyperparameter Tuning
Model performance was improved using:
- Increased training time
- AutoGluon ensemble models
- Automated hyperparameter search

---

## 📊 Results

| Model Stage | RMSE Score |
|-------------|-----------|
| Baseline Model | 1.79 |
| Feature Engineering | 0.64 |
| Hyperparameter Tuning | 0.46 |

---

## 📈 Visualizations
The project includes:
- Feature distribution histograms
- Model performance comparison plots
- Kaggle score improvement visualization

---

## Key Learnings

- Feature engineering significantly improves model performance
- AutoGluon simplifies model selection and training
- Time-based features are critical for demand forecasting
- Ensemble models outperform single models in tabular datasets

