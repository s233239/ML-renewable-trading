# ML Renewable Trading - Assignment 1

This repository contains the work for Assignment 1 of the **Machine Learning for Energy Systems** course, focusing on **renewable energy trading in day-ahead and balancing electricity markets**. The project combines **machine learning, regression models, and optimization techniques** to forecast wind power production and optimize trading strategies.

---

## 👥 Contributors

This project was developed in collaboration with:  

- [@nic0lew0ng](https://github.com/nic0lew0ng)  
- [@ZilentKnight](https://github.com/ZilentKnight) *(Albert R. H.)*  
- [@s233239](https://github.com/s233239) *(zoewr)*  
- [@MVKA-hub](https://github.com/MVKA-hub)

---

## 📌 Project Overview

The assignment simulates the role of a wind farm owner aiming to trade energy in the electricity market. It processes **historical wind and climate data** from Bornholm, Denmark, along with day-ahead and balancing market prices, to develop predictive models and optimize trading strategies.

Two main modeling approaches were implemented:

1. **Model 1 – Indirect Regression for Trading**
   - Predict wind power production using **linear and non-linear regression** with L1 (Lasso) and L2 (Ridge) regularization.
   - Evaluate models based on **prediction accuracy** and **expected revenue**.
   - Explore **data clustering (k-means)** to improve prediction.

2. **Model 2 – Direct Regression for Trading**
   - Use regression or classification models to directly predict the **optimal offering strategy** for each trading period.
   - Compare performance with Model 1.

---

## 🛠 Skills & Techniques Demonstrated

- Data preprocessing and feature engineering
- Linear and non-linear regression
- Regularization (L1/Lasso and L2/Ridge)
- Optimization for energy trading (linear programming)
- Unsupervised learning (k-means clustering)
- Model evaluation (RMSE, MAE, R², revenue metrics)
- Python programming

---

## 📂 Repository Structure

├── data/ # Raw and processed datasets      
│ ├── features-targets.csv # Model 1 dataset      
│ └── model2_dataset.csv # Model 2 dataset      
├── scripts/ # Python scripts for analysis, models, and optimization      
│ ├── data_collection/ # Functions for dataset generation      
│ │ ├── data_generator.py      
│ │ └── model2datageneration.py      
│ ├── linear_regression/      
│ │ └── linear_regression.py      
│ ├── nonlinear_regression/      
│ │ ├── nonlinear_regression.py      
│ │ ├── nonlinear_regression_metrics.py      
│ │ └── weighted_regression.py      
│ ├── optimization/      
│ │ ├── bid_optimization.py      
│ │ └── revenue_calc.py      
│ └── regularization/      
│ └── regularization.py      
├── main.py # Script to run all relevant programs      
└── report.pdf # Assignment report   

---

## 📈 Key Results

- Non-linear regression with L2 (Ridge) regularization achieved the **highest prediction accuracy** for wind power.
- Model 1 (predictive regression approach) generated the **highest expected revenue** in day-ahead and balancing markets.
- Weighted regression and feature expansion improved accuracy, particularly during periods of high variability.

---

## 📌 Optional Improvements

- Explore **ensemble methods or deep learning models** to improve prediction accuracy.
- Extend analysis to **multiple wind farms** for portfolio-level trading optimization.
- Implement **real-time prediction and trading workflow** for dynamic market conditions.
- Incorporate additional features such as **weather forecasts or intraday market signals**.

---

## 📝 References

- Machine Learning for Energy Systems course materials.
- Historical wind data from Bornholm wind farms (DK2 market area).
- Day-ahead and balancing market data from Danish DK2 market area.

---

