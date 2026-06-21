# Stock Price Prediction Using Machine Learning and LSTM

## Overview

This project predicts stock prices using both:

1. Linear Regression
2. Long Short-Term Memory (LSTM) Neural Networks

The application analyzes historical stock market data, creates technical indicators, trains machine learning models, and generates future price predictions.

---

## Features

- Historical stock price analysis
- Data preprocessing and cleaning
- Technical indicator generation
  - Moving Average (MA5)
  - Moving Average (MA20)
  - Daily Returns
  - Volume Analysis
- Linear Regression model
- LSTM Deep Learning model
- Future stock price prediction
- Prediction export to CSV

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- TensorFlow
- Keras
- Matplotlib

---

## Project Structure

```
Stock_Prediction/
│
├── stock_code.py
├── price.csv
├── predictions.csv
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Data Preparation

The following features are created from historical stock data:

```python
df["RETURN"] = df["CLOSE"].pct_change()
df["MA5"] = df["CLOSE"].rolling(5).mean()
df["MA20"] = df["CLOSE"].rolling(20).mean()
```

These features help the models identify trends and patterns in stock prices.

---

## Linear Regression Model

The Linear Regression model uses:

- Close Price
- Volume
- MA5
- MA20
- Return
- High Price
- Low Price

to predict future stock prices.

---

## LSTM Model

The LSTM model is trained on sequences of historical stock data.

Features used:

- Close Price
- Volume
- MA5
- MA20
- Return
- High Price
- Low Price

A rolling window of previous days is used to predict the next day's closing price.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sandradileep3009/Stock_Prediction.git
cd Stock_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python stock_code.py
```

---

## Results

The project generates:

- Model predictions
- Performance metrics
- Prediction CSV files

Output file:

```
predictions.csv
```

---

## Future Improvements

- Streamlit Web Application
- Real-time stock data integration
- Multiple stock support
- Hyperparameter tuning
- Advanced forecasting models

---

## Author

Sandra Dileep

AI & Machine Learning Enthusiast
