# Stock Price Analysis with LSTM Model

This project is designed to analyze stock prices using Long Short-Term Memory (LSTM) neural network model. The project utilizes Google Colaboratory for its development environment, leveraging Python and various libraries including numpy, pandas, scikit-learn, keras, and matplotlib.

## Overview

Stock price analysis is crucial for investors and traders to make informed decisions. This project aims to provide a tool for analyzing historical stock prices and predicting future price movements using LSTM, a type of recurrent neural network (RNN) that is well-suited for sequence data.

## Libraries Used

- `math`: Standard library for mathematical operations.
- `numpy`: Fundamental package for scientific computing with Python.
- `pandas`: Data manipulation and analysis library.
- `pandas_datareader`: Enables fetching data from various online sources including Yahoo Finance.
- `MinMaxScaler` from `sklearn.preprocessing`: Used for feature scaling.
- `Sequential`, `Dense`, `LSTM` from `keras.models`: Components for building the LSTM model.
- `matplotlib.pyplot`: Plotting library for data visualization.

## Model Architecture

The LSTM model is constructed using the Keras API, a high-level neural networks API running on top of TensorFlow. Below is a brief overview of the model architecture:

```python
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
```

## Acknowledgements

Google Colaboratory: Free cloud-based Jupyter notebook environment.
Keras Documentation: Official documentation for Keras API.
Yahoo Finance: Source of historical stock price data.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
