import numpy as np
import tensorflow as tf
import joblib
from typing import Any

TIME_STEP = 100 # Number of past prices to consider for prediction

def predict_next_day_price(
    model_instance: tf.keras.Model,
    scaler_instance: Any,
    past_100_prices: list[float]
) -> float:
    """
    Predicts the next day's stock price using the provided LSTM model and scaler.
    Assumes `past_100_prices` contains exactly TIME_STEP (100) prices.
    """
    input_array = np.array(past_100_prices).reshape(-1, 1)
    scaled_input = scaler_instance.transform(input_array)
    scaled_input_reshaped = scaled_input.reshape(1, TIME_STEP, 1) # Uses TIME_STEP here

    scaled_prediction = model_instance.predict(scaled_input_reshaped, verbose=0)[0][0]
    predicted_price_unscaled = scaler_instance.inverse_transform(
        np.array([[scaled_prediction]])
    )[0][0]
    return float(predicted_price_unscaled)


def predict_multi_step_prices(
    model_instance: tf.keras.Model,
    scaler_instance: Any,
    initial_prices: list[float],
    forecast_days: int
) -> list[float]:
    """
    Performs multi-step forecasting using the provided LSTM model and scaler.
    """
    if len(initial_prices) != TIME_STEP: # Uses TIME_STEP here
        raise ValueError(f"Initial prices must contain exactly {TIME_STEP} entries for multi-step prediction.")

    current_window = list(initial_prices)
    predicted_future_prices = []

    for _ in range(forecast_days):
        next_day_prediction = predict_next_day_price(model_instance, scaler_instance, current_window)
        predicted_future_prices.append(float(next_day_prediction))

        current_window.pop(0)
        current_window.append(next_day_prediction)

    return [float(price) for price in predicted_future_prices]