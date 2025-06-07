
import numpy as np
import tensorflow as tf
import joblib
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "stock_lstm_model.h5"
)
SCALER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "stock_minmax_scaler.pkl"
)
TIME_STEP = 100 # This constant will be used for the input window size

_model = None
_scaler = None

def load_model_and_scaler():
    """Loads the pre-trained LSTM model and MinMaxScaler into global variables."""
    global _model, _scaler
    if _model is None:
        try:
            print("Loading ML model...")
            _model = tf.keras.models.load_model(MODEL_PATH)
            print("ML model loaded successfully.")
        except Exception as e:
            print(f"Error loading ML model: {e}")
            _model = None 

    if _scaler is None:
        try:
            print("Loading scaler...")
            _scaler = joblib.load(SCALER_PATH)
            print("Scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            _scaler = None 


def get_model():
    """Returns the loaded LSTM model."""
    if _model is None:
       
        load_model_and_scaler()
        if _model is None: 
            raise RuntimeError("ML model not loaded. Cannot perform prediction.")
    return _model


def get_scaler():
    """Returns the loaded MinMaxScaler."""
    if _scaler is None:
        load_model_and_scaler()
        if _scaler is None: 
            raise RuntimeError("Scaler not loaded. Cannot perform prediction.")
    return _scaler


def predict_next_day_price(past_100_prices: list[float]) -> float:
    """
    Predicts the next day's stock price using the loaded LSTM model.
    Assumes `past_100_prices` contains exactly TIME_STEP (100) prices.
    """
    model_instance = get_model()
    scaler_instance = get_scaler()

    input_array = np.array(past_100_prices).reshape(-1, 1)
    scaled_input = scaler_instance.transform(input_array)
    scaled_input_reshaped = scaled_input.reshape(1, TIME_STEP, 1)

    scaled_prediction = model_instance.predict(scaled_input_reshaped, verbose=0)[0][0]
    predicted_price_unscaled = scaler_instance.inverse_transform(
        np.array([[scaled_prediction]])
    )[0][0]
    return predicted_price_unscaled


# --- FUNCTION FOR MULTI-STEP FORECASTING ---
def predict_multi_step_prices(initial_prices: list[float], forecast_days: int) -> list[float]:
    """
    Performs multi-step forecasting using the loaded LSTM model.

    Args:
        initial_prices: A list of the initial `TIME_STEP` (100) historical prices
                        to start the prediction from.
        forecast_days: The number of days to forecast into the future.

    Returns:
        A list of predicted prices for all `forecast_days`.
    """
    if len(initial_prices) != TIME_STEP:
        raise ValueError(f"Initial prices must contain exactly {TIME_STEP} entries for multi-step prediction.")

    # Start with a copy of the initial prices to avoid modifying the original list
    current_window = list(initial_prices)
    predicted_future_prices = []

    for _ in range(forecast_days):
        # Use the current 100-day window to predict the next price
        next_day_prediction = predict_next_day_price(current_window)

        # Add the prediction to our list of predicted prices
        predicted_future_prices.append(next_day_prediction)

       
        current_window.pop(0) 
        current_window.append(next_day_prediction) 

    return predicted_future_prices