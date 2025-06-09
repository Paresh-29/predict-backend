import os
from pathlib import Path
import tensorflow as tf
import joblib
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

from app.models.request_models import StockPredictionInput, MultiStepPredictionInput
from app.services.prediction_service import predict_next_day_price, predict_multi_step_prices, TIME_STEP

router = APIRouter(prefix="/lstm", tags=["lstm"])

MODEL_DIR = Path("app/models")
SCALER_DIR = Path("app/models") 

# --- Global dictionaries to hold loaded models and scalers ---
# These will be populated at application startup
loaded_models: Dict[str, tf.keras.Model] = {}
loaded_scalers: Dict[str, Any] = {} 


SUPPORTED_MODEL_SYMBOLS = [
    "AAPL",
    "GOOGL",
    "MSFT",
    "AMZN",
    "TSLA",
    "BHARTIARTL.NS", 
    "RELI"           
]

def preload_all_models():
    print("Preloading LSTM models and scalers for supported stocks...")
    global loaded_models, loaded_scalers
    loaded_models = {}
    loaded_scalers = {}

    # --- Load the generic 'stock' model and scaler once ---
    generic_model_path = MODEL_DIR / "stock_lstm_model.h5"
    generic_scaler_path = SCALER_DIR / "stock_minmax_scaler.pkl"

    generic_model = None
    generic_scaler = None

    try:
        if generic_model_path.exists() and generic_scaler_path.exists():
            print(f"Loading generic model: {generic_model_path} and scaler: {generic_scaler_path}")
            generic_model = tf.keras.models.load_model(generic_model_path)
            generic_scaler = joblib.load(generic_scaler_path)
            print("Successfully loaded generic stock model and scaler.")
        else:
            print(f"WARNING: Generic stock model or scaler not found at {generic_model_path} / {generic_scaler_path}. Other models might not load.")
        

    except Exception as e:
        print(f"ERROR: Could not load generic stock model or scaler: {e}")
        raise RuntimeError(f"Failed to load generic stock model/scaler: {e}")


    # --- Iterate through SUPPORTED_MODEL_SYMBOLS to assign models/scalers ---
    for symbol in SUPPORTED_MODEL_SYMBOLS:
        try:
            # Check for specific models first (like BHARTIARTL.NS, RELI)
            specific_model_path = MODEL_DIR / f"{symbol}_lstm_model.h5"
            specific_scaler_path = SCALER_DIR / f"{symbol}_scaler.pkl"

            if specific_model_path.exists() and specific_scaler_path.exists():
                print(f"Loading specific model for {symbol} from {specific_model_path}")
                loaded_models[symbol] = tf.keras.models.load_model(specific_model_path)
                loaded_scalers[symbol] = joblib.load(specific_scaler_path)
                print(f"Successfully preloaded model and scaler for {symbol}.")
            elif generic_model and generic_scaler: # Assign generic model if specific not found
                print(f"Assigning generic model and scaler to {symbol} (no specific model found).")
                loaded_models[symbol] = generic_model
                loaded_scalers[symbol] = generic_scaler
            else:
                print(f"WARNING: Could not preload model for {symbol}: No specific model found and generic model is not available.")
        except Exception as e:
            print(f"ERROR: Could not preload model for {symbol}: {e}")

    if not loaded_models:
        print("WARNING: No models were loaded. All prediction requests might fail.")



@router.post("/predict")
async def predict_stock_price(input_data: StockPredictionInput):
    # This endpoint is for single-day prediction, ensure your frontend is using multi-predict for forecasting
    symbol = input_data.symbol.upper()

    if symbol not in loaded_models or symbol not in loaded_scalers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction model not found for stock symbol: {symbol}. Please ensure it's pre-trained and available."
        )

    model_instance = loaded_models[symbol]
    scaler_instance = loaded_scalers[symbol]

    try:
        predicted_price = predict_next_day_price(
            model_instance, scaler_instance, input_data.past_100_prices
        )
        return {"symbol": symbol, "predicted_price": predicted_price}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed for {symbol}: {e}"
        )


@router.post("/multi-predict")
async def predict_multi_day_prices(input_data: MultiStepPredictionInput):
    symbol = input_data.symbol.upper() # Ensure symbol is uppercase for consistency

    # Get the specific model and scaler for this symbol
    if symbol not in loaded_models or symbol not in loaded_scalers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction model not found for stock symbol: {symbol}. Please ensure it's pre-trained and available."
        )

    model_instance = loaded_models[symbol]
    scaler_instance = loaded_scalers[symbol]

    try:
        predicted_prices = predict_multi_step_prices(
            model_instance,
            scaler_instance,
            input_data.initial_prices,
            input_data.forecast_days,
        )
        return {"symbol": symbol, "predicted_prices": predicted_prices}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-step prediction failed for {symbol}: {e}"
        )