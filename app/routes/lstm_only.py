import os
from pathlib import Path
import tensorflow as tf
import joblib
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

from app.models.request_models import StockPredictionInput, MultiStepPredictionInput
from app.services.prediction_service import (
    predict_next_day_price,
    predict_multi_step_prices,
    TIME_STEP,
)

router = APIRouter(prefix="/lstm", tags=["lstm"])

MODEL_DIR = Path("app/models")
SCALER_DIR = Path("app/models")

# --- Global dictionaries to hold loaded models and scalers ---
# These will be populated at application startup
loaded_models: Dict[str, tf.keras.Model] = {}
loaded_scalers: Dict[str, Any] = {}


SUPPORTED_MODEL_SYMBOLS = [
    "ADANIPORTS.NS",
    "BHARTIARTL.NS",
    "RELIANCE.NS", 
]


def preload_all_models():
    print("Preloading LSTM models and scalers for supported stocks...")
    global loaded_models, loaded_scalers
    loaded_models = {}
    loaded_scalers = {}

    # Loading specific models for BHARTIARTL.NS and RELIANCE.NS
    for symbol in ["BHARTIARTL.NS", "RELIANCE.NS", "ADANIPORTS.NS"]:
        try:
            model_path = MODEL_DIR / f"{symbol}_lstm_model.h5"
            scaler_path = SCALER_DIR / f"{symbol}_minmax_scaler.pkl"

            if model_path.exists() and scaler_path.exists():
                print(f"Loading specific model for {symbol}")
                model = tf.keras.models.load_model(model_path)
                scaler = joblib.load(scaler_path)
                loaded_models[symbol] = model
                loaded_scalers[symbol] = scaler
                print(f"✅ Loaded model and scaler for {symbol}.")
            else:
                print(f"❌ Model or scaler for {symbol} not found.")
        except Exception as e:
            print(f"ERROR loading {symbol} model: {e}")

    if not loaded_models:
        print("🚨 No models loaded. All predictions will fail.")


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
    symbol = input_data.symbol.upper()  # Ensure symbol is uppercase for consistency

    # Get the specific model and scaler for this symbol
    if symbol not in loaded_models or symbol not in loaded_scalers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction model not found for stock symbol: {symbol}. Please ensure it's pre-trained and available.",
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
            detail=f"Multi-step prediction failed for {symbol}: {e}",
        )
