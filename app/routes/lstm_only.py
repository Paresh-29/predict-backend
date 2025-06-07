

from fastapi import APIRouter, HTTPException, status
from app.models.request_models import StockPredictionInput, MultiStepPredictionInput
from app.models.response_models import StockPredictionOutput, MultiStepPredictionOutput
from app.services.prediction_service import predict_next_day_price, predict_multi_step_prices

# APIRouter instance for LSTM-only predictions
router = APIRouter(
    prefix="/lstm", 
    tags=["Direct LSTM Prediction"],  
)


@router.post(
    "/predict",
    response_model=StockPredictionOutput,
    status_code=200, 
    summary="Predict Next Day Price Direct", 
    description="Predicts the next day's stock price given 100 historical prices. This endpoint bypasses the Agentic AI and directly uses the pre-trained LSTM model and MinMaxScaler. It requires a list of exactly 100 historical stock prices as input.",
)
async def predict_next_day_price_direct(input_data: StockPredictionInput):
    """
    **Direct LSTM Prediction:** Predicts the next day's stock price given 100 historical prices.

    This endpoint bypasses the Agentic AI and directly uses the pre-trained LSTM model and MinMaxScaler.
    It requires a list of exactly 100 historical stock prices as input.
    """
    try:
        predicted_price = predict_next_day_price(input_data.past_100_prices)
       
        return StockPredictionOutput(predicted_price=predicted_price, message="Prediction successful") 
    except RuntimeError as e: 
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        )
    except Exception as e:
        print(
            f"Error during direct LSTM prediction: {e}"
        )  
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during LSTM prediction: {e}",
        )


# --- FOR MULTI-STEP FORECASTING ---
@router.post(
    "/multi-predict",
    response_model=MultiStepPredictionOutput,
    summary="Predict Multiple Days Ahead (LSTM)",
    description="Forecasts stock prices for a specified number of future days using a stacked LSTM model. Requires initial 100 historical prices.",
)
async def predict_multi_day_prices(input_data: MultiStepPredictionInput):
    try:
        predicted_prices_list = predict_multi_step_prices(
            input_data.initial_prices, input_data.forecast_days
        )
        return MultiStepPredictionOutput(
            predicted_prices=predicted_prices_list,
            message=f"Successfully predicted {input_data.forecast_days} days.",
        )
    except ValueError as e: 
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except RuntimeError as e: 
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        )
    except Exception as e:
        print(f"Error during multi-step LSTM prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-step prediction failed: {str(e)}",
        )