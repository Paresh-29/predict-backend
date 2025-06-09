from pydantic import BaseModel, Field
from typing import List

TIME_STEP = 100

class StockPredictionInput(BaseModel):
    past_100_prices: list[float] = Field(
        ...,
        min_items=TIME_STEP,
        max_items=TIME_STEP,
        description=f"""A list of the last {
            TIME_STEP} historical stock prices (float values).""",
    )
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20, # max_length as appropriate for ticker symbols
        description="Stock ticker symbol (e.g., AAPL, BRTI, RELI).",
    )


class StockRequest(BaseModel):
    stock_name: str

# --- MULTI-STEP PREDICTION REQUEST ---
class MultiStepPredictionInput(BaseModel):
    initial_prices: list[float] = Field(
        ...,
        min_items=TIME_STEP,
        max_items=TIME_STEP,
        description=f"""A list of the initial {TIME_STEP} historical stock prices to start the forecast from."""
    )
    forecast_days: int = Field(
        ...,
        gt=0,
        le=365, # Limit to number of forecast days  -> 1 year
        description="Number of days to forecast into the future (e.g., 1, 5, 30)."
    )
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20, # max_length as appropriate for ticker symbols
        description="Stock ticker symbol (e.g., AAPL, BRTI, RELI).",
    )