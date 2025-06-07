
from pydantic import BaseModel, Field

TIME_STEP = 100


class StockPredictionInput(BaseModel):
    past_100_prices: list[float] = Field(
        ...,
        min_items=TIME_STEP,
        max_items=TIME_STEP,
        description=f"""A list of the last {
            TIME_STEP} historical stock prices (float values).""",
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
        le=365, # Limit to reasonable number of forecast days, e.g., 1 year
        description="Number of days to forecast into the future (e.g., 1, 5, 30)."
    )