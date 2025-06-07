
from pydantic import BaseModel, Field


class StockResponse(BaseModel):
    content: str


# Model for the direct LSTM prediction (returns a structured number)
class StockPredictionOutput(BaseModel):
    predicted_price: float = Field(
        ..., description="The predicted stock price for the next period."
    )
    message: str = "Prediction successful"

# --- FOR MULTI-STEP PREDICTION RESPONSE ---
class MultiStepPredictionOutput(BaseModel):
    predicted_prices: list[float]
    message: str