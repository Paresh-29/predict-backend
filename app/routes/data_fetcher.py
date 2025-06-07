
from fastapi import APIRouter, HTTPException, Query
from typing import List
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

router = APIRouter()

DEFAULT_HISTORICAL_LOOKBACK_DAYS = 250

@router.get("/historical_prices", response_model=List[float])
async def get_historical_prices(
    symbol: str = Query(..., description="Stock ticker symbol (e.g., AAPL, GOOGL)"),
    lookback_days: int = Query(DEFAULT_HISTORICAL_LOOKBACK_DAYS, ge=1, description="Number of past days to fetch historical data for. Minimum 1.")
) -> List[float]:
    """
    Fetches historical closing prices for a given stock symbol.

    Args:
        symbol: The stock ticker symbol (e.g., 'AAPL').
        lookback_days: The number of past days to fetch data for.

    Returns:
        A list of historical closing prices in chronological order.
    """
    if not symbol:
        raise HTTPException(status_code=400, detail="Stock symbol cannot be empty.")

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days * 1.5)

        data = yf.download(symbol, start=start_date, end=end_date, interval='1d')

        if data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol} within the specified period. Please check the symbol or date range.")

        # Check if 'Close' column exists in the main DataFrame
        if 'Close' not in data.columns:
            raise HTTPException(status_code=500, detail=f"The 'Close' price column was not found in the historical data for {symbol}.")

        
        # Access the specific stock symbol column from the 'Close' DataFrame
        # This handles the case where data['Close'] is a DataFrame itself
        close_series = data['Close'][symbol] 

        # Ensure the series is numeric and handle NaNs
        close_series = pd.to_numeric(close_series, errors='coerce') # <--- Apply to the series
        close_series.dropna(inplace=True) # <--- Apply to the series

        if close_series.empty: # Check after cleaning
            raise HTTPException(status_code=400, detail=f"No valid numerical historical data found for {symbol} after cleaning. Please check the symbol or date range.")

        historical_prices = [float(price) for price in close_series.values.tolist()] # <--- Use the cleaned series

        if len(historical_prices) < lookback_days:
             raise HTTPException(status_code=400, detail=f"Not enough historical data ({len(historical_prices)} trading days) found for {symbol} for the requested {lookback_days} days after cleaning. Try a smaller lookback period or a different symbol.")
        elif len(historical_prices) > lookback_days:
            historical_prices = historical_prices[-lookback_days:]

        return historical_prices

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data for {symbol}: {str(e)}")