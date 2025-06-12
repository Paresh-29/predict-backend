from fastapi import APIRouter, HTTPException, Query
from typing import List
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from fastapi.responses import JSONResponse

router = APIRouter()

DEFAULT_HISTORICAL_LOOKBACK_DAYS = 250

@router.get("/historical_prices")
async def get_historical_prices(
    symbol: str = Query(..., description="Stock ticker symbol (e.g., AAPL, GOOGL)"),
    lookback_days: int = Query(DEFAULT_HISTORICAL_LOOKBACK_DAYS, ge=1, description="Number of past days to fetch historical data for. Minimum 1.")
):
    """
    Fetches historical closing prices for a given stock symbol, and returns the prices along with the last available trading date.
    """
    if not symbol:
        raise HTTPException(status_code=400, detail="Stock symbol cannot be empty.")

    try:
        # Extend lookback to ensure we get enough trading days (account for weekends/holidays)
        start_date = datetime.now() - timedelta(days=lookback_days * 2)

        # Fetch data until the latest available date (don't set end date)
        data = yf.download(symbol, start=start_date, interval='1d', progress=False)

        if data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}.")

        close_series = None

        if isinstance(data.columns, pd.MultiIndex):
            if ('Close', symbol) in data.columns:
                close_series = data[('Close', symbol)]
            elif ('Adj Close', symbol) in data.columns:
                close_series = data[('Adj Close', symbol)]
            elif 'Close' in data.columns.get_level_values(0):
                if (data['Close'].columns == symbol).any():
                    close_series = data['Close'][symbol]
        if close_series is None and 'Close' in data.columns:
            close_series = data['Close']

        if close_series is None:
            raise HTTPException(status_code=500, detail="Could not extract closing prices.")

        close_series = pd.to_numeric(close_series, errors='coerce')
        close_series.dropna(inplace=True)

        if close_series.empty:
            raise HTTPException(status_code=400, detail="No valid data after cleaning.")

        historical_prices = close_series.values[-lookback_days:].tolist()
        historical_prices = [float(price) for price in historical_prices]

        # Use last available date from actual data
        last_date = close_series.index[-1]
        last_date_str = last_date.strftime("%Y-%m-%d")

        return {
            "historical_prices": historical_prices,
            "last_date": last_date_str
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data for {symbol}: {str(e)}")
