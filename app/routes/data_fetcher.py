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
    """
    if not symbol:
        raise HTTPException(status_code=400, detail="Stock symbol cannot be empty.")

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days * 1.5) 

        print(f"DEBUG: Attempting to download data for {symbol} from {start_date.date()} to {end_date.date()}")
        data = yf.download(symbol, start=start_date, end=end_date, interval='1d', progress=False)

        print(f"DEBUG: yf.download result for {symbol}. Is empty: {data.empty}. Columns: {data.columns.tolist()}")

        if data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol} within the specified period. Please check the symbol or date range.")

      
        close_series = None 

        if isinstance(data.columns, pd.MultiIndex):
            if ('Close', symbol) in data.columns:
                close_series = data[('Close', symbol)]
                print(f"DEBUG: Extracted close_series using MultiIndex (('Close', '{symbol}')) for {symbol}.")
            elif ('Adj Close', symbol) in data.columns: 
                close_series = data[('Adj Close', symbol)]
                print(f"DEBUG: Extracted close_series using MultiIndex (('Adj Close', '{symbol}')) for {symbol}.")
            else:
              
                if 'Close' in data.columns.get_level_values(0): 
                    if (data['Close'].columns == symbol).any(): 
                         close_series = data['Close'][symbol]
                         print(f"DEBUG: Extracted close_series using data['Close'][symbol] for {symbol}.")
        
        if close_series is None and 'Close' in data.columns:
            close_series = data['Close']
            print(f"DEBUG: Extracted close_series using simple 'Close' column for {symbol}.")

        if close_series is None:
            raise HTTPException(status_code=500, detail=f"Could not find a valid 'Close' price column for {symbol} in any expected format. Columns found: {data.columns.tolist()}")


        print(f"DEBUG: Before pd.to_numeric for {symbol}:")
        print(f"DEBUG: Type of close_series: {type(close_series)}")
        if isinstance(close_series, pd.Series):
            print(f"DEBUG: close_series dtype: {close_series.dtype}")
            print(f"DEBUG: close_series head:\n{close_series.head()}")
            print(f"DEBUG: close_series shape: {close_series.shape}")
        else:
            print(f"DEBUG: close_series is NOT a pandas Series. Value: {close_series}")


        close_series = pd.to_numeric(close_series, errors='coerce')
        close_series.dropna(inplace=True)

        if close_series.empty:
            raise HTTPException(status_code=400, detail=f"No valid numerical historical data found for {symbol} after cleaning. Please check the symbol or date range.")

        historical_prices = [float(price) for price in close_series.values.tolist()]

        if len(historical_prices) < lookback_days:
             raise HTTPException(status_code=400, detail=f"Not enough historical data ({len(historical_prices)} trading days) found for {symbol} for the requested {lookback_days} days after cleaning. Try a smaller lookback period or a different symbol.")
        elif len(historical_prices) > lookback_days:
            historical_prices = historical_prices[-lookback_days:]
        
        print(f"DEBUG: Successfully prepared {len(historical_prices)} prices for {symbol}.")
        return historical_prices

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"ERROR: Unhandled exception in get_historical_prices for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data for {symbol}: {str(e)}")