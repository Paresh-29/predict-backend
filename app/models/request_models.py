
from pydantic import BaseModel

class StockRequest(BaseModel):
    stock_name: str
