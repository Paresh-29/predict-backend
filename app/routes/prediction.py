from fastapi import APIRouter, HTTPException
from app.services.agent_service import query_aggregator_agent
from app.models.request_models import StockRequest
from app.models.response_models import StockResponse

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/", response_model=StockResponse)
async def predict_stock(request: StockRequest):
    try:
        result = query_aggregator_agent(request.stock_name)
        return StockResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
