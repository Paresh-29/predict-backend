from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import prediction
from app.routes import lstm_only 
from app.routes import data_fetcher

from dotenv import load_dotenv
from contextlib import asynccontextmanager




load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI app starting up...")
    try:
        lstm_only.preload_all_models() 

    except Exception as e:
        print(f"Failed during application startup: {e}")
        raise RuntimeError(
            "Application startup failed due to essential component loading error."
        )
    yield
    print("FastAPI app shutting down.")


app = FastAPI(lifespan=lifespan)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include your routers
app.include_router(prediction.router)
app.include_router(lstm_only.router) 
app.include_router(data_fetcher.router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "Backend is running!"}