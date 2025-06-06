from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import prediction


from dotenv import load_dotenv

load_dotenv()  # Ensure this is called before accessing environment variables

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routes
app.include_router(prediction.router)


@app.get("/")
async def root():
    return {"message": "Backend is running!"}
