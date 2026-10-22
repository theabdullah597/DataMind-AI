from dotenv import load_dotenv
import os

if os.path.exists("backend/.env"):
    load_dotenv("backend/.env")
else:
    load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.datasets import router as dataset_router
from app.api.eda import router as eda_router
from app.api.clean import router as clean_router
from app.api.ml import router as ml_router
from app.api.predict import router as predict_router
from app.api.explain import router as explain_router
from app.api.agent import router as agent_router


app = FastAPI(
    title="DataMind API",
    description="AI-Powered Data Science and Machine Learning Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(dataset_router)
app.include_router(eda_router)
app.include_router(clean_router)
app.include_router(ml_router)
app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(agent_router)


@app.get("/")
def root():
    return {
        "message": "DataMind API is running",
        "status": "success"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }