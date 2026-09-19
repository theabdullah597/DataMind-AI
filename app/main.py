from fastapi import FastAPI

from app.api.datasets import router as dataset_router


app = FastAPI(
    title="DataMind API",
    description="AI-Powered Data Science and Machine Learning Agent",
    version="1.0.0"
)


app.include_router(dataset_router)


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