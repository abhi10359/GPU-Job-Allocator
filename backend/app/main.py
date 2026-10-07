from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.jobs import router as jobs_router
from app.api.resources import router as resources_router
from app.api.analytics import router as analytics_router


app = FastAPI(
    title="GPU Job Allocator",
    description="Fragmentation-Aware GPU Resource Management System",
    version="1.0.0"
)


# Allow the React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Routers
app.include_router(jobs_router)
app.include_router(resources_router)
app.include_router(analytics_router)


@app.get("/")
def root():
    return {
        "message": "GPU Job Allocator API is running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "GPU Job Allocator"
    }