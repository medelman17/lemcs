from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os

from api.routes import health, documents_simple
from config.settings_simple import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting LeMCS application...")
    yield
    print("Shutting down LeMCS application...")


app = FastAPI(
    title="LeMCS - Legal Memoranda Consolidation System",
    description="AI-powered document processing platform for consolidating legal memoranda",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(documents_simple.router, prefix="/api/v1", tags=["documents"])


if __name__ == "__main__":
    uvicorn.run("main_simple:app", host="0.0.0.0", port=8000, reload=True)
