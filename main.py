from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from api.routes import consolidation, documents, health
from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting LeMCS application...")
    yield
    print("Shutting down LeMCS application...")


app = FastAPI(
    title="LeMCS - Legal Memoranda Consolidation System",
    description="AI-powered document processing platform for consolidating legal memoranda",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(consolidation.router, prefix="/api/v1", tags=["consolidation"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )