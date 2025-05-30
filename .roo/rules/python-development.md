---
description: Python development standards and patterns for LeMCS legal document processing
globs: ["**/*.py", "!tests/**/*.py"]
alwaysApply: false
---

# Python Development Standards

## Language and Version
- Use Python 3.12+ features and best practices
- All dependencies must be Python 3.12 compatible
- Follow async/await patterns for I/O operations

## Code Style and Quality
```python
# Use type hints throughout
from typing import Dict, List, Any, Optional
import asyncio

async def process_legal_document(
    text: str, 
    nlp_service: HybridLegalNLP
) -> Dict[str, Any]:
    """Process legal document with comprehensive analysis."""
    try:
        analysis = nlp_service.comprehensive_analysis(text)
        return {"status": "success", "analysis": analysis}
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise
```

## Key Dependencies (Python 3.12 Compatible)
- **FastAPI + Uvicorn**: Web framework
- **LangGraph**: Agent orchestration
- **spaCy 3.8.7**: Fast NLP processing
- **transformers 4.52.3 + torch 2.7.0**: LEGAL-BERT
- **eyecite 2.7.5**: Legal citation extraction
- **PostgreSQL + pgvector**: Semantic search_files
- **asyncpg**: Async database operations

## Async/Await Patterns

### Database Operations
```python
async def store_legal_document(
    db: AsyncSession,
    document_data: Dict[str, Any]
) -> Document:
    """Store legal document with proper async handling."""
    try:
        document = Document(**document_data)
        db.add(document)
        await db.commit()
        await db.refresh(document)
        return document
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to store document: {e}")
        raise
```

### API Endpoints
```python
@router.post("/analyze-legal-document")
async def analyze_legal_document(
    document: UploadFile = File(...),
    nlp_service: HybridLegalNLP = Depends(get_hybrid_nlp)
) -> Dict[str, Any]:
    """Analyze legal document with hybrid NLP."""
    try:
        text = await extract_document_text(document)
        analysis = nlp_service.comprehensive_analysis(text)
        return {"status": "success", "analysis": analysis}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Legal analysis failed: {e}"
        )
```

## Error Handling Patterns

### Legal Document Processing
```python
def process_with_fallback(text: str) -> Dict[str, Any]:
    """Process legal text with graceful fallbacks."""
    try:
        # Primary processing with hybrid NLP
        return nlp.comprehensive_analysis(text)
    except ModelLoadError:
        logger.warning("Hybrid NLP unavailable, using spaCy fallback")
        return spacy_service.analyze_document_structure(text)
    except Exception as e:
        logger.error(f"All processing failed: {e}")
        return {"error": str(e), "document_type": "unknown"}
```

### Database Error Handling
```python
async def retry_database_operation(operation, max_retries: int = 3):
    """Retry database operations with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await operation()
        except ConnectionError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            logger.warning(f"DB connection failed, retrying in {wait_time}s")
            await asyncio.sleep(wait_time)
```

## Performance Optimization

### Model Caching
```python
# Cache expensive model instances
@lru_cache(maxsize=1)
def get_hybrid_nlp() -> HybridLegalNLP:
    """Get cached hybrid NLP instance."""
    return HybridLegalNLP()

# Use dependency injection for consistent instances
async def get_nlp_service() -> HybridLegalNLP:
    """FastAPI dependency for NLP service."""
    return get_hybrid_nlp()
```

### Batch Processing
```python
async def process_documents_batch(
    documents: List[str],
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """Process multiple documents efficiently."""
    nlp = get_hybrid_nlp()
    results = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_results = [
            nlp.comprehensive_analysis(doc) 
            for doc in batch
        ]
        results.extend(batch_results)
        
        # Allow other coroutines to run
        await asyncio.sleep(0)
    
    return results
```

## Logging and Monitoring

### Structured Logging
```python
import structlog

logger = structlog.get_logger()

# Log with legal document context
logger.info(
    "Processing legal document",
    document_type=analysis["document_analysis"]["type"],
    entity_count=len(analysis["entities"]["combined_unique"]),
    processing_time=elapsed_time
)
```

### Performance Monitoring
```python
import time
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(
                f"{func.__name__} completed",
                elapsed_time=elapsed,
                status="success"
            )
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"{func.__name__} failed",
                elapsed_time=elapsed,
                error=str(e),
                status="error"
            )
            raise
    return wrapper
```

## Security Considerations

### Environment Variables
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Never hardcode secrets
settings = Settings()
```

### Input Validation
```python
from pydantic import BaseModel, validator

class LegalDocumentRequest(BaseModel):
    content: str
    document_type: Optional[str] = None
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Document content cannot be empty')
        if len(v) > 1_000_000:  # 1MB text limit
            raise ValueError('Document too large')
        return v
```

## Common Anti-Patterns to Avoid

```python
# ❌ Don't block the event loop
def blocking_nlp_processing(text):
    return nlp.comprehensive_analysis(text)  # Blocks

# ✅ Use async properly
async def async_nlp_processing(text):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, nlp.comprehensive_analysis, text
    )

# ❌ Don't create new model instances repeatedly
def analyze_document(text):
    nlp = HybridLegalNLP()  # Expensive!
    return nlp.comprehensive_analysis(text)

# ✅ Use cached instances
def analyze_document(text):
    nlp = get_hybrid_nlp()  # Cached
    return nlp.comprehensive_analysis(text)
```

## Reference Files
@config/settings.py
@main_simple.py
@requirements.txt