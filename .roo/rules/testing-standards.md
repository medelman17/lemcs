---
description: Testing standards and practices for legal document processing system
globs: ["tests/**/*.py", "**/*test*.py", "**/*spec*.py"]
alwaysApply: false
---

# Testing Standards

## Testing Framework
- Use pytest with async support
- Comprehensive coverage across all legal NLP components
- Real legal document testing when appropriate

## Test Structure

### File Organization
```
tests/
├── test_hybrid_nlp.py      # Hybrid NLP service tests
├── test_legal_bert.py      # LEGAL-BERT specific tests
├── test_legal_nlp.py       # spaCy service tests
├── test_basic.py           # Basic application tests
└── fixtures/               # Test data and fixtures
    ├── legal_documents/    # Sample legal documents
    └── expected_results/   # Expected analysis results
```

### Test Naming Conventions
```python
def test_hybrid_nlp_comprehensive_analysis():
    """Test comprehensive analysis of legal documents."""
    
def test_document_classification_lease_agreement():
    """Test classification of lease agreement documents."""
    
def test_entity_extraction_with_legal_concepts():
    """Test extraction of legal-specific entities."""
```

## Legal NLP Testing Patterns

### Service Initialization Tests
```python
import pytest
from nlp.hybrid_legal_nlp import HybridLegalNLP

@pytest.fixture
def hybrid_nlp():
    """Create a hybrid NLP service instance for testing."""
    return HybridLegalNLP()

def test_hybrid_nlp_initialization(hybrid_nlp):
    """Test that the hybrid NLP service initializes correctly."""
    assert hybrid_nlp.spacy_service is not None
    assert hybrid_nlp.bert_service is not None
    
    status = hybrid_nlp.get_service_status()
    assert status["spacy_available"] is True
    assert status["bert_available"] is True
```

### Document Processing Tests
```python
@pytest.fixture
def sample_legal_text():
    """Sample legal text for testing."""
    return """
    This Residential Lease Agreement is entered into on January 1, 2024,
    between John Smith (Landlord) and Jane Doe (Tenant) for the property
    located at 123 Main Street, New York, NY 10001. The monthly rent is
    $2,500.00 and shall be paid by the first day of each month.
    """

def test_comprehensive_analysis(hybrid_nlp, sample_legal_text):
    """Test comprehensive analysis functionality."""
    analysis = hybrid_nlp.comprehensive_analysis(sample_legal_text)
    
    assert isinstance(analysis, dict)
    assert "document_analysis" in analysis
    assert "entities" in analysis
    assert "semantic_analysis" in analysis
    
    # Should classify as lease agreement
    doc_analysis = analysis["document_analysis"]
    assert doc_analysis["type"] == "lease_agreement"
    assert doc_analysis["confidence"] > 0.5
```

### Entity Extraction Tests
```python
def test_extract_all_entities(hybrid_nlp, sample_legal_text):
    """Test combined entity extraction."""
    entities = hybrid_nlp.extract_all_entities(sample_legal_text)
    
    assert isinstance(entities, dict)
    assert "spacy_entities" in entities
    assert "bert_entities" in entities
    assert "combined_unique" in entities
    
    # Should have entities from spaCy
    assert len(entities["spacy_entities"]) > 0
    
    # Combined should be deduplicated
    combined = entities["combined_unique"]
    assert isinstance(combined, list)
```

### Document Classification Tests
```python
def test_document_type_classification(hybrid_nlp):
    """Test classification of different legal document types."""
    test_cases = [
        ("This lease agreement between landlord and tenant...", "lease_agreement"),
        ("Plaintiff files this complaint against defendant...", "legal_complaint"),
        ("Legal memorandum regarding contract interpretation...", "legal_memorandum"),
        ("This agreement is entered into between parties...", "contract")
    ]
    
    for text, expected_type in test_cases:
        analysis = hybrid_nlp.comprehensive_analysis(text)
        assert analysis["document_analysis"]["type"] == expected_type
```

### Performance Tests
```python
import time

def test_processing_speed(hybrid_nlp, sample_legal_text):
    """Test that processing completes within acceptable time."""
    start_time = time.time()
    analysis = hybrid_nlp.comprehensive_analysis(sample_legal_text)
    processing_time = time.time() - start_time
    
    # Should process under 1 second for typical documents
    assert processing_time < 1.0
    assert analysis is not None

def test_model_comparison_performance(hybrid_nlp, sample_legal_text):
    """Test performance comparison between models."""
    comparison = hybrid_nlp.compare_models(sample_legal_text)
    
    assert "performance" in comparison
    perf = comparison["performance"]
    assert perf["spacy_time_seconds"] > 0
    assert perf["bert_time_seconds"] > 0
    
    # spaCy should be faster
    assert perf["spacy_time_seconds"] < perf["bert_time_seconds"]
```

### Error Handling Tests
```python
def test_empty_text_handling(hybrid_nlp):
    """Test handling of edge cases like empty text."""
    empty_result = hybrid_nlp.comprehensive_analysis("")
    assert empty_result["document_analysis"]["type"] == "unknown"

def test_very_long_text_handling(hybrid_nlp):
    """Test handling of very long documents."""
    long_text = "Legal document text. " * 1000
    analysis = hybrid_nlp.comprehensive_analysis(long_text)
    assert isinstance(analysis, dict)
    assert "document_analysis" in analysis

def test_model_failure_graceful_handling():
    """Test graceful handling when models fail to load."""
    # Mock model loading failure
    with pytest.raises(Exception):
        # Test that appropriate exceptions are raised
        pass
```

### Semantic Similarity Tests
```python
def test_similarity_calculation(hybrid_nlp):
    """Test semantic similarity using LEGAL-BERT."""
    text1 = "This lease agreement is between landlord and tenant."
    text2 = "The rental contract involves property owner and renter."
    text3 = "The weather is sunny today."
    
    # Similar legal texts should have higher similarity
    sim_legal = hybrid_nlp.calculate_similarity(text1, text2)
    sim_unrelated = hybrid_nlp.calculate_similarity(text1, text3)
    
    assert 0 <= sim_legal <= 1
    assert 0 <= sim_unrelated <= 1
    assert sim_legal > sim_unrelated

def test_embedding_consistency(hybrid_nlp):
    """Test that embeddings are consistent for the same text."""
    text = "This is a legal contract between two parties."
    
    embedding1 = hybrid_nlp.get_text_embedding(text)
    embedding2 = hybrid_nlp.get_text_embedding(text)
    
    # Embeddings should be identical for the same text
    import torch
    torch.testing.assert_close(embedding1, embedding2)
```

## API Testing Patterns

### FastAPI Endpoint Tests
```python
import pytest
from fastapi.testclient import TestClient
from main_simple import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_document_upload_endpoint():
    """Test document upload and processing."""
    # Mock legal document file
    files = {"file": ("test.txt", "Legal document content", "text/plain")}
    
    response = client.post("/documents/upload", files=files)
    assert response.status_code == 200
    
    result = response.json()
    assert "analysis" in result
    assert result["analysis"]["document_analysis"]["type"] in [
        "lease_agreement", "legal_complaint", "legal_memorandum", 
        "contract", "unknown"
    ]
```

### Database Testing
```python
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import get_database

@pytest.mark.asyncio
async def test_document_storage():
    """Test storing legal documents in database."""
    async with AsyncSession() as session:
        # Test document creation and retrieval
        document_data = {
            "content": "Legal document text",
            "document_type": "lease_agreement",
            "metadata": {"confidence": 0.9}
        }
        
        # Store and retrieve document
        # Assert proper storage and retrieval
        pass
```

## Test Data Management

### Sample Legal Documents
```python
# Use realistic but anonymized legal text
SAMPLE_LEASE = """
This Residential Lease Agreement ("Agreement") is entered into on [DATE],
between [LANDLORD] ("Landlord") and [TENANT] ("Tenant") for the premises
located at [ADDRESS]. The monthly rent is $[AMOUNT] due on the first day
of each month.
"""

SAMPLE_COMPLAINT = """
Plaintiff [PLAINTIFF] hereby files this Complaint against Defendant
[DEFENDANT] for breach of contract and seeks damages in the amount of
$[AMOUNT] plus costs and attorneys' fees.
"""
```

### Fixtures for Complex Tests
```python
@pytest.fixture
def mock_legal_documents():
    """Provide various legal document types for testing."""
    return {
        "lease": SAMPLE_LEASE,
        "complaint": SAMPLE_COMPLAINT,
        "memo": "Legal memorandum regarding...",
        "contract": "This agreement between parties..."
    }

@pytest.fixture
def expected_analysis_results():
    """Expected results for test documents."""
    return {
        "lease": {"type": "lease_agreement", "confidence": 0.9},
        "complaint": {"type": "legal_complaint", "confidence": 0.85},
        "memo": {"type": "legal_memorandum", "confidence": 0.85},
        "contract": {"type": "contract", "confidence": 0.8}
    }
```

## Test Commands

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_hybrid_nlp.py -v
pytest tests/test_legal_bert.py -v
pytest tests/test_legal_nlp.py -v

# Run with coverage
pytest tests/ --cov=nlp --cov-report=html

# Run performance tests
pytest tests/ -k "performance" -v
```

### Test Quality Requirements
- Minimum 90% test coverage for legal NLP components
- All tests must pass before commits
- Include both unit and integration tests
- Test with realistic legal document samples
- Validate entity extraction accuracy
- Test error handling and edge cases

## Mock and Fixture Patterns

### Mocking Expensive Operations
```python
@pytest.fixture
def mock_bert_model():
    """Mock LEGAL-BERT model for faster tests."""
    with patch('nlp.legal_bert_service.AutoModel') as mock:
        mock.from_pretrained.return_value = MagicMock()
        yield mock

def test_with_mocked_model(mock_bert_model):
    """Test functionality without loading actual models."""
    # Test logic without expensive model loading
    pass
```

Remember: Legal accuracy is critical - test with real legal documents when possible and validate results carefully!

## Reference Files
@tests/test_hybrid_nlp.py
@tests/test_legal_bert.py
@tests/test_legal_nlp.py
@tests/test_basic.py
@pyproject.toml