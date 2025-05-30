"""
Test suite for semantic similarity API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import uuid
from datetime import datetime

from main_simple import app
from db.database import get_db
from db.models import Base, Document, Citation, DocumentStatus


# Test database setup
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def test_db():
    """Create a test database session"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session_maker = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session_maker() as session:
        yield session
    
    await engine.dispose()


@pytest.fixture
def client(test_db):
    """Create a test client with overridden database"""
    async def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()


@pytest.fixture
async def sample_document(test_db):
    """Create a sample document for testing"""
    document = Document(
        id=str(uuid.uuid4()),
        filename="test_document.txt",
        file_size=1000,
        file_hash="test_hash",
        status=DocumentStatus.PROCESSED,
        extracted_text="""
        The Supreme Court held in Miranda v. Arizona, 384 U.S. 436 (1966) that 
        suspects must be informed of their rights before interrogation. This landmark
        decision established the Miranda warnings as a constitutional requirement.
        
        Later, in Dickerson v. United States, 530 U.S. 428 (2000), the Court reaffirmed
        Miranda's constitutional status, rejecting attempts to overturn it through legislation.
        
        The Fourth Amendment protections were further clarified in Terry v. Ohio, 392 U.S. 1 (1968),
        which established the reasonable suspicion standard for brief investigative stops.
        """,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    test_db.add(document)
    await test_db.commit()
    return document


@pytest.fixture
async def sample_citations(test_db, sample_document):
    """Create sample citations for testing"""
    citations = [
        Citation(
            id=str(uuid.uuid4()),
            document_id=sample_document.id,
            citation_text="Miranda v. Arizona, 384 U.S. 436 (1966)",
            citation_type="case",
            reporter="U.S.",
            volume="384",
            page="436",
            year="1966",
            position_start=31,
            position_end=69,
            confidence_score=0.95,
            created_at=datetime.utcnow()
        ),
        Citation(
            id=str(uuid.uuid4()),
            document_id=sample_document.id,
            citation_text="Dickerson v. United States, 530 U.S. 428 (2000)",
            citation_type="case",
            reporter="U.S.",
            volume="530",
            page="428",
            year="2000",
            position_start=245,
            position_end=293,
            confidence_score=0.95,
            created_at=datetime.utcnow()
        ),
        Citation(
            id=str(uuid.uuid4()),
            document_id=sample_document.id,
            citation_text="Terry v. Ohio, 392 U.S. 1 (1968)",
            citation_type="case",
            reporter="U.S.",
            volume="392",
            page="1",
            year="1968",
            position_start=456,
            position_end=488,
            confidence_score=0.95,
            created_at=datetime.utcnow()
        )
    ]
    
    for citation in citations:
        test_db.add(citation)
    await test_db.commit()
    
    return citations


class TestSemanticSimilarityAPI:
    """Test cases for semantic similarity endpoints"""
    
    def test_extract_citation_context(self, client, sample_citations):
        """Test citation context extraction endpoint"""
        citation = sample_citations[0]  # Miranda citation
        
        response = client.post(
            "/api/v1/semantic/context",
            json={
                "citation_id": citation.id,
                "context_window": 200
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["citation_id"] == citation.id
        assert data["citation_text"] == citation.citation_text
        assert "surrounding_text" in data
        assert isinstance(data["extracted_entities"], list)
        assert isinstance(data["legal_concepts"], list)
        assert isinstance(data["case_names"], list)
        assert isinstance(data["statutory_references"], list)
        assert 0.0 <= data["position_in_document"] <= 1.0
    
    def test_generate_citation_embedding(self, client, sample_citations):
        """Test citation embedding generation endpoint"""
        citation = sample_citations[0]
        
        response = client.post(
            "/api/v1/semantic/embedding",
            json={
                "citation_id": citation.id,
                "include_surrounding_context": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["citation_id"] == citation.id
        assert data["embedding_dimension"] == 1536  # OpenAI embedding size
        assert "cache_hit" in data
        assert "embedding_text_preview" in data
    
    def test_calculate_semantic_similarity(self, client, sample_citations):
        """Test semantic similarity calculation endpoint"""
        source_citation = sample_citations[0]  # Miranda
        target_citation = sample_citations[1]  # Dickerson
        
        response = client.post(
            "/api/v1/semantic/similarity",
            json={
                "source_citation_id": source_citation.id,
                "target_citation_id": target_citation.id,
                "include_context": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["source_citation_id"] == source_citation.id
        assert data["target_citation_id"] == target_citation.id
        assert 0.0 <= data["similarity_score"] <= 1.0
        assert 0.0 <= data["context_overlap"] <= 1.0
        assert 0.0 <= data["combined_confidence"] <= 1.0
        assert "match_reason" in data
        assert isinstance(data["semantic_features"], dict)
    
    def test_find_semantic_matches(self, client, sample_citations):
        """Test semantic match finding endpoint"""
        source_citation = sample_citations[0]  # Miranda
        
        response = client.post(
            "/api/v1/semantic/matches",
            json={
                "source_citation_id": source_citation.id,
                "threshold": 0.5,
                "max_matches": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        # Should find at least one match (Dickerson mentions Miranda)
        assert len(data) >= 1
        
        for match in data:
            assert match["source_citation_id"] == source_citation.id
            assert "target_citation_id" in match
            assert "target_citation_text" in match
            assert 0.0 <= match["similarity_score"] <= 1.0
            assert 0.0 <= match["combined_confidence"] <= 1.0
            assert "match_reason" in match
    
    def test_cleanup_cache(self, client):
        """Test cache cleanup endpoint"""
        response = client.post("/api/v1/semantic/cache/cleanup")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert data["embedding_cache_cleared"] is True
        assert data["context_cache_cleared"] is True
    
    def test_get_statistics(self, client):
        """Test statistics endpoint"""
        response = client.get("/api/v1/semantic/statistics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "embedding_cache_size" in data
        assert "context_cache_size" in data
        assert "cache_expiry_hours" in data
        assert "last_cache_cleanup" in data
        assert "supported_legal_concepts" in data
        assert data["embedding_dimension"] == 1536
    
    def test_citation_not_found(self, client):
        """Test error handling for non-existent citation"""
        fake_id = str(uuid.uuid4())
        
        response = client.post(
            "/api/v1/semantic/context",
            json={
                "citation_id": fake_id,
                "context_window": 500
            }
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_invalid_threshold(self, client, sample_citations):
        """Test validation for invalid threshold value"""
        response = client.post(
            "/api/v1/semantic/matches",
            json={
                "source_citation_id": sample_citations[0].id,
                "threshold": 1.5,  # Invalid: > 1.0
                "max_matches": 5
            }
        )
        
        assert response.status_code == 422  # Validation error