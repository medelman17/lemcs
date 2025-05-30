"""
Test suite for citation relationship resolution functionality.
Tests the sophisticated citation matching algorithms implemented in citation_service.py.
"""
import pytest
import asyncio
from typing import List
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from nlp.citation_service import CitationExtractionService
from db.models import Citation, CitationRelationship


@pytest.fixture
def citation_service():
    """Fixture providing a CitationExtractionService instance"""
    return CitationExtractionService()


@pytest.fixture
def sample_citations():
    """Fixture providing sample citations for testing relationships"""
    citations = []
    
    # Full case citation - will be the target for references
    full_citation = Citation(
        id="full-citation-1",
        document_id="doc-1",
        citation_text="Brown v. Board of Education, 347 U.S. 483 (1954)",
        citation_type="FullCaseCitation",
        reporter="U.S.",
        volume="347",
        page="483",
        position_start=100,
        position_end=150,
        confidence_score=0.95,
        doc_metadata={
            "original_text": "Brown v. Board of Education, 347 U.S. 483 (1954)",
            "corrected_citation": "Brown v. Board of Education, 347 U.S. 483 (1954)",
            "citation_class": "FullCaseCitation",
            "groups": {"case_name": "Brown v. Board of Education"},
            "court": "Supreme Court",
            "year": "1954"
        },
        created_at=datetime.utcnow()
    )
    citations.append(full_citation)
    
    # Id citation - should reference the full citation
    id_citation = Citation(
        id="id-citation-1",
        document_id="doc-1",
        citation_text="Id. at 493",
        citation_type="IdCitation",
        reporter=None,
        volume=None,
        page="493",
        position_start=200,
        position_end=210,
        confidence_score=0.85,
        doc_metadata={
            "original_text": "Id. at 493",
            "corrected_citation": "Id. at 493",
            "citation_class": "IdCitation",
            "antecedent": {
                "text": "Brown v. Board of Education, 347 U.S. 483 (1954)",
                "score": 0.9
            }
        },
        created_at=datetime.utcnow()
    )
    citations.append(id_citation)
    
    # Supra citation - should reference the full citation
    supra_citation = Citation(
        id="supra-citation-1",
        document_id="doc-1",
        citation_text="Brown, supra, at 495",
        citation_type="SupraCitation",
        reporter=None,
        volume=None,
        page="495",
        position_start=300,
        position_end=320,
        confidence_score=0.80,
        doc_metadata={
            "original_text": "Brown, supra, at 495",
            "corrected_citation": "Brown, supra, at 495",
            "citation_class": "SupraCitation",
            "antecedent": {
                "text": "Brown v. Board of Education, 347 U.S. 483 (1954)",
                "score": 0.85
            }
        },
        created_at=datetime.utcnow()
    )
    citations.append(supra_citation)
    
    # Short citation - should reference the full citation
    short_citation = Citation(
        id="short-citation-1",
        document_id="doc-1",
        citation_text="Brown at 487",
        citation_type="ShortCaseCitation",
        reporter="U.S.",
        volume="347",
        page="487",
        position_start=400,
        position_end=415,
        confidence_score=0.88,
        doc_metadata={
            "original_text": "Brown at 487",
            "corrected_citation": "Brown at 487",
            "citation_class": "ShortCaseCitation"
        },
        created_at=datetime.utcnow()
    )
    citations.append(short_citation)
    
    return citations


@pytest.fixture
def mock_db_session():
    """Mock database session"""
    session = AsyncMock()
    return session


class TestCitationRelationshipResolver:
    """Test suite for citation relationship resolution"""
    
    @pytest.mark.asyncio
    async def test_create_citation_relationships_basic(self, citation_service, sample_citations, mock_db_session):
        """Test basic citation relationship creation"""
        relationships = await citation_service.create_citation_relationships(
            sample_citations, mock_db_session
        )
        
        # Should create relationships for id, supra, and short citations
        assert len(relationships) == 3
        
        # Check relationship types
        relationship_types = [r.relationship_type for r in relationships]
        assert "id_reference" in relationship_types
        assert "supra_reference" in relationship_types
        assert "short_form_reference" in relationship_types
    
    @pytest.mark.asyncio
    async def test_resolve_id_citation(self, citation_service, sample_citations):
        """Test resolution of Id. citations"""
        # Get citations
        full_citation = sample_citations[0]  # Brown v. Board
        id_citation = sample_citations[1]    # Id. at 493
        
        # Test id citation resolution
        target = citation_service._resolve_id_citation(id_citation, [full_citation])
        
        assert target is not None
        assert target.id == full_citation.id
        assert target.citation_type == "FullCaseCitation"
    
    @pytest.mark.asyncio
    async def test_resolve_supra_citation(self, citation_service, sample_citations):
        """Test resolution of supra citations"""
        # Build citation index
        citation_index = citation_service._build_citation_index(sample_citations)
        
        # Get supra citation
        supra_citation = sample_citations[2]  # Brown, supra, at 495
        
        # Test supra citation resolution
        target = citation_service._resolve_supra_citation(supra_citation, citation_index)
        
        assert target is not None
        assert target.citation_type == "FullCaseCitation"
        assert "Brown" in target.citation_text
    
    @pytest.mark.asyncio
    async def test_resolve_short_citation(self, citation_service, sample_citations):
        """Test resolution of short citations"""
        # Build citation index
        citation_index = citation_service._build_citation_index(sample_citations)
        
        # Get short citation
        short_citation = sample_citations[3]  # Brown at 487
        
        # Test short citation resolution
        target = citation_service._resolve_short_citation(short_citation, citation_index)
        
        assert target is not None
        assert target.citation_type == "FullCaseCitation"
        assert target.reporter == "U.S."
        assert target.volume == "347"
    
    def test_build_citation_index(self, citation_service, sample_citations):
        """Test citation index building"""
        index = citation_service._build_citation_index(sample_citations)
        
        # Check index structure
        assert "by_position" in index
        assert "by_reporter" in index
        assert "by_case_name" in index
        assert "full_citations" in index
        assert "reference_citations" in index
        
        # Check full citations
        assert len(index["full_citations"]) == 1
        assert index["full_citations"][0].citation_type == "FullCaseCitation"
        
        # Check reference citations
        assert len(index["reference_citations"]) == 3
        
        # Check reporter index
        assert "U.S._347" in index["by_reporter"]
        
        # Check case name index
        case_names = list(index["by_case_name"].keys())
        assert any("brown" in name.lower() for name in case_names)
    
    def test_is_reference_citation(self, citation_service, sample_citations):
        """Test reference citation identification"""
        full_citation = sample_citations[0]
        id_citation = sample_citations[1]
        supra_citation = sample_citations[2]
        short_citation = sample_citations[3]
        
        assert not citation_service._is_reference_citation(full_citation)
        assert citation_service._is_reference_citation(id_citation)
        assert citation_service._is_reference_citation(supra_citation)
        assert citation_service._is_reference_citation(short_citation)
    
    def test_extract_case_name(self, citation_service, sample_citations):
        """Test case name extraction from full citations"""
        full_citation = sample_citations[0]
        
        case_name = citation_service._extract_case_name(full_citation)
        
        assert case_name is not None
        assert "Brown" in case_name
    
    def test_extract_case_name_from_supra(self, citation_service):
        """Test case name extraction from supra citations"""
        # Test various supra patterns
        test_cases = [
            ("Brown, supra, at 495", "brown"),
            ("Smith v. Jones, supra", "smith v. jones"),
            ("Doe supra at 123", "doe"),
        ]
        
        for supra_text, expected in test_cases:
            result = citation_service._extract_case_name_from_supra(supra_text)
            assert result is not None
            assert expected in result.lower()
    
    def test_citations_match(self, citation_service):
        """Test citation text matching logic"""
        text1 = "Brown v. Board of Education, 347 U.S. 483 (1954)"
        text2 = "Brown v. Board of Education, 347 U.S. 483"
        text3 = "Smith v. Jones, 123 F.3d 456"
        
        # Should match similar citations
        assert citation_service._citations_match(text1, text2)
        
        # Should not match different citations
        assert not citation_service._citations_match(text1, text3)
        
        # Exact match
        assert citation_service._citations_match(text1, text1)
    
    def test_determine_relationship_type(self, citation_service, sample_citations):
        """Test relationship type determination"""
        full_citation = sample_citations[0]
        id_citation = sample_citations[1]
        supra_citation = sample_citations[2]
        short_citation = sample_citations[3]
        
        assert citation_service._determine_relationship_type(id_citation, full_citation) == "id_reference"
        assert citation_service._determine_relationship_type(supra_citation, full_citation) == "supra_reference"
        assert citation_service._determine_relationship_type(short_citation, full_citation) == "short_form_reference"
    
    def test_calculate_relationship_confidence(self, citation_service, sample_citations):
        """Test relationship confidence calculation"""
        full_citation = sample_citations[0]
        id_citation = sample_citations[1]
        
        confidence = citation_service._calculate_relationship_confidence(id_citation, full_citation)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably confident
    
    @pytest.mark.asyncio
    async def test_empty_citations_list(self, citation_service, mock_db_session):
        """Test behavior with empty citations list"""
        relationships = await citation_service.create_citation_relationships([], mock_db_session)
        assert relationships == []
    
    @pytest.mark.asyncio
    async def test_no_reference_citations(self, citation_service, mock_db_session):
        """Test behavior with only full citations (no references)"""
        # Create a list with only full citations
        full_citation = Citation(
            id="full-1",
            document_id="doc-1",
            citation_text="Brown v. Board, 347 U.S. 483 (1954)",
            citation_type="FullCaseCitation",
            position_start=100,
            position_end=150,
            created_at=datetime.utcnow()
        )
        
        relationships = await citation_service.create_citation_relationships([full_citation], mock_db_session)
        assert relationships == []


@pytest.mark.asyncio
async def test_integration_with_real_legal_text():
    """Integration test with realistic legal document text"""
    legal_text = """
    The warranty of habitability is implied in all residential leases in California. 
    Landford v. Tenant, 123 Cal. App. 4th 456 (2004). When a landlord fails to 
    maintain the premises in habitable condition, the tenant may seek damages. 
    Id. at 460. The court in Landford held that such damages may include rent 
    reduction. Landford, supra, at 465.
    """
    
    service = CitationExtractionService()
    
    # This test would require actual eyecite integration
    # For now, we just verify the service can be instantiated
    assert service is not None
    assert hasattr(service, 'create_citation_relationships')


if __name__ == "__main__":
    # Run a simple test
    asyncio.run(test_integration_with_real_legal_text()) 