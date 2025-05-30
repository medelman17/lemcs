"""
Tests for legal NLP service functionality.
"""

import pytest
from nlp.legal_nlp_service import LegalNLPService


@pytest.fixture
def nlp_service():
    """Create a legal NLP service instance for testing."""
    return LegalNLPService()


@pytest.fixture
def sample_legal_text():
    """Sample legal text for testing."""
    return """
    This Residential Lease Agreement is entered into on January 1, 2024,
    between John Smith (Landlord) and Jane Doe (Tenant) for the property
    located at 123 Main Street, New York, NY 10001. The monthly rent is
    $2,500.00 and shall be paid by the first day of each month.
    """


def test_nlp_service_initialization(nlp_service):
    """Test that the NLP service initializes correctly."""
    assert nlp_service.nlp is not None
    assert nlp_service.nlp.lang == "en"


def test_extract_entities(nlp_service, sample_legal_text):
    """Test entity extraction functionality."""
    entities = nlp_service.extract_entities(sample_legal_text)
    
    assert isinstance(entities, list)
    assert len(entities) > 0
    
    # Check entity structure
    for entity in entities:
        assert "text" in entity
        assert "label" in entity
        assert "start" in entity
        assert "end" in entity


def test_extract_legal_entities(nlp_service, sample_legal_text):
    """Test legal-specific entity extraction."""
    legal_entities = nlp_service.extract_legal_entities(sample_legal_text)
    
    assert isinstance(legal_entities, dict)
    assert "organizations" in legal_entities
    assert "persons" in legal_entities
    assert "locations" in legal_entities
    assert "dates" in legal_entities
    assert "money" in legal_entities
    
    # Should find persons
    assert len(legal_entities["persons"]) > 0
    
    # Should find money amounts
    assert len(legal_entities["money"]) > 0


def test_analyze_document_structure(nlp_service, sample_legal_text):
    """Test document structure analysis."""
    analysis = nlp_service.analyze_document_structure(sample_legal_text)
    
    assert isinstance(analysis, dict)
    assert "sentence_count" in analysis
    assert "word_count" in analysis
    assert "paragraph_count" in analysis
    assert "key_phrases" in analysis
    assert "document_type" in analysis
    assert "complexity_score" in analysis
    
    # Should classify as lease agreement
    assert analysis["document_type"] == "lease_agreement"
    
    # Should have reasonable counts
    assert analysis["sentence_count"] > 0
    assert analysis["word_count"] > 0


def test_legal_patterns_extraction(nlp_service):
    """Test extraction of legal patterns."""
    text_with_citations = """
    According to 42 U.S.C. § 1983, the plaintiff has a valid claim.
    See also Smith v. Jones, 123 F.3d 456 (2d Cir. 2020).
    """
    
    legal_entities = nlp_service.extract_legal_entities(text_with_citations)
    
    # Should find legal concepts
    assert len(legal_entities["legal_concepts"]) > 0
    assert "plaintiff" in [concept.lower() for concept in legal_entities["legal_concepts"]]


def test_document_type_classification(nlp_service):
    """Test document type classification."""
    lease_text = "This lease agreement between landlord and tenant..."
    complaint_text = "Plaintiff hereby files this complaint against defendant..."
    memo_text = "Legal memorandum regarding the case..."
    
    lease_analysis = nlp_service.analyze_document_structure(lease_text)
    complaint_analysis = nlp_service.analyze_document_structure(complaint_text)
    memo_analysis = nlp_service.analyze_document_structure(memo_text)
    
    assert lease_analysis["document_type"] == "lease_agreement"
    assert complaint_analysis["document_type"] == "legal_complaint"
    assert memo_analysis["document_type"] == "legal_memorandum"


def test_complexity_calculation(nlp_service):
    """Test document complexity calculation."""
    simple_text = "This is a simple sentence."
    complex_text = """
    The aforementioned contractual obligations necessitate comprehensive
    analysis of the multifaceted legal ramifications pertaining to the
    jurisdictional complexities inherent in this unprecedented litigation.
    """
    
    simple_analysis = nlp_service.analyze_document_structure(simple_text)
    complex_analysis = nlp_service.analyze_document_structure(complex_text)
    
    assert simple_analysis["complexity_score"] < complex_analysis["complexity_score"]
    assert 0 <= simple_analysis["complexity_score"] <= 10
    assert 0 <= complex_analysis["complexity_score"] <= 10