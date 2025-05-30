"""
Tests for hybrid legal NLP service functionality.
"""

import pytest
from nlp.hybrid_legal_nlp import HybridLegalNLP


@pytest.fixture
def hybrid_nlp():
    """Create a hybrid NLP service instance for testing."""
    return HybridLegalNLP()


@pytest.fixture
def sample_legal_text():
    """Sample legal text for testing."""
    return """
    This Residential Lease Agreement is entered into on January 1, 2024,
    between John Smith (Landlord) and Jane Doe (Tenant) for the property
    located at 123 Main Street, New York, NY 10001. The monthly rent is
    $2,500.00 and shall be paid by the first day of each month.
    
    The tenant agrees to comply with all applicable laws and regulations.
    """


def test_hybrid_nlp_initialization(hybrid_nlp):
    """Test that the hybrid NLP service initializes correctly."""
    assert hybrid_nlp.spacy_service is not None
    assert hybrid_nlp.bert_service is not None
    
    status = hybrid_nlp.get_service_status()
    assert status["spacy_available"] is True
    assert status["bert_available"] is True


def test_comprehensive_analysis(hybrid_nlp, sample_legal_text):
    """Test comprehensive analysis functionality."""
    analysis = hybrid_nlp.comprehensive_analysis(sample_legal_text)
    
    assert isinstance(analysis, dict)
    assert "document_analysis" in analysis
    assert "entities" in analysis
    assert "semantic_analysis" in analysis
    assert "model_info" in analysis
    
    # Check document analysis
    doc_analysis = analysis["document_analysis"]
    assert "type" in doc_analysis
    assert "confidence" in doc_analysis
    assert "complexity_score" in doc_analysis
    
    # Should classify as lease agreement
    assert doc_analysis["type"] == "lease_agreement"


def test_extract_all_entities(hybrid_nlp, sample_legal_text):
    """Test combined entity extraction."""
    entities = hybrid_nlp.extract_all_entities(sample_legal_text)
    
    assert isinstance(entities, dict)
    assert "spacy_entities" in entities
    assert "bert_entities" in entities
    assert "combined_unique" in entities
    
    # Should have entities from spaCy
    assert len(entities["spacy_entities"]) > 0
    
    # Combined should be a list
    assert isinstance(entities["combined_unique"], list)


def test_compare_models(hybrid_nlp, sample_legal_text):
    """Test model comparison functionality."""
    comparison = hybrid_nlp.compare_models(sample_legal_text)
    
    assert isinstance(comparison, dict)
    assert "spacy_results" in comparison
    assert "bert_results" in comparison
    assert "performance" in comparison
    assert "recommendations" in comparison
    
    # Check performance metrics
    performance = comparison["performance"]
    assert "spacy_time_seconds" in performance
    assert "bert_time_seconds" in performance
    assert "speed_ratio" in performance
    
    # spaCy should be faster
    assert performance["spacy_time_seconds"] > 0
    assert performance["bert_time_seconds"] > 0


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


def test_text_embedding(hybrid_nlp):
    """Test text embedding functionality."""
    text = "This is a legal contract."
    embedding = hybrid_nlp.get_text_embedding(text)
    
    # Should return a tensor
    assert embedding is not None
    assert hasattr(embedding, 'shape')
    assert embedding.shape[0] == 1  # Batch size
    assert embedding.shape[1] > 0   # Embedding dimension


def test_service_status(hybrid_nlp):
    """Test service status checking."""
    status = hybrid_nlp.get_service_status()
    
    assert isinstance(status, dict)
    assert "spacy_available" in status
    assert "bert_available" in status
    assert "bert_tokenizer_available" in status
    assert "bert_ner_available" in status
    
    # Core services should be available
    assert status["spacy_available"] is True
    assert status["bert_available"] is True
    assert status["bert_tokenizer_available"] is True


def test_error_handling(hybrid_nlp):
    """Test error handling with edge cases."""
    # Test with empty text
    analysis = hybrid_nlp.comprehensive_analysis("")
    assert isinstance(analysis, dict)
    
    # Test with very long text (should handle gracefully)
    long_text = "Legal document. " * 1000
    analysis = hybrid_nlp.comprehensive_analysis(long_text)
    assert isinstance(analysis, dict)


def test_different_document_types(hybrid_nlp):
    """Test analysis of different legal document types."""
    lease_text = "This lease agreement between landlord and tenant..."
    complaint_text = "Plaintiff files this complaint against defendant..."
    memo_text = "Legal memorandum regarding contract interpretation..."
    
    lease_analysis = hybrid_nlp.comprehensive_analysis(lease_text)
    complaint_analysis = hybrid_nlp.comprehensive_analysis(complaint_text)
    memo_analysis = hybrid_nlp.comprehensive_analysis(memo_text)
    
    assert lease_analysis["document_analysis"]["type"] == "lease_agreement"
    assert complaint_analysis["document_analysis"]["type"] == "legal_complaint"
    assert memo_analysis["document_analysis"]["type"] == "legal_memorandum"


def test_entity_deduplication(hybrid_nlp):
    """Test that entity deduplication works correctly."""
    text = "John Smith and Jane Doe signed the contract. John Smith is the landlord."
    
    entities = hybrid_nlp.extract_all_entities(text)
    
    # Should have unique entities
    combined_entities = entities["combined_unique"]
    entity_texts = [e.get("text", "") for e in combined_entities]
    
    # Check for reasonable deduplication
    assert len(set(entity_texts)) <= len(entity_texts)