"""
Tests for Legal BERT service functionality.
"""

import pytest
import torch
from nlp.legal_bert_service import LegalBERTService


@pytest.fixture
def legal_bert_service():
    """Create a Legal BERT service instance for testing."""
    return LegalBERTService()


@pytest.fixture
def sample_legal_text():
    """Sample legal text for testing."""
    return """
    This Residential Lease Agreement is entered into on January 1, 2024,
    between John Smith (Landlord) and Jane Doe (Tenant) for the property
    located at 123 Main Street, New York, NY 10001. The monthly rent is
    $2,500.00 and shall be paid by the first day of each month.
    
    In accordance with 42 U.S.C. § 1983, tenant rights are protected.
    """


def test_legal_bert_service_initialization(legal_bert_service):
    """Test that the Legal BERT service initializes correctly."""
    assert legal_bert_service.model is not None
    assert legal_bert_service.tokenizer is not None
    assert legal_bert_service.model_name == "nlpaueb/legal-bert-base-uncased"


def test_encode_text(legal_bert_service, sample_legal_text):
    """Test text encoding functionality."""
    embeddings = legal_bert_service.encode_text(sample_legal_text)
    
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == 1  # Batch size of 1
    assert embeddings.shape[1] > 0   # Embedding dimension
    
    # Test with shorter text
    short_embeddings = legal_bert_service.encode_text("This is a contract.")
    assert isinstance(short_embeddings, torch.Tensor)
    assert short_embeddings.shape[1] == embeddings.shape[1]  # Same embedding dim


def test_text_similarity(legal_bert_service):
    """Test semantic similarity calculation."""
    text1 = "This is a lease agreement between landlord and tenant."
    text2 = "The rental contract is between property owner and renter."
    text3 = "The weather is nice today."
    
    # Similar legal texts should have higher similarity
    sim_legal = legal_bert_service.calculate_text_similarity(text1, text2)
    sim_unrelated = legal_bert_service.calculate_text_similarity(text1, text3)
    
    assert 0 <= sim_legal <= 1
    assert 0 <= sim_unrelated <= 1
    assert sim_legal > sim_unrelated  # Legal texts should be more similar


def test_classify_legal_text(legal_bert_service, sample_legal_text):
    """Test legal text classification."""
    classification = legal_bert_service.classify_legal_text(sample_legal_text)
    
    assert isinstance(classification, dict)
    assert "document_type" in classification
    assert "confidence" in classification
    assert "legal_domain" in classification
    assert "complexity_indicators" in classification
    
    # Should classify as lease agreement
    assert classification["document_type"] == "lease_agreement"
    assert classification["confidence"] > 0.5
    assert "real_estate" in classification["legal_domain"]


def test_classify_different_document_types(legal_bert_service):
    """Test classification of different legal document types."""
    complaint_text = "Plaintiff hereby files this complaint against defendant for breach of contract."
    memo_text = "Legal memorandum regarding the interpretation of contract clause 5.2."
    contract_text = "This agreement is entered into between the parties for services."
    
    complaint_result = legal_bert_service.classify_legal_text(complaint_text)
    memo_result = legal_bert_service.classify_legal_text(memo_text)
    contract_result = legal_bert_service.classify_legal_text(contract_text)
    
    assert complaint_result["document_type"] == "legal_complaint"
    assert memo_result["document_type"] == "legal_memorandum"
    assert contract_result["document_type"] == "contract"


def test_extract_legal_concepts(legal_bert_service, sample_legal_text):
    """Test legal concept extraction."""
    concepts = legal_bert_service.extract_legal_concepts(sample_legal_text)
    
    assert isinstance(concepts, dict)
    assert "legal_terms" in concepts
    assert "parties" in concepts
    assert "dates" in concepts
    assert "amounts" in concepts
    assert "citations" in concepts
    assert "statutes" in concepts
    
    # Should find legal terms
    assert len(concepts["legal_terms"]) > 0
    
    # Should find statute reference
    assert len(concepts["statutes"]) > 0


def test_get_model_info(legal_bert_service):
    """Test model information retrieval."""
    info = legal_bert_service.get_model_info()
    
    assert isinstance(info, dict)
    assert "model_name" in info
    assert "tokenizer_vocab_size" in info
    assert "model_type" in info
    assert "ner_available" in info
    
    assert info["model_name"] == "nlpaueb/legal-bert-base-uncased"
    assert isinstance(info["tokenizer_vocab_size"], int)
    assert info["tokenizer_vocab_size"] > 0


def test_legal_entities_bert_fallback(legal_bert_service, sample_legal_text):
    """Test Legal BERT entity extraction (may not be available)."""
    entities = legal_bert_service.extract_legal_entities_bert(sample_legal_text)
    
    # Should return a list (empty if NER pipeline not available)
    assert isinstance(entities, list)
    
    # If entities are found, check structure
    for entity in entities:
        assert "text" in entity
        assert "label" in entity
        assert "confidence" in entity
        assert "start" in entity
        assert "end" in entity


def test_embedding_consistency(legal_bert_service):
    """Test that embeddings are consistent for the same text."""
    text = "This is a legal contract between two parties."
    
    embedding1 = legal_bert_service.encode_text(text)
    embedding2 = legal_bert_service.encode_text(text)
    
    # Embeddings should be identical for the same text
    torch.testing.assert_close(embedding1, embedding2)


def test_empty_text_handling(legal_bert_service):
    """Test handling of edge cases like empty text."""
    empty_result = legal_bert_service.classify_legal_text("")
    assert empty_result["document_type"] == "unknown"
    
    # Test very short text
    short_result = legal_bert_service.classify_legal_text("Contract.")
    assert "document_type" in short_result