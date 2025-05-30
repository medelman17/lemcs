"""
Test suite for semantic similarity service for legal citation resolution.
Tests context extraction, embedding generation, and ambiguous citation matching.
"""
import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import uuid
from datetime import datetime
from typing import List

# Mock the OpenAI service before importing semantic similarity
with patch('nlp.openai_service.openai_service') as mock_openai:
    mock_openai.get_embedding = AsyncMock(return_value=np.random.rand(1536).tolist())
    
    from nlp.semantic_similarity import (
        SemanticSimilarityService, 
        SemanticMatch, 
        CitationContext,
        semantic_similarity_service
    )

from db.models import Citation


@pytest.fixture
def sample_legal_document_text():
    """Fixture providing sample legal document text for context testing"""
    return """
    The Supreme Court in Brown v. Board of Education, 347 U.S. 483 (1954), held that 
    racial segregation of children in public schools was unconstitutional. The Court stated 
    that "separate educational facilities are inherently unequal." This landmark decision 
    overruled the "separate but equal" doctrine from Plessy v. Ferguson, 163 U.S. 537 (1896).
    
    In a later case, the Court in Cooper v. Aaron, 358 U.S. 1 (1958), reaffirmed the Brown 
    decision. Id. at 17. The Arkansas officials had argued that they were not bound by the 
    federal court's desegregation order. Supra, at 16. The Court firmly rejected this 
    argument, stating that federal law is supreme over state law.
    
    The Court's reasoning in Brown, supra, has been applied in numerous subsequent cases.
    """


@pytest.fixture
def sample_citations():
    """Fixture providing sample citations for testing"""
    citations = [
        Citation(
            id=str(uuid.uuid4()),
            document_id="doc1",
            citation_text="Brown v. Board of Education, 347 U.S. 483 (1954)",
            citation_type="FullCaseCitation",
            page_number=1,
            paragraph_index=0,
            position_start=25,
            position_end=75,
            extracted_date=datetime.now(),
            metadata={"case_name": "Brown v. Board of Education"}
        ),
        Citation(
            id=str(uuid.uuid4()),
            document_id="doc1", 
            citation_text="Plessy v. Ferguson, 163 U.S. 537 (1896)",
            citation_type="FullCaseCitation",
            page_number=1,
            paragraph_index=0,
            position_start=250,
            position_end=290,
            extracted_date=datetime.now(),
            metadata={"case_name": "Plessy v. Ferguson"}
        ),
        Citation(
            id=str(uuid.uuid4()),
            document_id="doc1",
            citation_text="Id.",
            citation_type="IdCitation", 
            page_number=1,
            paragraph_index=1,
            position_start=450,
            position_end=453,
            extracted_date=datetime.now(),
            metadata={}
        ),
        Citation(
            id=str(uuid.uuid4()),
            document_id="doc1",
            citation_text="Supra",
            citation_type="SupraCitation",
            page_number=1,
            paragraph_index=1,
            position_start=550,
            position_end=555,
            extracted_date=datetime.now(),
            metadata={}
        ),
        Citation(
            id=str(uuid.uuid4()),
            document_id="doc1",
            citation_text="Brown, supra",
            citation_type="SupraCitation",
            page_number=1,
            paragraph_index=2,
            position_start=650,
            position_end=662,
            extracted_date=datetime.now(),
            metadata={"case_name": "Brown"}
        )
    ]
    return citations


class TestCitationContext:
    """Test citation context extraction functionality"""
    
    def test_extract_citation_context_basic(self, sample_legal_document_text):
        """Test basic context extraction around a citation"""
        service = SemanticSimilarityService()
        
        # Position of "Brown v. Board" citation
        position_start = 25
        position_end = 75
        
        context = service._extract_citation_context(
            sample_legal_document_text, 
            position_start, 
            position_end
        )
        
        assert context.before_citation is not None
        assert context.after_citation is not None
        assert "Supreme Court" in context.before_citation
        assert "held that" in context.after_citation
        assert context.sentence_context is not None
        assert "Brown v. Board of Education" in context.sentence_context
    
    def test_extract_citation_context_edge_cases(self):
        """Test context extraction edge cases (start/end of document)"""
        service = SemanticSimilarityService()
        
        # Citation at very beginning
        text = "Brown v. Board, 347 U.S. 483 (1954) was a landmark case."
        context = service._extract_citation_context(text, 0, 33)
        
        assert context.before_citation == ""
        assert "was a landmark" in context.after_citation
        
        # Citation at very end
        text = "The landmark case was Brown v. Board, 347 U.S. 483 (1954)"
        start = text.find("Brown")
        context = service._extract_citation_context(text, start, len(text))
        
        assert "landmark case was" in context.before_citation
        assert context.after_citation == ""
    
    def test_extract_semantic_features(self, sample_legal_document_text):
        """Test extraction of semantic features from context"""
        service = SemanticSimilarityService()
        
        context = CitationContext(
            before_citation="The Supreme Court in",
            after_citation="held that racial segregation",
            sentence_context="The Supreme Court in Brown v. Board held that racial segregation was unconstitutional",
            paragraph_context=sample_legal_document_text[:200]
        )
        
        features = service._extract_semantic_features(context)
        
        assert "Supreme Court" in features
        assert "held" in features
        assert "racial segregation" in features
        assert "unconstitutional" in features


class TestSemanticMatching:
    """Test semantic similarity matching functionality"""
    
    @pytest.mark.asyncio
    async def test_calculate_semantic_similarity_basic(self, sample_citations):
        """Test basic semantic similarity calculation"""
        service = SemanticSimilarityService()
        
        # Mock embeddings that are similar
        similar_embedding = np.random.rand(1536)
        different_embedding = np.random.rand(1536)
        
        with patch.object(service, '_get_citation_embedding') as mock_embedding:
            mock_embedding.side_effect = [similar_embedding, similar_embedding + 0.1]
            
            similarity = await service._calculate_semantic_similarity(
                sample_citations[0], sample_citations[1], "test document text"
            )
            
            assert 0 <= similarity <= 1
            assert similarity > 0.8  # Should be high due to similar embeddings
    
    @pytest.mark.asyncio
    async def test_find_best_semantic_matches(self, sample_citations):
        """Test finding best semantic matches for ambiguous citations"""
        service = SemanticSimilarityService()
        
        # Mock the embedding and similarity calculations
        with patch.object(service, '_calculate_semantic_similarity') as mock_similarity:
            # Return different similarity scores
            mock_similarity.side_effect = [0.9, 0.7, 0.6]
            
            reference_citation = sample_citations[2]  # "Id." citation
            candidate_citations = sample_citations[:2]  # Full citations
            
            matches = await service.find_best_semantic_matches(
                reference_citation,
                candidate_citations,
                "sample document text",
                threshold=0.5,
                max_matches=3
            )
            
            assert len(matches) == 2  # Should return 2 matches above threshold
            assert matches[0].similarity_score == 0.9
            assert matches[1].similarity_score == 0.7
            assert matches[0].target_citation_id == sample_citations[0].id
    
    @pytest.mark.asyncio
    async def test_semantic_matching_with_context(self, sample_citations, sample_legal_document_text):
        """Test semantic matching using legal context"""
        service = SemanticSimilarityService()
        
        with patch.object(service, '_get_citation_embedding') as mock_embedding:
            # Mock different embeddings for different citations
            mock_embedding.side_effect = [
                np.random.rand(1536),  # Reference citation embedding
                np.random.rand(1536),  # First candidate embedding  
                np.random.rand(1536)   # Second candidate embedding
            ]
            
            reference_citation = sample_citations[4]  # "Brown, supra"
            candidate_citations = sample_citations[:2]
            
            matches = await service.find_best_semantic_matches(
                reference_citation,
                candidate_citations, 
                sample_legal_document_text,
                threshold=0.3
            )
            
            assert len(matches) >= 1
            # The match with "Brown" in the name should score higher due to context
            brown_match = next((m for m in matches if "Brown" in candidate_citations[0].citation_text), None)
            assert brown_match is not None


class TestSemanticSimilarityService:
    """Test the main semantic similarity service"""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test service initializes correctly"""
        service = SemanticSimilarityService()
        assert service is not None
        assert hasattr(service, 'confidence_threshold')
        assert hasattr(service, 'max_context_chars')
    
    def test_singleton_instance(self):
        """Test that semantic_similarity_service is a singleton"""
        assert semantic_similarity_service is not None
        # Test that importing again gives same instance
        from nlp.semantic_similarity import semantic_similarity_service as service2
        assert semantic_similarity_service is service2
    
    @pytest.mark.asyncio
    async def test_performance_with_large_candidate_set(self):
        """Test performance with many candidate citations"""
        service = SemanticSimilarityService()
        
        # Create many candidate citations
        large_candidate_set = []
        for i in range(50):
            citation = Citation(
                id=str(uuid.uuid4()),
                document_id="doc1",
                citation_text=f"Case {i} v. Defendant, {100+i} U.S. {200+i} (2000)",
                citation_type="FullCaseCitation",
                page_number=1,
                paragraph_index=i,
                position_start=i*100,
                position_end=i*100+50,
                extracted_date=datetime.now(),
                metadata={"case_name": f"Case {i}"}
            )
            large_candidate_set.append(citation)
        
        reference_citation = Citation(
            id=str(uuid.uuid4()),
            document_id="doc1",
            citation_text="Id.",
            citation_type="IdCitation",
            page_number=1,
            paragraph_index=25,
            position_start=2500,
            position_end=2503,
            extracted_date=datetime.now(),
            metadata={}
        )
        
        with patch.object(service, '_calculate_semantic_similarity') as mock_similarity:
            # Mock similarity scores - decreasing for performance test
            mock_similarity.side_effect = [0.9 - (i * 0.01) for i in range(50)]
            
            matches = await service.find_best_semantic_matches(
                reference_citation,
                large_candidate_set,
                "Sample legal document text",
                threshold=0.7,
                max_matches=10
            )
            
            # Should efficiently return top matches within threshold
            assert len(matches) <= 10
            assert all(match.similarity_score >= 0.7 for match in matches)
            # Should be sorted by similarity score descending
            scores = [match.similarity_score for match in matches]
            assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 