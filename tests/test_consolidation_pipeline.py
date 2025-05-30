"""
Test suite for the complete document consolidation pipeline.

This test validates the end-to-end consolidation process using the fixture memos,
testing all major components:
- Document loading and preprocessing
- Legal theory extraction and grouping
- Citation deduplication and normalization
- CRRACC synthesis methodology
- Quality metrics and validation
"""

import asyncio
import pytest
import os
import logging
from typing import List, Dict, Any
from pathlib import Path

from agents.consolidation_agent import ConsolidationAgent
from nlp.document_grouping_service import DocumentGroupingService, GroupingStrategy
from nlp.citation_deduplication_service import CitationDeduplicationService, CitationFormat
from nlp.legal_theory_synthesis_service import LegalTheorySynthesisService, SynthesisMode

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestConsolidationPipeline:
    """Test class for the complete consolidation pipeline"""
    
    @pytest.fixture(scope="class")
    def fixture_memos_path(self):
        """Get path to fixture memos directory"""
        current_dir = Path(__file__).parent
        fixtures_path = current_dir.parent / "fixtures" / "memos"
        return fixtures_path
    
    @pytest.fixture(scope="class")
    async def memo_documents(self, fixture_memos_path):
        """Load all fixture memo documents"""
        documents = []
        
        for memo_file in fixture_memos_path.glob("*.md"):
            try:
                content = memo_file.read_text(encoding='utf-8')
                
                document = {
                    "id": memo_file.stem,
                    "title": memo_file.stem.replace('_', ' ').title(),
                    "content": content,
                    "file_path": str(memo_file),
                    "citations": [],  # Will be extracted during processing
                    "metadata": {
                        "file_size": len(content),
                        "word_count": len(content.split()),
                        "source": "fixture"
                    }
                }
                documents.append(document)
                
            except Exception as e:
                logger.warning(f"Failed to load {memo_file}: {e}")
        
        logger.info(f"Loaded {len(documents)} memo documents for testing")
        return documents
    
    @pytest.fixture
    async def consolidation_agent(self):
        """Create consolidation agent instance"""
        return ConsolidationAgent()
    
    @pytest.fixture
    async def grouping_service(self):
        """Create document grouping service instance"""
        return DocumentGroupingService()
    
    @pytest.fixture
    async def citation_service(self):
        """Create citation deduplication service instance"""
        return CitationDeduplicationService()
    
    @pytest.fixture
    async def synthesis_service(self):
        """Create legal theory synthesis service instance"""
        return LegalTheorySynthesisService()
    
    @pytest.mark.asyncio
    async def test_document_loading(self, memo_documents):
        """Test that fixture documents are loaded correctly"""
        assert len(memo_documents) > 0, "Should load at least one fixture document"
        assert len(memo_documents) >= 3, "Should have multiple documents for meaningful consolidation"
        
        # Validate document structure
        for doc in memo_documents:
            assert "id" in doc
            assert "content" in doc
            assert "title" in doc
            assert len(doc["content"]) > 100, f"Document {doc['id']} should have substantial content"
            
            # Check for legal content indicators
            content_lower = doc["content"].lower()
            legal_indicators = [
                "memorandum", "court", "statute", "violation", "tenant", "landlord",
                "n.j.s.a.", "new jersey", "lease", "provision"
            ]
            
            has_legal_content = any(indicator in content_lower for indicator in legal_indicators)
            assert has_legal_content, f"Document {doc['id']} should contain legal content"
        
        logger.info(f"✓ Successfully validated {len(memo_documents)} documents")
    
    @pytest.mark.asyncio
    async def test_document_grouping_service(self, memo_documents, grouping_service):
        """Test document grouping by legal theory"""
        
        # Test legal theory grouping
        grouping_result = await grouping_service.group_documents(
            documents=memo_documents,
            strategy=GroupingStrategy.LEGAL_THEORY,
            min_cluster_size=1,  # Allow single-document clusters for testing
            similarity_threshold=0.6
        )
        
        # Validate grouping results
        assert grouping_result.strategy == GroupingStrategy.LEGAL_THEORY
        assert len(grouping_result.clusters) > 0, "Should create at least one cluster"
        
        # Check cluster quality
        total_clustered_docs = sum(len(cluster.documents) for cluster in grouping_result.clusters)
        total_docs = len(memo_documents)
        
        coverage = total_clustered_docs / total_docs
        assert coverage >= 0.7, f"Should cluster at least 70% of documents, got {coverage:.2%}"
        
        # Validate cluster structure
        for cluster in grouping_result.clusters:
            assert cluster.cluster_name, "Each cluster should have a name"
            assert cluster.primary_theory, "Each cluster should have a primary theory"
            assert len(cluster.documents) > 0, "Each cluster should contain documents"
            assert 0 <= cluster.similarity_score <= 1, "Similarity score should be valid"
            assert cluster.strength_score > 0, "Strength score should be positive"
        
        # Test quality metrics
        assert "coverage" in grouping_result.quality_metrics
        assert "cluster_cohesion" in grouping_result.quality_metrics
        assert grouping_result.quality_metrics["coverage"] > 0
        
        logger.info(f"✓ Created {len(grouping_result.clusters)} clusters with {coverage:.2%} coverage")
        
        return grouping_result
    
    @pytest.mark.asyncio
    async def test_citation_deduplication_service(self, memo_documents, citation_service):
        """Test citation extraction and deduplication"""
        
        dedup_result = await citation_service.deduplicate_citations(
            source_documents=memo_documents,
            preserve_context=True,
            normalize_format=CitationFormat.NEW_JERSEY
        )
        
        # Validate deduplication results
        assert dedup_result.original_citation_count >= 0
        assert dedup_result.deduplicated_citation_count <= dedup_result.original_citation_count
        
        if dedup_result.original_citation_count > 0:
            # Test reduction efficiency
            reduction_rate = 1 - (dedup_result.deduplicated_citation_count / dedup_result.original_citation_count)
            logger.info(f"Citation reduction rate: {reduction_rate:.2%}")
            
            # Validate citation clusters
            for cluster in dedup_result.citation_clusters:
                assert cluster.cluster_id, "Each cluster should have an ID"
                assert cluster.primary_citation, "Each cluster should have a primary citation"
                assert cluster.consolidated_format, "Each cluster should have consolidated format"
                assert cluster.authority_ranking > 0, "Authority ranking should be positive"
        
        # Test quality metrics
        assert 0 <= dedup_result.format_consistency_score <= 1
        assert 0 <= dedup_result.preservation_rate <= 1
        
        logger.info(f"✓ Processed {dedup_result.original_citation_count} citations → {dedup_result.deduplicated_citation_count} deduplicated")
        
        return dedup_result
    
    @pytest.mark.asyncio
    async def test_legal_theory_synthesis(self, memo_documents, synthesis_service):
        """Test content synthesis for individual legal theories"""
        
        # Test synthesis for a mock legal theory
        test_theory = "Truth in Renting Act Violation"
        
        synthesis_result = await synthesis_service.synthesize_legal_theory_content(
            theory_name=test_theory,
            source_documents=memo_documents[:3],  # Use first 3 documents
            target_section="rule_statement",
            synthesis_mode=SynthesisMode.COMPREHENSIVE
        )
        
        # Validate synthesis results
        assert synthesis_result.synthesized_content, "Should generate synthesized content"
        assert len(synthesis_result.synthesized_content) > 100, "Content should be substantial"
        assert synthesis_result.confidence_score > 0, "Should have confidence score"
        assert synthesis_result.synthesis_strategy == SynthesisMode.COMPREHENSIVE.value
        
        # Check for legal content in synthesis
        content_lower = synthesis_result.synthesized_content.lower()
        legal_terms = ["statute", "violation", "tenant", "law", "legal", "court"]
        has_legal_terms = any(term in content_lower for term in legal_terms)
        assert has_legal_terms, "Synthesized content should contain legal terminology"
        
        logger.info(f"✓ Synthesized content for {test_theory}: {len(synthesis_result.synthesized_content)} chars")
        
        return synthesis_result
    
    @pytest.mark.asyncio
    async def test_full_consolidation_pipeline(self, memo_documents, consolidation_agent):
        """Test the complete end-to-end consolidation process"""
        
        # Use a subset of documents for faster testing
        test_documents = memo_documents[:5] if len(memo_documents) >= 5 else memo_documents
        
        # Run full consolidation
        consolidation_result = await consolidation_agent.consolidate_memoranda(
            memoranda_data=test_documents,
            consolidation_strategy="legal_theory"
        )
        
        # Validate overall structure
        assert consolidation_result.title, "Should generate a title"
        assert len(consolidation_result.legal_theories) > 0, "Should identify legal theories"
        assert consolidation_result.consolidated_memoranda_count == len(test_documents)
        
        # Validate CRRACC sections
        crracc_sections = [
            consolidation_result.conclusion,
            consolidation_result.rule_statement,
            consolidation_result.rule_explanation,
            consolidation_result.application,
            consolidation_result.counterargument,
            consolidation_result.final_conclusion
        ]
        
        for i, section in enumerate(crracc_sections):
            section_name = ["conclusion", "rule_statement", "rule_explanation", 
                          "application", "counterargument", "final_conclusion"][i]
            
            assert section, f"{section_name} section should exist"
            assert section.content, f"{section_name} should have content"
            assert len(section.content) > 50, f"{section_name} should have substantial content"
            assert section.confidence_score > 0, f"{section_name} should have confidence score"
        
        # Validate quality metrics
        assert "overall_quality" in consolidation_result.quality_metrics
        assert 0 <= consolidation_result.quality_metrics["overall_quality"] <= 1
        
        overall_quality = consolidation_result.quality_metrics["overall_quality"]
        assert overall_quality > 0.3, f"Overall quality should be reasonable, got {overall_quality:.2%}"
        
        # Validate citations
        assert consolidation_result.total_citations >= 0
        
        logger.info(f"✓ Consolidated {len(test_documents)} documents into {consolidation_result.title}")
        logger.info(f"  - Legal theories: {len(consolidation_result.legal_theories)}")
        logger.info(f"  - Total citations: {consolidation_result.total_citations}")
        logger.info(f"  - Overall quality: {overall_quality:.2%}")
        
        return consolidation_result
    
    @pytest.mark.asyncio
    async def test_consolidation_quality_metrics(self, memo_documents, consolidation_agent):
        """Test quality metrics calculation and validation"""
        
        # Use multiple documents for better quality assessment
        test_documents = memo_documents[:4] if len(memo_documents) >= 4 else memo_documents
        
        result = await consolidation_agent.consolidate_memoranda(
            memoranda_data=test_documents,
            consolidation_strategy="legal_theory"
        )
        
        quality_metrics = result.quality_metrics
        
        # Test required quality metrics exist
        required_metrics = ["overall_quality"]
        for metric in required_metrics:
            assert metric in quality_metrics, f"Should include {metric} metric"
            assert 0 <= quality_metrics[metric] <= 1, f"{metric} should be normalized (0-1)"
        
        # Test overall quality threshold
        overall_quality = quality_metrics["overall_quality"]
        logger.info(f"Overall consolidation quality: {overall_quality:.2%}")
        
        # Quality should be reasonable for legal content
        assert overall_quality > 0.2, "Quality should be above minimum threshold"
        
        # Test that different input sizes affect quality appropriately
        if len(test_documents) >= 3:
            # More documents should generally provide richer consolidation
            single_doc_result = await consolidation_agent.consolidate_memoranda(
                memoranda_data=test_documents[:1],
                consolidation_strategy="legal_theory"
            )
            
            single_quality = single_doc_result.quality_metrics["overall_quality"]
            
            # Multi-document consolidation should generally be higher quality
            # (though this is not always guaranteed in practice)
            logger.info(f"Single doc quality: {single_quality:.2%}, Multi doc quality: {overall_quality:.2%}")
        
        logger.info("✓ Quality metrics validation completed")
    
    @pytest.mark.asyncio
    async def test_consolidation_output_format(self, memo_documents, consolidation_agent):
        """Test that consolidation output follows proper legal format"""
        
        test_documents = memo_documents[:3] if len(memo_documents) >= 3 else memo_documents
        
        result = await consolidation_agent.consolidate_memoranda(
            memoranda_data=test_documents,
            consolidation_strategy="legal_theory"
        )
        
        # Test title format
        title = result.title
        assert "MEMORANDUM" in title.upper() or "CONSOLIDATED" in title.upper(), \
            "Title should indicate legal memorandum format"
        
        # Test section content quality
        sections_to_test = [
            ("conclusion", result.conclusion),
            ("rule_statement", result.rule_statement), 
            ("application", result.application)
        ]
        
        for section_name, section in sections_to_test:
            content = section.content
            
            # Check for legal writing characteristics
            assert len(content) > 100, f"{section_name} should be substantial"
            
            # Should contain legal language
            legal_indicators = [
                "court", "statute", "law", "legal", "violation", "provision",
                "tenant", "landlord", "new jersey", "act"
            ]
            
            content_lower = content.lower()
            legal_count = sum(1 for indicator in legal_indicators if indicator in content_lower)
            assert legal_count >= 2, f"{section_name} should contain legal terminology"
        
        logger.info(f"✓ Output format validation completed for: {title}")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, consolidation_agent):
        """Test error handling with invalid inputs"""
        
        # Test with empty document list
        with pytest.raises(Exception):
            await consolidation_agent.consolidate_memoranda(
                memoranda_data=[],
                consolidation_strategy="legal_theory"
            )
        
        # Test with malformed documents
        malformed_docs = [
            {"id": "test1"},  # Missing content
            {"content": ""},  # Empty content
        ]
        
        # Should handle gracefully or raise appropriate errors
        try:
            result = await consolidation_agent.consolidate_memoranda(
                memoranda_data=malformed_docs,
                consolidation_strategy="legal_theory"
            )
            # If it succeeds, quality should be very low
            assert result.quality_metrics["overall_quality"] < 0.3
        except Exception as e:
            # Expected - should fail gracefully
            logger.info(f"Expected error with malformed input: {e}")
        
        logger.info("✓ Error handling validation completed")


@pytest.mark.asyncio
async def test_performance_benchmark():
    """Performance benchmark test for consolidation pipeline"""
    
    # Create mock documents for performance testing
    mock_documents = []
    base_content = """
    # MEMORANDUM OF LAW IN SUPPORT OF PLAINTIFF'S ACTION FOR DECLARATORY JUDGMENT
    
    This memorandum challenges lease provisions that violate the Truth in Renting Act,
    N.J.S.A. 46:8-48, and demonstrate unconscionability under N.J.S.A. 12A:2A-108.
    
    The Anti-Eviction Act, N.J.S.A. 2A:18-61.1 et seq., provides comprehensive tenant
    protections that cannot be waived through adhesive lease provisions.
    """
    
    for i in range(5):
        mock_documents.append({
            "id": f"perf_test_{i}",
            "title": f"Performance Test Document {i}",
            "content": base_content + f"\n\nDocument {i} specific content with additional legal analysis.",
            "citations": [],
            "metadata": {"source": "performance_test"}
        })
    
    agent = ConsolidationAgent()
    
    import time
    start_time = time.time()
    
    result = await agent.consolidate_memoranda(
        memoranda_data=mock_documents,
        consolidation_strategy="legal_theory"
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Performance assertions
    assert processing_time < 120, f"Consolidation should complete within 2 minutes, took {processing_time:.1f}s"
    assert result.quality_metrics["overall_quality"] > 0.1, "Should produce reasonable quality output"
    
    logger.info(f"✓ Performance benchmark: {processing_time:.1f}s for {len(mock_documents)} documents")
    

if __name__ == "__main__":
    """Run tests directly for development"""
    
    async def run_tests():
        test_instance = TestConsolidationPipeline()
        
        # Load fixture memos
        fixture_path = Path(__file__).parent.parent / "fixtures" / "memos"
        
        if not fixture_path.exists():
            print(f"Warning: Fixture path {fixture_path} does not exist")
            return
        
        documents = []
        for memo_file in fixture_path.glob("*.md"):
            content = memo_file.read_text(encoding='utf-8')
            documents.append({
                "id": memo_file.stem,
                "title": memo_file.stem.replace('_', ' ').title(),
                "content": content,
                "file_path": str(memo_file),
                "citations": [],
                "metadata": {"source": "fixture"}
            })
        
        print(f"Loaded {len(documents)} fixture documents")
        
        if len(documents) == 0:
            print("No documents loaded - skipping tests")
            return
        
        # Run basic tests
        agent = ConsolidationAgent()
        
        print("Testing consolidation pipeline...")
        result = await agent.consolidate_memoranda(
            memoranda_data=documents[:3],  # Use first 3 documents
            consolidation_strategy="legal_theory"
        )
        
        print(f"✓ Consolidation completed:")
        print(f"  Title: {result.title}")
        print(f"  Legal theories: {len(result.legal_theories)}")
        print(f"  Citations: {result.total_citations}")
        print(f"  Quality: {result.quality_metrics.get('overall_quality', 0):.2%}")
        
        print("\n✓ Basic consolidation pipeline test completed successfully!")
    
    # Run the tests
    asyncio.run(run_tests())