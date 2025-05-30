#!/usr/bin/env python3
"""
Basic test script for the consolidation pipeline using fixture memos.

This script validates the core functionality without requiring pytest.
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.consolidation_agent import ConsolidationAgent
from nlp.document_grouping_service import DocumentGroupingService, GroupingStrategy
from nlp.citation_deduplication_service import CitationDeduplicationService, CitationFormat

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_fixture_memos() -> List[Dict[str, Any]]:
    """Load fixture memos from the fixtures directory"""
    
    fixtures_path = project_root / "fixtures" / "memos"
    
    if not fixtures_path.exists():
        logger.error(f"Fixtures directory not found: {fixtures_path}")
        return []
    
    documents = []
    
    for memo_file in fixtures_path.glob("*.md"):
        try:
            content = memo_file.read_text(encoding='utf-8')
            
            document = {
                "id": memo_file.stem,
                "title": memo_file.stem.replace('_', ' ').title(),
                "content": content,
                "file_path": str(memo_file),
                "citations": [],
                "metadata": {
                    "file_size": len(content),
                    "word_count": len(content.split()),
                    "source": "fixture"
                }
            }
            documents.append(document)
            logger.info(f"Loaded memo: {memo_file.stem} ({len(content)} chars)")
            
        except Exception as e:
            logger.warning(f"Failed to load {memo_file}: {e}")
    
    logger.info(f"Successfully loaded {len(documents)} memo documents")
    return documents


async def test_document_grouping(documents: List[Dict[str, Any]]) -> bool:
    """Test document grouping functionality"""
    
    logger.info("Testing document grouping service...")
    
    try:
        grouping_service = DocumentGroupingService()
        
        result = await grouping_service.group_documents(
            documents=documents,
            strategy=GroupingStrategy.LEGAL_THEORY,
            min_cluster_size=1,
            similarity_threshold=0.6
        )
        
        logger.info(f"✓ Created {len(result.clusters)} clusters")
        logger.info(f"✓ Coverage: {result.quality_metrics.get('coverage', 0):.2%}")
        
        for i, cluster in enumerate(result.clusters):
            logger.info(f"  Cluster {i+1}: {cluster.cluster_name} ({len(cluster.documents)} docs)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Document grouping failed: {e}")
        return False


async def test_citation_deduplication(documents: List[Dict[str, Any]]) -> bool:
    """Test citation deduplication functionality"""
    
    logger.info("Testing citation deduplication service...")
    
    try:
        citation_service = CitationDeduplicationService()
        
        result = await citation_service.deduplicate_citations(
            source_documents=documents,
            preserve_context=True,
            normalize_format=CitationFormat.NEW_JERSEY
        )
        
        logger.info(f"✓ Citations: {result.original_citation_count} → {result.deduplicated_citation_count}")
        logger.info(f"✓ Format consistency: {result.format_consistency_score:.2%}")
        logger.info(f"✓ Preservation rate: {result.preservation_rate:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Citation deduplication failed: {e}")
        return False


async def test_consolidation_agent(documents: List[Dict[str, Any]]) -> bool:
    """Test the main consolidation agent"""
    
    logger.info("Testing consolidation agent...")
    
    try:
        # Use a subset for faster testing
        test_docs = documents[:3] if len(documents) >= 3 else documents
        
        agent = ConsolidationAgent()
        
        result = await agent.consolidate_memoranda(
            memoranda_data=test_docs,
            consolidation_strategy="legal_theory"
        )
        
        logger.info(f"✓ Title: {result.title}")
        logger.info(f"✓ Legal theories: {len(result.legal_theories)}")
        logger.info(f"✓ Citations: {result.total_citations}")
        logger.info(f"✓ Quality: {result.quality_metrics.get('overall_quality', 0):.2%}")
        
        # Validate CRRACC sections
        sections = [
            ("Conclusion", result.conclusion),
            ("Rule Statement", result.rule_statement),
            ("Rule Explanation", result.rule_explanation),
            ("Application", result.application),
            ("Counterargument", result.counterargument),
            ("Final Conclusion", result.final_conclusion)
        ]
        
        for section_name, section in sections:
            if section and section.content:
                logger.info(f"✓ {section_name}: {len(section.content)} chars (confidence: {section.confidence_score:.2f})")
            else:
                logger.warning(f"⚠ {section_name}: No content generated")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Consolidation agent failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_end_to_end_pipeline(documents: List[Dict[str, Any]]) -> bool:
    """Test the complete end-to-end pipeline"""
    
    logger.info("Testing complete end-to-end pipeline...")
    
    try:
        # Test with different document subsets
        test_cases = [
            ("Small subset", documents[:2] if len(documents) >= 2 else documents),
            ("Medium subset", documents[:4] if len(documents) >= 4 else documents),
        ]
        
        for test_name, test_docs in test_cases:
            logger.info(f"Testing {test_name} ({len(test_docs)} documents)...")
            
            agent = ConsolidationAgent()
            result = await agent.consolidate_memoranda(
                memoranda_data=test_docs,
                consolidation_strategy="legal_theory"
            )
            
            quality = result.quality_metrics.get('overall_quality', 0)
            logger.info(f"✓ {test_name}: Quality {quality:.2%}, Theories {len(result.legal_theories)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ End-to-end pipeline failed: {e}")
        return False


async def main():
    """Main test execution"""
    
    logger.info("Starting consolidation pipeline tests...")
    
    # Load fixture documents
    documents = load_fixture_memos()
    
    if not documents:
        logger.error("No documents loaded - cannot run tests")
        return False
    
    if len(documents) < 2:
        logger.error("Need at least 2 documents for meaningful consolidation tests")
        return False
    
    # Run tests
    test_results = []
    
    # Test 1: Document grouping
    test_results.append(await test_document_grouping(documents))
    
    # Test 2: Citation deduplication
    test_results.append(await test_citation_deduplication(documents))
    
    # Test 3: Consolidation agent
    test_results.append(await test_consolidation_agent(documents))
    
    # Test 4: End-to-end pipeline
    test_results.append(await test_end_to_end_pipeline(documents))
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Consolidation pipeline is working correctly.")
        return True
    else:
        logger.error(f"❌ {total_tests - passed_tests} tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    """Run the tests"""
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)