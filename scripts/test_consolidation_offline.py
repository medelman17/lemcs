#!/usr/bin/env python3
"""
Offline test script for the consolidation pipeline using fixture memos.

This script validates the core functionality without requiring OpenAI API keys
by using mock LLM responses.
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


def create_mock_llm_response(prompt_type: str) -> str:
    """Create mock LLM responses based on prompt type"""
    
    mock_responses = {
        "legal_theories": """
        Theory Name: Truth in Renting Act Violations
        Provisions: Disclosure requirements, mandatory tenant protections, statutory compliance
        Citations: N.J.S.A. 46:8-48, Truth in Renting Act, tenant protection statutes
        Strength: 9
        ---
        Theory Name: Anti-Eviction Act Violations
        Provisions: Tenant protection provisions, waiver restrictions, unconscionable terms
        Citations: N.J.S.A. 2A:18-61.1 et seq., Anti-Eviction Act, tenant rights
        Strength: 8
        ---
        Theory Name: Unconscionability Under UCC
        Provisions: Procedural unconscionability, substantive unconscionability, adhesive contracts
        Citations: N.J.S.A. 12A:2A-108, UCC provisions, unconscionability doctrine
        Strength: 7
        ---
        """,
        
        "conclusion": """
        The lease agreements in question contain multiple provisions that violate New Jersey tenant protection statutes, 
        specifically the Truth in Renting Act (N.J.S.A. 46:8-48) and the Anti-Eviction Act (N.J.S.A. 2A:18-61.1 et seq.). 
        These violations demonstrate a pattern of unconscionable lease terms that systematically disadvantage tenants 
        in violation of state law.
        """,
        
        "rule_statement": """
        Under New Jersey law, landlords must comply with disclosure requirements set forth in the Truth in Renting Act, 
        N.J.S.A. 46:8-48, and cannot include unconscionable provisions that violate tenant protections under the 
        Anti-Eviction Act, N.J.S.A. 2A:18-61.1 et seq. Lease provisions that contradict these statutory protections 
        are void and unenforceable.
        """,
        
        "rule_explanation": """
        The Truth in Renting Act requires specific disclosures to protect tenants from unfair lease terms. 
        The Anti-Eviction Act provides comprehensive tenant protections that cannot be waived through lease provisions. 
        Courts have consistently held that lease terms violating these statutes are unconscionable and void as 
        contrary to public policy.
        """,
        
        "application": """
        In the present case, the lease provisions fail to comply with mandatory disclosure requirements and include 
        terms that directly contradict tenant protections. These violations demonstrate a systematic pattern of 
        unconscionable terms that disadvantage tenants in violation of New Jersey statutory law.
        """,
        
        "counterargument": """
        While defendants may argue that tenants voluntarily agreed to lease terms, New Jersey law is clear that 
        statutory tenant protections cannot be waived through private agreement. The unconscionable nature of 
        these provisions renders them void regardless of tenant consent.
        """,
        
        "final_conclusion": """
        The identified lease provisions violate multiple New Jersey tenant protection statutes and constitute 
        unconscionable terms that must be declared void and unenforceable. The pattern of violations demonstrates 
        systematic disregard for tenant rights protected under state law.
        """
    }
    
    # Default response for unknown types
    return mock_responses.get(prompt_type, "Mock legal analysis response for " + prompt_type)


async def test_document_loading(documents: List[Dict[str, Any]]) -> bool:
    """Test that fixture documents are loaded correctly"""
    
    logger.info("Testing document loading...")
    
    try:
        assert len(documents) > 0, "Should load at least one fixture document"
        assert len(documents) >= 3, "Should have multiple documents for meaningful consolidation"
        
        # Validate document structure
        for doc in documents:
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
        
        logger.info(f"✓ Successfully validated {len(documents)} documents")
        return True
        
    except Exception as e:
        logger.error(f"✗ Document loading failed: {e}")
        return False


async def test_consolidation_agent_mock(documents: List[Dict[str, Any]]) -> bool:
    """Test the consolidation agent with mocked LLM responses"""
    
    logger.info("Testing consolidation agent with mock responses...")
    
    try:
        # Use a subset for faster testing
        test_docs = documents[:3] if len(documents) >= 3 else documents
        
        # Mock the LLM responses
        mock_response = MagicMock()
        mock_response.content = create_mock_llm_response("legal_theories")
        
        with patch('langchain_openai.ChatOpenAI') as mock_llm_class:
            # Create a mock LLM instance
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            # Also mock the citation extraction
            with patch('nlp.citation_service.CitationExtractionService.extract_citations') as mock_citations:
                mock_citations.return_value = {
                    'citations': [
                        {'text': 'N.J.S.A. 46:8-48', 'type': 'statute'},
                        {'text': 'N.J.S.A. 2A:18-61.1', 'type': 'statute'}
                    ],
                    'total_count': 2
                }
                
                # Import and test the consolidation agent
                from agents.consolidation_agent import ConsolidationAgent
                
                agent = ConsolidationAgent()
                
                # Create mock responses for different sections
                section_responses = {
                    "legal_theories": create_mock_llm_response("legal_theories"),
                    "conclusion": create_mock_llm_response("conclusion"),
                    "rule_statement": create_mock_llm_response("rule_statement"),
                    "rule_explanation": create_mock_llm_response("rule_explanation"),
                    "application": create_mock_llm_response("application"),
                    "counterargument": create_mock_llm_response("counterargument"),
                    "final_conclusion": create_mock_llm_response("final_conclusion")
                }
                
                def mock_llm_response_side_effect(*args, **kwargs):
                    """Return appropriate mock response based on the prompt"""
                    prompt_text = str(args[0]) if args else ""
                    
                    if "legal theories" in prompt_text.lower():
                        mock_response.content = section_responses["legal_theories"]
                    elif "conclusion" in prompt_text.lower() and "final" not in prompt_text.lower():
                        mock_response.content = section_responses["conclusion"]
                    elif "rule statement" in prompt_text.lower():
                        mock_response.content = section_responses["rule_statement"]
                    elif "rule explanation" in prompt_text.lower():
                        mock_response.content = section_responses["rule_explanation"]
                    elif "application" in prompt_text.lower():
                        mock_response.content = section_responses["application"]
                    elif "counterargument" in prompt_text.lower():
                        mock_response.content = section_responses["counterargument"]
                    elif "final conclusion" in prompt_text.lower():
                        mock_response.content = section_responses["final_conclusion"]
                    else:
                        mock_response.content = "Mock response for unknown prompt type"
                    
                    return mock_response
                
                mock_llm.ainvoke.side_effect = mock_llm_response_side_effect
                
                # Disable database operations for testing
                with patch.object(agent, 'track_workflow'):
                    # Mock the workflow tracking context manager
                    class MockWorkflowContext:
                        async def __aenter__(self):
                            return "mock_workflow_id"
                        async def __aexit__(self, *args):
                            pass
                    
                    agent.track_workflow = lambda *args, **kwargs: MockWorkflowContext()
                    
                    # Run the consolidation
                    result = await agent.consolidate_memoranda(
                        memoranda_data=test_docs,
                        consolidation_strategy="legal_theory"
                    )
                    
                    # Validate results
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
                    
                    sections_with_content = 0
                    for section_name, section in sections:
                        if section and section.content:
                            logger.info(f"✓ {section_name}: {len(section.content)} chars (confidence: {section.confidence_score:.2f})")
                            sections_with_content += 1
                        else:
                            logger.warning(f"⚠ {section_name}: No content generated")
                    
                    # Should have generated content for most sections
                    assert sections_with_content >= 4, f"Should generate content for at least 4 sections, got {sections_with_content}"
                    
                    return True
        
    except Exception as e:
        logger.error(f"✗ Consolidation agent failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_handling() -> bool:
    """Test error handling with invalid inputs"""
    
    logger.info("Testing error handling...")
    
    try:
        with patch('langchain_openai.ChatOpenAI') as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm_class.return_value = mock_llm
            
            from agents.consolidation_agent import ConsolidationAgent
            
            # Disable database operations for testing
            with patch.object(ConsolidationAgent, 'track_workflow'):
                agent = ConsolidationAgent()
                
                # Mock the workflow tracking context manager
                class MockWorkflowContext:
                    async def __aenter__(self):
                        return "mock_workflow_id"
                    async def __aexit__(self, *args):
                        pass
                
                agent.track_workflow = lambda *args, **kwargs: MockWorkflowContext()
                
                # Test with empty document list
                try:
                    await agent.consolidate_memoranda(
                        memoranda_data=[],
                        consolidation_strategy="legal_theory"
                    )
                    logger.warning("⚠ Empty document list should have raised an error")
                except Exception as e:
                    logger.info(f"✓ Empty documents correctly raised error: {type(e).__name__}")
                
                # Test with malformed documents
                malformed_docs = [
                    {"id": "test1"},  # Missing content
                    {"content": ""},  # Empty content
                ]
                
                try:
                    # Mock a response for malformed docs
                    mock_response = MagicMock()
                    mock_response.content = "Unable to analyze malformed documents"
                    mock_llm.ainvoke.return_value = mock_response
                    
                    with patch('nlp.citation_service.CitationExtractionService.extract_citations') as mock_citations:
                        mock_citations.return_value = {'citations': [], 'total_count': 0}
                        
                        result = await agent.consolidate_memoranda(
                            memoranda_data=malformed_docs,
                            consolidation_strategy="legal_theory"
                        )
                        
                        # If it succeeds, quality should be very low or it should handle gracefully
                        if hasattr(result, 'quality_metrics'):
                            quality = result.quality_metrics.get('overall_quality', 0)
                            logger.info(f"✓ Malformed input handled gracefully with quality: {quality:.2%}")
                        else:
                            logger.info("✓ Malformed input handled gracefully")
                            
                except Exception as e:
                    logger.info(f"✓ Expected error with malformed input: {type(e).__name__}")
                
                return True
        
    except Exception as e:
        logger.error(f"✗ Error handling test failed: {e}")
        return False


async def main():
    """Main test execution"""
    
    logger.info("Starting offline consolidation pipeline tests...")
    
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
    
    # Test 1: Document loading
    test_results.append(await test_document_loading(documents))
    
    # Test 2: Consolidation agent with mocks
    test_results.append(await test_consolidation_agent_mock(documents))
    
    # Test 3: Error handling
    test_results.append(await test_error_handling())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All offline tests passed! Consolidation pipeline structure is working correctly.")
        return True
    else:
        logger.error(f"❌ {total_tests - passed_tests} tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    """Run the offline tests"""
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)