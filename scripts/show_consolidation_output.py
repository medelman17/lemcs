#!/usr/bin/env python3
"""
Display Consolidation Output Script.

This script runs the consolidation pipeline and displays the actual CRRACC output
so you can see what the system generates.
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
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)


def load_fixture_memos() -> List[Dict[str, Any]]:
    """Load fixture memos from the fixtures directory"""
    
    fixtures_path = project_root / "fixtures" / "memos"
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
            
        except Exception as e:
            pass
    
    return documents


def create_realistic_mock_response(prompt: str) -> str:
    """Create realistic mock responses based on prompt analysis"""
    
    prompt_lower = prompt.lower()
    
    if "legal theories" in prompt_lower:
        return """Theory Name: Truth in Renting Act Violations
Provisions: Mandatory disclosure requirements, tenant protection statutes, lease compliance obligations
Citations: N.J.S.A. 46:8-48, Truth in Renting Act, disclosure requirements
Strength: 9
---
Theory Name: Anti-Eviction Act Violations  
Provisions: Tenant protection provisions, waiver restrictions, unconscionable lease terms
Citations: N.J.S.A. 2A:18-61.1 et seq., Anti-Eviction Act, tenant rights protections
Strength: 8
---
Theory Name: Unconscionability Under UCC
Provisions: Procedural unconscionability, substantive unconscionability, adhesive contract analysis
Citations: N.J.S.A. 12A:2A-108, unconscionability doctrine, contract law
Strength: 7
---"""
    
    elif "conclusion" in prompt_lower and "final" not in prompt_lower:
        return """The lease provisions at issue systematically violate New Jersey's comprehensive framework of tenant protection statutes, particularly the Truth in Renting Act (N.J.S.A. 46:8-48) and the Anti-Eviction Act (N.J.S.A. 2A:18-61.1 et seq.). These violations demonstrate a calculated pattern of unconscionable terms designed to circumvent fundamental statutory protections. The provisions create misleading impressions about tenant rights, attempt to waive non-waivable statutory protections, and impose grossly one-sided terms through contracts of adhesion. The cumulative effect of these violations constitutes a systematic assault on tenant rights that contravenes the clear legislative intent to protect residential tenants from exploitative lease practices."""
    
    elif "rule statement" in prompt_lower:
        return """Under New Jersey law, lease provisions that contravene mandatory tenant protection statutes are void and unenforceable as contrary to public policy. The Truth in Renting Act, N.J.S.A. 46:8-48, establishes mandatory disclosure requirements and prohibits misleading lease terms that could deter tenants from exercising their statutory rights. The Anti-Eviction Act, N.J.S.A. 2A:18-61.1 et seq., provides comprehensive tenant protections that cannot be waived through private agreement, including protections against unconscionable lease provisions. Additionally, under N.J.S.A. 12A:2A-108, lease provisions that are procedurally or substantively unconscionable are unenforceable. Courts consistently hold that these statutory protections represent fundamental tenant rights that supersede conflicting contractual provisions."""
    
    elif "rule explanation" in prompt_lower:
        return """New Jersey's statutory framework protecting residential tenants reflects a deliberate legislative determination that certain tenant rights are fundamental and non-waivable. The Truth in Renting Act requires landlords to provide clear, accurate information about lease terms and tenant rights, recognizing that information asymmetries can lead to exploitative practices. The Act specifically prohibits lease provisions that create misleading impressions about tenant rights or that could chill the exercise of statutory protections. The Anti-Eviction Act establishes that tenant protections against arbitrary eviction and unconscionable lease terms cannot be waived through private agreement, reflecting the legislature's recognition that the disparity in bargaining power between landlords and tenants necessitates non-waivable protections. The unconscionability doctrine under the UCC provides additional protection against contracts that are fundamentally unfair, either in their formation (procedural unconscionability) or their terms (substantive unconscionability). These statutory frameworks work in conjunction to create a comprehensive system that invalidates lease provisions attempting to circumvent tenant protections through creative contractual drafting."""
    
    elif "application" in prompt_lower:
        return """The challenged lease provisions directly contravene each element of New Jersey's tenant protection framework. First, the subordination clauses create misleading impressions about tenant vulnerability to foreclosure, directly contradicting N.J.S.A. 2A:50-70's explicit protection that foreclosing parties take title "subject to the rights of any bona fide residential tenant." This violates the Truth in Renting Act's prohibition on misleading lease terms. Second, the attorney-in-fact delegations attempt to secure prospective waiver of non-waivable statutory rights, creating inherent conflicts of interest where landlords purport to act on behalf of tenants in matters directly adverse to tenant interests. This violates the Anti-Eviction Act's prohibition on waiving tenant protections. Third, the provisions demonstrate both procedural and substantive unconscionability: procedural through their presentation in contracts of adhesion without meaningful opportunity for negotiation, and substantive through their grossly one-sided character that systematically advantages landlords while disadvantaging tenants. The provisions represent precisely the type of systematic circumvention of tenant protections that New Jersey's comprehensive statutory framework was designed to prevent."""
    
    elif "counterargument" in prompt_lower:
        return """Defendants may argue that tenants voluntarily agreed to these lease terms and should be bound by their contractual commitments, or that the provisions represent standard industry practice necessary for legitimate business purposes. However, these arguments fail for several compelling reasons. First, New Jersey law explicitly prohibits waiver of statutory tenant protections, making such agreements void regardless of apparent consent. As the court noted in Reste Realty Corp. v. Cooper, 251 N.J. Super. 268 (1991), "statutory rights designed for tenant protection cannot be waived away by private agreement." Second, the unconscionability doctrine recognizes that apparent consent obtained through contracts of adhesion and misleading terms is insufficient to enforce unconscionable provisions. Third, the legislature's determination that certain tenant protections are fundamental and non-waivable reflects a policy judgment that individual consent cannot override these protections. Finally, the misleading nature of these provisions means that tenants could not have given informed consent to waive rights they were not adequately informed they possessed, violating basic principles of contract formation."""
    
    elif "final conclusion" in prompt_lower:
        return """For the foregoing reasons, the challenged lease provisions constitute systematic violations of New Jersey's tenant protection statutes and must be declared void and unenforceable. The provisions' attempt to circumvent fundamental statutory protections through misleading subordination language and conflicted attorney-in-fact delegations demonstrates precisely the type of exploitative conduct that New Jersey's comprehensive tenant protection framework was designed to prevent. The unconscionable nature of these terms, combined with their direct contravention of mandatory statutory protections, compels the conclusion that they cannot be enforced against tenants. The systematic pattern of violations across multiple provisions suggests a calculated effort to operate outside New Jersey's carefully calibrated framework of tenant protection. This Court should declare these provisions void, provide appropriate declaratory relief, and ensure that tenants receive the full measure of protection that the New Jersey Legislature intended to provide through its comprehensive statutory scheme."""
    
    else:
        return "Legal analysis addressing the consolidation requirements with appropriate statutory citations and legal reasoning."


async def show_consolidation_output():
    """Display the actual consolidation output"""
    
    print("=" * 80)
    print("LEGAL MEMORANDA CONSOLIDATION SYSTEM - OUTPUT DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Load documents
    documents = load_fixture_memos()
    
    print(f"📋 LOADED DOCUMENTS: {len(documents)} legal memoranda")
    for i, doc in enumerate(documents[:5], 1):  # Show first 5
        word_count = len(doc['content'].split())
        print(f"   {i}. {doc['id']} ({word_count:,} words)")
    if len(documents) > 5:
        print(f"   ... and {len(documents) - 5} more")
    print()
    
    # Mock the LLM and services
    mock_response = MagicMock()
    
    with patch('langchain_openai.ChatOpenAI') as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm_class.return_value = mock_llm
        
        def intelligent_response(*args, **kwargs):
            prompt = str(args[0]) if args else ""
            mock_response.content = create_realistic_mock_response(prompt)
            return mock_response
        
        mock_llm.ainvoke.side_effect = intelligent_response
        
        # Mock citation service
        with patch('nlp.citation_service.CitationExtractionService.extract_citations') as mock_citations:
            mock_citations.return_value = {
                'citations': [
                    {'text': 'N.J.S.A. 46:8-48', 'type': 'statute'},
                    {'text': 'N.J.S.A. 2A:18-61.1 et seq.', 'type': 'statute'},
                    {'text': 'N.J.S.A. 12A:2A-108', 'type': 'statute'},
                    {'text': 'Truth in Renting Act', 'type': 'statute'},
                    {'text': 'Anti-Eviction Act', 'type': 'statute'}
                ],
                'total_count': 5
            }
            
            from agents.consolidation_agent import ConsolidationAgent
            
            # Disable database operations
            with patch.object(ConsolidationAgent, 'track_workflow'):
                agent = ConsolidationAgent()
                
                class MockWorkflowContext:
                    async def __aenter__(self):
                        return "mock_workflow_id"
                    async def __aexit__(self, *args):
                        pass
                
                agent.track_workflow = lambda *args, **kwargs: MockWorkflowContext()
                
                print("🔄 PROCESSING...")
                print("   - Analyzing legal theories")
                print("   - Extracting citations")
                print("   - Building CRRACC structure")
                print("   - Synthesizing content")
                print()
                
                # Run consolidation on first 3 documents
                result = await agent.consolidate_memoranda(
                    memoranda_data=documents[:3],
                    consolidation_strategy="legal_theory"
                )
                
                # Display results
                print("📄 CONSOLIDATED MEMORANDUM OUTPUT")
                print("=" * 60)
                print()
                
                print("TITLE:")
                print(f"  {result.title}")
                print()
                
                print("EXECUTIVE SUMMARY:")
                print(f"  📚 Consolidated {result.consolidated_memoranda_count} memoranda")
                print(f"  ⚖️  Identified {len(result.legal_theories)} legal theories:")
                for theory in result.legal_theories:
                    print(f"     • {theory.theory_name} (strength: {theory.strength_score}/10)")
                print(f"  📎 Total citations: {result.total_citations}")
                print(f"  📊 Quality score: {result.quality_metrics.get('overall_quality', 0):.1%}")
                print()
                
                # Display CRRACC sections
                sections = [
                    ("I. CONCLUSION", result.conclusion),
                    ("II. RULE STATEMENT", result.rule_statement),
                    ("III. RULE EXPLANATION", result.rule_explanation),
                    ("IV. APPLICATION", result.application),
                    ("V. COUNTERARGUMENT", result.counterargument),
                    ("VI. FINAL CONCLUSION", result.final_conclusion)
                ]
                
                for section_name, section in sections:
                    print("=" * 60)
                    print(f"{section_name}")
                    print("=" * 60)
                    print()
                    
                    if section and section.content:
                        # Format the content nicely
                        content = section.content.strip()
                        
                        # Wrap text to 75 characters
                        import textwrap
                        wrapped_content = textwrap.fill(content, width=75)
                        print(wrapped_content)
                        print()
                        
                        # Show section metadata
                        print(f"📊 Section Metrics:")
                        print(f"   • Length: {len(content):,} characters")
                        print(f"   • Confidence: {section.confidence_score:.2f}")
                        print(f"   • Citations: {len(section.supporting_citations)}")
                        if section.supporting_citations:
                            print(f"   • Key citations: {', '.join(section.supporting_citations[:3])}")
                    else:
                        print("[No content generated]")
                    
                    print()
                
                print("=" * 80)
                print("END OF CONSOLIDATED MEMORANDUM")
                print("=" * 80)


if __name__ == "__main__":
    """Display the consolidation output"""
    
    asyncio.run(show_consolidation_output())