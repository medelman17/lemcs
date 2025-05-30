---
description: LangGraph multi-agent system guidelines for legal document processing workflows
globs: ["agents/**/*.py", "**/*agent*.py", "**/*orchestrat*.py"]
alwaysApply: false
---

# LangGraph Agent Development Guidelines

## Agent Architecture Overview

The LeMCS system uses LangGraph for orchestrating specialized legal agents in document processing workflows.

### Core Agent Components
1. **Orchestrator** (@agents/orchestrator.py) - Workflow coordination
2. **Legal Analyzer** (@agents/legal_analyzer.py) - Document analysis
3. **Citation Extractor** (@agents/citation_extractor.py) - Citation processing
4. **Synthesis Agent** (@agents/synthesis_agent.py) - Document consolidation

## LangGraph Patterns

### State Graph Definition
```python
from langgraph import StateGraph
from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    """Shared state across all agents."""
    documents: List[str]
    analyses: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    consolidated_output: str
    errors: List[str]
    current_step: str

def create_workflow() -> StateGraph:
    """Create the legal document processing workflow."""
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("analyze", legal_analysis_agent)
    workflow.add_node("extract_citations", citation_extraction_agent)
    workflow.add_node("synthesize", synthesis_agent)
    
    # Define workflow edges
    workflow.add_edge("analyze", "extract_citations")
    workflow.add_edge("extract_citations", "synthesize")
    
    return workflow
```

### Agent Implementation Pattern
```python
from nlp.hybrid_legal_nlp import HybridLegalNLP

async def legal_analysis_agent(state: AgentState) -> AgentState:
    """Analyze legal documents using hybrid NLP."""
    nlp = HybridLegalNLP()
    analyses = []
    
    for document in state["documents"]:
        try:
            analysis = nlp.comprehensive_analysis(document)
            analyses.append(analysis)
        except Exception as e:
            state["errors"].append(f"Analysis failed: {e}")
    
    state["analyses"] = analyses
    state["current_step"] = "analysis_complete"
    return state
```

## Agent Specialization

### Legal Analyzer Agent
```python
class LegalAnalyzer:
    """Agent specialized in legal document analysis."""
    
    def __init__(self):
        self.nlp = HybridLegalNLP()
    
    async def analyze_document(self, document: str) -> Dict[str, Any]:
        """Perform comprehensive legal document analysis."""
        analysis = self.nlp.comprehensive_analysis(document)
        
        return {
            "document_type": analysis["document_analysis"]["type"],
            "entities": analysis["entities"]["combined_unique"],
            "legal_concepts": analysis["entities"]["legal_concepts"],
            "complexity_score": analysis["document_analysis"]["complexity_score"],
            "confidence": analysis["document_analysis"]["confidence"]
        }
    
    async def classify_legal_domain(self, analysis: Dict[str, Any]) -> str:
        """Determine the legal domain of the document."""
        doc_type = analysis["document_type"]
        entities = analysis["entities"]
        
        if doc_type == "lease_agreement":
            return "real_estate"
        elif doc_type == "legal_complaint":
            return "litigation"
        elif any("contract" in str(entity).lower() for entity in entities):
            return "commercial"
        else:
            return "general"
```

### Citation Extractor Agent
```python
from nlp.citation_service import CitationService

class CitationExtractor:
    """Agent specialized in legal citation extraction and validation."""
    
    def __init__(self):
        self.citation_service = CitationService()
    
    async def extract_citations(self, document: str) -> List[Dict[str, Any]]:
        """Extract and validate legal citations."""
        citations = self.citation_service.extract_citations(document)
        
        validated_citations = []
        for citation in citations:
            validation = await self.validate_citation(citation)
            validated_citations.append({
                **citation,
                "validation": validation
            })
        
        return validated_citations
    
    async def validate_citation(self, citation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate citation format and authority."""
        # Implement citation validation logic
        return {
            "is_valid": True,
            "authority_level": "primary",
            "jurisdiction": "federal"
        }
```

### Synthesis Agent
```python
class SynthesisAgent:
    """Agent specialized in consolidating multiple legal documents."""
    
    def __init__(self):
        self.nlp = HybridLegalNLP()
    
    async def consolidate_documents(
        self, 
        analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Consolidate multiple legal document analyses."""
        
        # Group by document type
        grouped_docs = self.group_by_type(analyses)
        
        # Extract common themes
        common_themes = await self.extract_common_themes(analyses)
        
        # Generate consolidated narrative
        narrative = await self.generate_narrative(grouped_docs, common_themes)
        
        return {
            "consolidated_narrative": narrative,
            "document_groups": grouped_docs,
            "common_themes": common_themes,
            "summary_statistics": self.calculate_statistics(analyses)
        }
    
    def group_by_type(self, analyses: List[Dict[str, Any]]) -> Dict[str, List]:
        """Group documents by type for structured consolidation."""
        groups = {}
        for analysis in analyses:
            doc_type = analysis["document_type"]
            if doc_type not in groups:
                groups[doc_type] = []
            groups[doc_type].append(analysis)
        return groups
    
    async def extract_common_themes(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Extract common legal themes across documents."""
        all_concepts = []
        for analysis in analyses:
            concepts = analysis.get("legal_concepts", {})
            for concept_list in concepts.values():
                all_concepts.extend(concept_list)
        
        # Find most common themes
        from collections import Counter
        common = Counter(all_concepts).most_common(10)
        return [theme for theme, count in common if count > 1]
```

## Workflow Orchestration

### Orchestrator Implementation
```python
class LegalWorkflowOrchestrator:
    """Orchestrates the complete legal document processing workflow."""
    
    def __init__(self):
        self.legal_analyzer = LegalAnalyzer()
        self.citation_extractor = CitationExtractor()
        self.synthesis_agent = SynthesisAgent()
    
    async def process_documents(
        self, 
        documents: List[str]
    ) -> Dict[str, Any]:
        """Process multiple legal documents through the complete workflow."""
        
        state = AgentState(
            documents=documents,
            analyses=[],
            citations=[],
            consolidated_output="",
            errors=[],
            current_step="starting"
        )
        
        # Execute workflow steps
        state = await self.analyze_step(state)
        state = await self.extract_citations_step(state)
        state = await self.synthesize_step(state)
        
        return {
            "status": "completed",
            "analyses": state["analyses"],
            "citations": state["citations"],
            "consolidated_output": state["consolidated_output"],
            "errors": state["errors"]
        }
    
    async def analyze_step(self, state: AgentState) -> AgentState:
        """Execute legal analysis step."""
        analyses = []
        for document in state["documents"]:
            try:
                analysis = await self.legal_analyzer.analyze_document(document)
                analyses.append(analysis)
            except Exception as e:
                state["errors"].append(f"Analysis failed for document: {e}")
        
        state["analyses"] = analyses
        state["current_step"] = "analysis_complete"
        return state
    
    async def extract_citations_step(self, state: AgentState) -> AgentState:
        """Execute citation extraction step."""
        all_citations = []
        for document in state["documents"]:
            try:
                citations = await self.citation_extractor.extract_citations(document)
                all_citations.extend(citations)
            except Exception as e:
                state["errors"].append(f"Citation extraction failed: {e}")
        
        state["citations"] = all_citations
        state["current_step"] = "citations_complete"
        return state
    
    async def synthesize_step(self, state: AgentState) -> AgentState:
        """Execute synthesis step."""
        try:
            consolidated = await self.synthesis_agent.consolidate_documents(
                state["analyses"]
            )
            state["consolidated_output"] = consolidated["consolidated_narrative"]
            state["current_step"] = "synthesis_complete"
        except Exception as e:
            state["errors"].append(f"Synthesis failed: {e}")
        
        return state
```

## Error Handling and Recovery

### Graceful Failure Patterns
```python
async def robust_agent_execution(agent_func, *args, **kwargs):
    """Execute agent function with proper error handling."""
    try:
        return await agent_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return {
            "error": str(e),
            "status": "failed",
            "fallback_available": True
        }

def create_resilient_workflow():
    """Create workflow with error recovery."""
    workflow = StateGraph(AgentState)
    
    # Add conditional edges for error handling
    workflow.add_conditional_edges(
        "analyze",
        lambda state: "extract_citations" if not state["errors"] else "error_recovery"
    )
    
    workflow.add_node("error_recovery", error_recovery_agent)
    return workflow
```

### State Validation
```python
def validate_agent_state(state: AgentState) -> List[str]:
    """Validate agent state consistency."""
    errors = []
    
    if not state["documents"]:
        errors.append("No documents provided for processing")
    
    if state["current_step"] == "analysis_complete" and not state["analyses"]:
        errors.append("Analysis step completed but no analyses generated")
    
    # Validate that analysis count matches document count
    if len(state["analyses"]) != len(state["documents"]):
        errors.append("Mismatch between document and analysis counts")
    
    return errors
```

## Integration with Legal NLP

### Using Hybrid NLP in Agents
```python
class LegalAgent:
    """Base class for legal document processing agents."""
    
    def __init__(self):
        self.nlp = HybridLegalNLP()
        self.logger = structlog.get_logger()
    
    async def process_legal_text(self, text: str) -> Dict[str, Any]:
        """Process legal text with comprehensive analysis."""
        try:
            analysis = self.nlp.comprehensive_analysis(text)
            
            self.logger.info(
                "Legal text processed",
                document_type=analysis["document_analysis"]["type"],
                confidence=analysis["document_analysis"]["confidence"],
                entity_count=len(analysis["entities"]["combined_unique"])
            )
            
            return analysis
        except Exception as e:
            self.logger.error("Legal text processing failed", error=str(e))
            raise
    
    def extract_key_entities(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract key entities for agent decision making."""
        entities = analysis["entities"]["combined_unique"]
        return [entity["text"] for entity in entities if entity.get("confidence", 0) > 0.8]
```

## Performance Considerations

### Async Agent Execution
```python
import asyncio

async def parallel_agent_execution(documents: List[str]) -> List[Dict[str, Any]]:
    """Execute agents in parallel for better performance."""
    nlp = HybridLegalNLP()
    
    # Create tasks for parallel execution
    tasks = [
        nlp.comprehensive_analysis(doc) 
        for doc in documents
    ]
    
    # Execute in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle results and exceptions
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append({"error": str(result)})
        else:
            processed_results.append(result)
    
    return processed_results
```

### Resource Management
```python
class ResourceManagedAgent:
    """Agent with proper resource management."""
    
    def __init__(self):
        self._nlp = None
        self._semaphore = asyncio.Semaphore(5)  # Limit concurrent operations
    
    async def get_nlp(self) -> HybridLegalNLP:
        """Get NLP service with lazy loading."""
        if self._nlp is None:
            self._nlp = HybridLegalNLP()
        return self._nlp
    
    async def process_with_limits(self, text: str) -> Dict[str, Any]:
        """Process text with resource limits."""
        async with self._semaphore:
            nlp = await self.get_nlp()
            return nlp.comprehensive_analysis(text)
```

## Testing Agent Workflows

### Agent Unit Tests
```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_legal_analyzer_agent():
    """Test legal analyzer agent functionality."""
    agent = LegalAnalyzer()
    
    # Mock the NLP service for testing
    agent.nlp.comprehensive_analysis = AsyncMock(return_value={
        "document_analysis": {"type": "lease_agreement", "confidence": 0.9},
        "entities": {"combined_unique": [{"text": "John Smith", "label": "PERSON"}]},
        "entities": {"legal_concepts": {"legal_terms": ["landlord", "tenant"]}}
    })
    
    result = await agent.analyze_document("Sample lease text")
    
    assert result["document_type"] == "lease_agreement"
    assert result["confidence"] == 0.9
    assert len(result["entities"]) > 0

@pytest.mark.asyncio
async def test_workflow_orchestration():
    """Test complete workflow orchestration."""
    orchestrator = LegalWorkflowOrchestrator()
    documents = ["Legal document 1", "Legal document 2"]
    
    result = await orchestrator.process_documents(documents)
    
    assert result["status"] == "completed"
    assert len(result["analyses"]) == len(documents)
    assert "consolidated_output" in result
```

## Reference Files
@agents/orchestrator.py
@agents/legal_analyzer.py
@agents/citation_extractor.py
@agents/synthesis_agent.py
@agents/base.py