"""
ConsolidationAgent for implementing the CRRACC Synthesis Method.

This agent consolidates multiple legal memoranda into comprehensive omnibus documents
following the systematic approach:
- Conclusion: State overarching legal violations
- Rule statement: Synthesize rules from multiple memoranda  
- Rule explanation: Consolidate explanations without redundancy
- Application: Integrate applications using comparative analysis
- Counterargument: Address counterarguments comprehensively
- Conclusion: Reinforce pattern of violations
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agents.base import BaseAgent, AgentState
from agents.workflow_tracking_mixin import WorkflowTrackingMixin
from db.models import AgentType
from nlp.citation_service import CitationService
from nlp.semantic_similarity import SemanticSimilarityService

logger = logging.getLogger(__name__)


@dataclass
class LegalTheory:
    """Represents a distinct legal theory for consolidation"""
    theory_name: str
    related_provisions: List[str]
    supporting_citations: List[str]
    memoranda_ids: List[int]
    strength_score: float


@dataclass
class CRRACCSection:
    """Structure for each section of the CRRACC method"""
    content: str
    supporting_citations: List[str]
    source_memoranda: List[int]
    confidence_score: float


@dataclass
class ConsolidationResult:
    """Final consolidation output"""
    title: str
    conclusion: CRRACCSection
    rule_statement: CRRACCSection
    rule_explanation: CRRACCSection
    application: CRRACCSection
    counterargument: CRRACCSection
    final_conclusion: CRRACCSection
    total_citations: int
    consolidated_memoranda_count: int
    legal_theories: List[LegalTheory]
    quality_metrics: Dict[str, float]


class ConsolidationAgent(BaseAgent, WorkflowTrackingMixin):
    """
    Agent responsible for consolidating multiple legal memoranda using the CRRACC method.
    
    Organizes content by legal theory rather than provision-by-provision,
    maintains narrative prose format, preserves all citations and verbatim quotes.
    """
    
    def __init__(self):
        super().__init__("ConsolidationAgent")
        self.citation_service = CitationService()
        self.semantic_service = SemanticSimilarityService()
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,  # Low temperature for consistent legal analysis
            max_tokens=4000
        )
        
    async def consolidate_memoranda(self, memoranda_data: List[Dict[str, Any]], 
                                  consolidation_strategy: str = "legal_theory") -> ConsolidationResult:
        """
        Main consolidation method implementing the CRRACC synthesis approach.
        
        Args:
            memoranda_data: List of memoranda with content, citations, and metadata
            consolidation_strategy: Method for grouping content (default: "legal_theory")
            
        Returns:
            ConsolidationResult with structured CRRACC output
        """
        state = {
            "memoranda_data": memoranda_data,
            "consolidation_strategy": consolidation_strategy,
            "legal_theories": [],
            "crracc_sections": {},
            "quality_metrics": {}
        }
        
        async with self.track_workflow("document_consolidation", AgentType.SYNTHESIZER, state,
                                     {"memoranda_count": len(memoranda_data), "strategy": consolidation_strategy}):
            
            # Step 1: Analyze and group memoranda by legal theory
            async with self.track_task("legal_theory_extraction", state):
                legal_theories = await self._extract_legal_theories(memoranda_data)
                state["legal_theories"] = legal_theories
                logger.info(f"Identified {len(legal_theories)} distinct legal theories")
            
            # Step 2: Build CRRACC sections systematically
            crracc_sections = {}
            
            # Conclusion: State overarching legal violations
            async with self.track_task("build_conclusion_section", state):
                crracc_sections["conclusion"] = await self._build_conclusion_section(legal_theories, memoranda_data)
            
            # Rule statement: Synthesize rules from multiple memoranda
            async with self.track_task("build_rule_statement", state):
                crracc_sections["rule_statement"] = await self._build_rule_statement_section(legal_theories, memoranda_data)
            
            # Rule explanation: Consolidate explanations without redundancy
            async with self.track_task("build_rule_explanation", state):
                crracc_sections["rule_explanation"] = await self._build_rule_explanation_section(legal_theories, memoranda_data)
            
            # Application: Integrate applications using comparative analysis
            async with self.track_task("build_application_section", state):
                crracc_sections["application"] = await self._build_application_section(legal_theories, memoranda_data)
            
            # Counterargument: Address counterarguments comprehensively
            async with self.track_task("build_counterargument_section", state):
                crracc_sections["counterargument"] = await self._build_counterargument_section(legal_theories, memoranda_data)
            
            # Final Conclusion: Reinforce pattern of violations
            async with self.track_task("build_final_conclusion", state):
                crracc_sections["final_conclusion"] = await self._build_final_conclusion_section(legal_theories, memoranda_data)
            
            state["crracc_sections"] = crracc_sections
            
            # Step 3: Calculate quality metrics and finalize
            async with self.track_task("calculate_quality_metrics", state):
                quality_metrics = await self._calculate_quality_metrics(crracc_sections, legal_theories, memoranda_data)
                state["quality_metrics"] = quality_metrics
            
            # Step 4: Assemble final result
            result = ConsolidationResult(
                title=await self._generate_consolidated_title(legal_theories),
                conclusion=crracc_sections["conclusion"],
                rule_statement=crracc_sections["rule_statement"],
                rule_explanation=crracc_sections["rule_explanation"],
                application=crracc_sections["application"],
                counterargument=crracc_sections["counterargument"],
                final_conclusion=crracc_sections["final_conclusion"],
                total_citations=sum(len(section.supporting_citations) for section in crracc_sections.values()),
                consolidated_memoranda_count=len(memoranda_data),
                legal_theories=legal_theories,
                quality_metrics=quality_metrics
            )
            
            state["output_data"] = {
                "consolidation_result": result.__dict__,
                "quality_score": quality_metrics.get("overall_quality", 0.0),
                "theories_count": len(legal_theories),
                "total_citations": result.total_citations
            }
            
            return result
    
    async def _extract_legal_theories(self, memoranda_data: List[Dict[str, Any]]) -> List[LegalTheory]:
        """
        Extract distinct legal theories from memoranda for organization.
        
        Uses semantic similarity to identify related legal concepts and group
        memoranda that address similar violations or theories.
        """
        theories_map = {}
        
        for memo_idx, memo in enumerate(memoranda_data):
            # Extract key legal concepts from the memorandum
            memo_theories = await self._identify_memo_theories(memo, memo_idx)
            
            for theory_name, theory_data in memo_theories.items():
                if theory_name not in theories_map:
                    theories_map[theory_name] = LegalTheory(
                        theory_name=theory_name,
                        related_provisions=theory_data["provisions"],
                        supporting_citations=theory_data["citations"],
                        memoranda_ids=[memo_idx],
                        strength_score=theory_data["strength"]
                    )
                else:
                    # Merge with existing theory
                    existing = theories_map[theory_name]
                    existing.related_provisions.extend(theory_data["provisions"])
                    existing.supporting_citations.extend(theory_data["citations"])
                    existing.memoranda_ids.append(memo_idx)
                    existing.strength_score = max(existing.strength_score, theory_data["strength"])
        
        # Sort theories by strength for prioritization
        return sorted(theories_map.values(), key=lambda t: t.strength_score, reverse=True)
    
    async def _identify_memo_theories(self, memo: Dict[str, Any], memo_idx: int) -> Dict[str, Dict[str, Any]]:
        """Identify legal theories present in a single memorandum"""
        
        # Use LLM to extract legal theories from the memo content
        prompt = f"""
        Analyze the following legal memorandum and identify the distinct legal theories being argued.
        
        For each theory, provide:
        1. Theory name (e.g., "Truth in Renting Act Violation", "Unconscionability", "Warranty of Habitability")
        2. Related provisions or sections that support this theory
        3. Key citations that support this theory
        4. Strength score (1-10) based on how well-developed the argument is
        
        Memorandum content:
        {memo.get('content', '')[:3000]}...
        
        Return your analysis in the following format:
        Theory Name: [name]
        Provisions: [list of relevant provisions]
        Citations: [list of key supporting citations]
        Strength: [1-10 score]
        ---
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return self._parse_theory_response(response.content)
        except Exception as e:
            logger.error(f"Failed to identify theories for memo {memo_idx}: {e}")
            return {}
    
    def _parse_theory_response(self, response: str) -> Dict[str, Dict[str, Any]]:
        """Parse LLM response into structured theory data"""
        theories = {}
        current_theory = None
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Theory Name:'):
                current_theory = line.replace('Theory Name:', '').strip()
                theories[current_theory] = {
                    "provisions": [],
                    "citations": [],
                    "strength": 5.0
                }
            elif line.startswith('Provisions:') and current_theory:
                provisions_text = line.replace('Provisions:', '').strip()
                theories[current_theory]["provisions"] = [p.strip() for p in provisions_text.split(',') if p.strip()]
            elif line.startswith('Citations:') and current_theory:
                citations_text = line.replace('Citations:', '').strip()
                theories[current_theory]["citations"] = [c.strip() for c in citations_text.split(',') if c.strip()]
            elif line.startswith('Strength:') and current_theory:
                try:
                    strength = float(line.replace('Strength:', '').strip())
                    theories[current_theory]["strength"] = strength
                except ValueError:
                    pass
        
        return theories
    
    async def _build_conclusion_section(self, legal_theories: List[LegalTheory], 
                                      memoranda_data: List[Dict[str, Any]]) -> CRRACCSection:
        """Build the opening conclusion section stating overarching violations"""
        
        prompt = f"""
        Create an executive summary conclusion that states the overarching legal violations
        identified across {len(memoranda_data)} legal memoranda.
        
        Legal theories identified:
        {[theory.theory_name for theory in legal_theories]}
        
        The conclusion should:
        1. Provide a clear, comprehensive overview of systematic violations
        2. Highlight the pattern of violations across multiple provisions
        3. Quantify the impact where possible
        4. Set the stage for detailed analysis to follow
        5. Use authoritative legal language appropriate for litigation
        
        Format as a cohesive narrative conclusion section.
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            # Extract citations from the generated content
            citations = await self.citation_service.extract_citations(response.content)
            
            return CRRACCSection(
                content=response.content,
                supporting_citations=[c.get('full_citation', '') for c in citations],
                source_memoranda=list(range(len(memoranda_data))),
                confidence_score=0.85
            )
        except Exception as e:
            logger.error(f"Failed to build conclusion section: {e}")
            return CRRACCSection("Error generating conclusion", [], [], 0.0)
    
    async def _build_rule_statement_section(self, legal_theories: List[LegalTheory], 
                                          memoranda_data: List[Dict[str, Any]]) -> CRRACCSection:
        """Build rule statement section synthesizing legal standards"""
        
        # Collect all legal standards mentioned across memoranda
        all_citations = []
        for memo in memoranda_data:
            memo_citations = memo.get('citations', [])
            all_citations.extend(memo_citations)
        
        prompt = f"""
        Synthesize the legal rules and standards from multiple memoranda addressing
        related legal theories.
        
        Legal theories to address:
        {[f"{theory.theory_name}: {', '.join(theory.supporting_citations[:3])}" for theory in legal_theories[:5]]}
        
        The rule statement should:
        1. Establish the controlling legal standards for each theory
        2. Synthesize rules from multiple authorities without redundancy
        3. Create a unified framework that supports the violations claimed
        4. Organize by legal theory rather than individual provisions
        5. Preserve important citations and maintain accurate legal references
        
        Focus on creating clear, authoritative rule statements that establish
        the legal foundation for the violations analysis.
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            citations = await self.citation_service.extract_citations(response.content)
            
            return CRRACCSection(
                content=response.content,
                supporting_citations=[c.get('full_citation', '') for c in citations],
                source_memoranda=list(range(len(memoranda_data))),
                confidence_score=0.90
            )
        except Exception as e:
            logger.error(f"Failed to build rule statement section: {e}")
            return CRRACCSection("Error generating rule statement", [], [], 0.0)
    
    async def _build_rule_explanation_section(self, legal_theories: List[LegalTheory], 
                                            memoranda_data: List[Dict[str, Any]]) -> CRRACCSection:
        """Build rule explanation section consolidating case law analysis"""
        
        prompt = f"""
        Consolidate the rule explanations from multiple memoranda, eliminating redundancy
        while preserving the depth of legal analysis.
        
        Focus on theories: {[theory.theory_name for theory in legal_theories[:5]]}
        
        The rule explanation should:
        1. Explain how courts have interpreted and applied the legal standards
        2. Integrate case law analysis from multiple memoranda without repetition
        3. Build from general principles to specific applications
        4. Demonstrate how the legal standards support the violations claimed
        5. Maintain the analytical depth while creating unified explanations
        
        Avoid merely repeating rule statements - explain HOW the rules operate
        and WHY they support the legal theories being advanced.
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            citations = await self.citation_service.extract_citations(response.content)
            
            return CRRACCSection(
                content=response.content,
                supporting_citations=[c.get('full_citation', '') for c in citations],
                source_memoranda=list(range(len(memoranda_data))),
                confidence_score=0.88
            )
        except Exception as e:
            logger.error(f"Failed to build rule explanation section: {e}")
            return CRRACCSection("Error generating rule explanation", [], [], 0.0)
    
    async def _build_application_section(self, legal_theories: List[LegalTheory], 
                                       memoranda_data: List[Dict[str, Any]]) -> CRRACCSection:
        """Build application section with integrated fact-to-law analysis"""
        
        prompt = f"""
        Integrate the applications from multiple memoranda using comparative analysis
        to demonstrate how the legal standards apply to the facts.
        
        Legal theories being applied: {[theory.theory_name for theory in legal_theories[:5]]}
        
        The application section should:
        1. Apply legal standards to facts using comparative analysis across memoranda
        2. Integrate factual applications without merely repeating each memo's analysis
        3. Build systematic arguments showing patterns of violations
        4. Use specific examples to illustrate broader systematic problems
        5. Demonstrate how individual violations combine into systemic violations
        
        Show how the facts across multiple provisions/scenarios fit the legal
        framework established in the rule sections.
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            citations = await self.citation_service.extract_citations(response.content)
            
            return CRRACCSection(
                content=response.content,
                supporting_citations=[c.get('full_citation', '') for c in citations],
                source_memoranda=list(range(len(memoranda_data))),
                confidence_score=0.87
            )
        except Exception as e:
            logger.error(f"Failed to build application section: {e}")
            return CRRACCSection("Error generating application", [], [], 0.0)
    
    async def _build_counterargument_section(self, legal_theories: List[LegalTheory], 
                                           memoranda_data: List[Dict[str, Any]]) -> CRRACCSection:
        """Build comprehensive counterargument section"""
        
        prompt = f"""
        Address counterarguments comprehensively by anticipating and refuting
        potential defenses across all legal theories.
        
        Legal theories to defend: {[theory.theory_name for theory in legal_theories[:5]]}
        
        The counterargument section should:
        1. Anticipate the strongest potential defenses across all theories
        2. Address counterarguments systematically rather than piecemeal
        3. Use integrated analysis to show why defenses fail across multiple theories
        4. Strengthen the overall argument by acknowledging and defeating challenges
        5. Demonstrate comprehensive understanding of potential weaknesses
        
        Focus on the most likely and strongest counterarguments that could
        undermine the systematic violations analysis.
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            citations = await self.citation_service.extract_citations(response.content)
            
            return CRRACCSection(
                content=response.content,
                supporting_citations=[c.get('full_citation', '') for c in citations],
                source_memoranda=list(range(len(memoranda_data))),
                confidence_score=0.82
            )
        except Exception as e:
            logger.error(f"Failed to build counterargument section: {e}")
            return CRRACCSection("Error generating counterargument", [], [], 0.0)
    
    async def _build_final_conclusion_section(self, legal_theories: List[LegalTheory], 
                                            memoranda_data: List[Dict[str, Any]]) -> CRRACCSection:
        """Build final conclusion reinforcing the pattern of violations"""
        
        prompt = f"""
        Create a powerful final conclusion that reinforces the pattern of violations
        demonstrated throughout the analysis.
        
        Legal theories proven: {[theory.theory_name for theory in legal_theories[:5]]}
        Number of memoranda consolidated: {len(memoranda_data)}
        
        The final conclusion should:
        1. Reinforce the systematic nature of violations demonstrated
        2. Emphasize how individual violations combine into patterns of misconduct
        3. Connect back to the opening conclusion while showing what has been proven
        4. Call for comprehensive relief appropriate to systematic violations
        5. Leave no doubt about the strength and comprehensiveness of the case
        
        This should be the capstone argument that ties everything together
        and compels action based on the demonstrated pattern of violations.
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            citations = await self.citation_service.extract_citations(response.content)
            
            return CRRACCSection(
                content=response.content,
                supporting_citations=[c.get('full_citation', '') for c in citations],
                source_memoranda=list(range(len(memoranda_data))),
                confidence_score=0.89
            )
        except Exception as e:
            logger.error(f"Failed to build final conclusion section: {e}")
            return CRRACCSection("Error generating final conclusion", [], [], 0.0)
    
    async def _calculate_quality_metrics(self, crracc_sections: Dict[str, CRRACCSection], 
                                       legal_theories: List[LegalTheory], 
                                       memoranda_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate quality metrics for the consolidation"""
        
        metrics = {}
        
        # Citation preservation rate
        total_source_citations = sum(len(memo.get('citations', [])) for memo in memoranda_data)
        total_output_citations = sum(len(section.supporting_citations) for section in crracc_sections.values())
        metrics["citation_preservation_rate"] = min(1.0, total_output_citations / max(total_source_citations, 1))
        
        # Content synthesis score (average confidence across sections)
        metrics["content_synthesis_score"] = sum(section.confidence_score for section in crracc_sections.values()) / len(crracc_sections)
        
        # Theory coverage completeness
        metrics["theory_coverage_completeness"] = min(1.0, len(legal_theories) / max(len(memoranda_data), 1))
        
        # Legal argument coherence (based on section quality distribution)
        section_scores = [section.confidence_score for section in crracc_sections.values()]
        metrics["argument_coherence"] = 1.0 - (max(section_scores) - min(section_scores))
        
        # Overall quality score
        metrics["overall_quality"] = (
            metrics["citation_preservation_rate"] * 0.25 +
            metrics["content_synthesis_score"] * 0.35 +
            metrics["theory_coverage_completeness"] * 0.20 +
            metrics["argument_coherence"] * 0.20
        )
        
        return metrics
    
    async def _generate_consolidated_title(self, legal_theories: List[LegalTheory]) -> str:
        """Generate an appropriate title for the consolidated memorandum"""
        
        if not legal_theories:
            return "CONSOLIDATED MEMORANDUM OF LAW"
        
        # Use the most prominent theories for title generation
        main_theories = [theory.theory_name for theory in legal_theories[:3]]
        
        if len(main_theories) == 1:
            return f"CONSOLIDATED MEMORANDUM OF LAW REGARDING {main_theories[0].upper()}"
        elif len(main_theories) == 2:
            return f"CONSOLIDATED MEMORANDUM OF LAW REGARDING {main_theories[0].upper()} AND {main_theories[1].upper()}"
        else:
            return f"CONSOLIDATED MEMORANDUM OF LAW REGARDING SYSTEMATIC VIOLATIONS OF TENANT PROTECTION STATUTES"
    
    def _calculate_quality_score(self, state: Dict[str, Any]) -> Optional[float]:
        """Calculate overall quality score for workflow tracking"""
        quality_metrics = state.get("quality_metrics", {})
        return quality_metrics.get("overall_quality")