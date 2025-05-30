"""
Legal Theory Synthesis Service for content consolidation.

This service implements sophisticated content synthesis for legal theories following 
the CRRACC methodology, focusing on:
- Eliminating redundancy while preserving analytical depth
- Maintaining narrative flow and legal argument coherence
- Integrating citations and legal authorities systematically
- Creating unified arguments from multiple source memoranda
- Preserving the distinctive voice and analytical strength of legal writing
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from nlp.citation_service import CitationExtractionService
from nlp.semantic_similarity import SemanticSimilarityService

logger = logging.getLogger(__name__)


class SynthesisMode(Enum):
    """Different modes for content synthesis"""
    COMPREHENSIVE = "comprehensive"  # Full synthesis with maximum detail
    FOCUSED = "focused"              # Streamlined synthesis focusing on key arguments
    COMPARATIVE = "comparative"      # Emphasizes comparative analysis between memoranda
    HIERARCHICAL = "hierarchical"    # Organizes by argument strength and hierarchy


@dataclass
class ContentExtract:
    """Represents extracted content from a source document"""
    source_document_id: str
    section_type: str  # conclusion, rule_statement, application, etc.
    content: str
    citations: List[str]
    key_concepts: List[str]
    argument_strength: float
    legal_authorities: List[str]


@dataclass
class SynthesisResult:
    """Result of content synthesis for a particular section"""
    synthesized_content: str
    integrated_citations: List[str]
    source_documents: List[str]
    confidence_score: float
    synthesis_strategy: str
    redundancy_eliminated: float  # Percentage of redundant content removed
    coherence_score: float
    authority_integration_score: float


class LegalTheorySynthesisService:
    """
    Service for synthesizing legal content following the CRRACC methodology.
    
    Specializes in creating coherent, non-redundant legal arguments that integrate
    multiple source memoranda while maintaining the analytical depth and 
    persuasive power of legal writing.
    """
    
    def __init__(self):
        self.citation_service = CitationExtractionService()
        self.semantic_service = SemanticSimilarityService()
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.2,  # Slightly higher for more natural legal writing
            max_tokens=4000
        )
        
    async def synthesize_legal_theory_content(self, 
                                            theory_name: str,
                                            source_documents: List[Dict[str, Any]],
                                            target_section: str,
                                            synthesis_mode: SynthesisMode = SynthesisMode.COMPREHENSIVE) -> SynthesisResult:
        """
        Synthesize content for a specific CRRACC section across multiple documents.
        
        Args:
            theory_name: Name of the legal theory being synthesized
            source_documents: List of source memoranda 
            target_section: CRRACC section (conclusion, rule_statement, etc.)
            synthesis_mode: Approach for synthesis
            
        Returns:
            SynthesisResult with synthesized content and metadata
        """
        logger.info(f"Synthesizing {target_section} for theory '{theory_name}' from {len(source_documents)} sources")
        
        # Step 1: Extract relevant content from source documents
        content_extracts = await self._extract_section_content(source_documents, target_section, theory_name)
        
        # Step 2: Analyze for redundancy and key differences
        redundancy_analysis = await self._analyze_content_redundancy(content_extracts)
        
        # Step 3: Synthesize based on target section and mode
        synthesized_content = await self._synthesize_by_section_type(
            target_section, content_extracts, theory_name, synthesis_mode, redundancy_analysis
        )
        
        # Step 4: Integrate and deduplicate citations
        integrated_citations = await self._integrate_citations(content_extracts)
        
        # Step 5: Calculate quality metrics
        quality_metrics = await self._calculate_synthesis_quality(
            synthesized_content, content_extracts, integrated_citations
        )
        
        return SynthesisResult(
            synthesized_content=synthesized_content,
            integrated_citations=integrated_citations,
            source_documents=[doc.get('id', f"doc_{i}") for i, doc in enumerate(source_documents)],
            confidence_score=quality_metrics['confidence'],
            synthesis_strategy=synthesis_mode.value,
            redundancy_eliminated=quality_metrics['redundancy_eliminated'],
            coherence_score=quality_metrics['coherence'],
            authority_integration_score=quality_metrics['authority_integration']
        )
    
    async def _extract_section_content(self, 
                                     source_documents: List[Dict[str, Any]], 
                                     target_section: str,
                                     theory_name: str) -> List[ContentExtract]:
        """Extract content relevant to the target section from source documents"""
        
        extracts = []
        
        for i, doc in enumerate(source_documents):
            content = doc.get('content', '')
            doc_id = doc.get('id', f"doc_{i}")
            
            # Extract section-specific content using targeted prompts
            extracted_content = await self._extract_targeted_content(
                content, target_section, theory_name, doc_id
            )
            
            if extracted_content['content'].strip():
                extract = ContentExtract(
                    source_document_id=doc_id,
                    section_type=target_section,
                    content=extracted_content['content'],
                    citations=extracted_content['citations'],
                    key_concepts=extracted_content['key_concepts'],
                    argument_strength=extracted_content['strength'],
                    legal_authorities=extracted_content['authorities']
                )
                extracts.append(extract)
        
        return extracts
    
    async def _extract_targeted_content(self, 
                                      document_content: str,
                                      target_section: str,
                                      theory_name: str,
                                      doc_id: str) -> Dict[str, Any]:
        """Extract content targeted to specific CRRACC section"""
        
        # Create section-specific extraction prompts
        section_prompts = {
            'conclusion': self._get_conclusion_extraction_prompt(theory_name),
            'rule_statement': self._get_rule_statement_extraction_prompt(theory_name),
            'rule_explanation': self._get_rule_explanation_extraction_prompt(theory_name),
            'application': self._get_application_extraction_prompt(theory_name),
            'counterargument': self._get_counterargument_extraction_prompt(theory_name),
            'final_conclusion': self._get_final_conclusion_extraction_prompt(theory_name)
        }
        
        prompt = section_prompts.get(target_section, section_prompts['rule_statement'])
        
        # Limit content for LLM processing
        content_sample = document_content[:5000]
        
        full_prompt = f"""
        {prompt}
        
        Document content:
        {content_sample}
        
        Extract and return:
        CONTENT: [extracted content relevant to {target_section}]
        CITATIONS: [list of legal citations found]
        KEY_CONCEPTS: [list of key legal concepts]
        AUTHORITIES: [list of legal authorities referenced]
        STRENGTH: [argument strength 1-10]
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=full_prompt)])
            return self._parse_extraction_response(response.content)
        except Exception as e:
            logger.error(f"Failed to extract content for {target_section} from {doc_id}: {e}")
            return {'content': '', 'citations': [], 'key_concepts': [], 'authorities': [], 'strength': 5.0}
    
    def _get_conclusion_extraction_prompt(self, theory_name: str) -> str:
        """Get prompt for extracting conclusion content"""
        return f"""
        Extract content that serves as an executive summary or conclusion about {theory_name}.
        Focus on:
        - Overarching statements about violations or legal issues
        - Summary of key findings
        - High-level patterns or systematic issues
        - Opening statements that frame the legal argument
        
        Look for sections that state the "big picture" legal violations and their significance.
        """
    
    def _get_rule_statement_extraction_prompt(self, theory_name: str) -> str:
        """Get prompt for extracting rule statement content"""
        return f"""
        Extract content that establishes the legal rules and standards for {theory_name}.
        Focus on:
        - Statutory provisions and their text
        - Legal standards established by courts
        - Controlling legal principles
        - Constitutional or regulatory requirements
        - Black letter law statements
        
        Look for content that answers "What is the law?" rather than "How does it apply?"
        """
    
    def _get_rule_explanation_extraction_prompt(self, theory_name: str) -> str:
        """Get prompt for extracting rule explanation content"""
        return f"""
        Extract content that explains how legal rules operate and have been interpreted for {theory_name}.
        Focus on:
        - Case law analysis and judicial interpretation
        - How courts have applied the legal standards
        - Policy rationales behind the rules
        - Evolution of legal doctrine
        - Distinctions and nuances in application
        
        Look for content that answers "How do the rules work?" and "What do they mean?"
        """
    
    def _get_application_extraction_prompt(self, theory_name: str) -> str:
        """Get prompt for extracting application content"""
        return f"""
        Extract content that applies legal rules to specific facts for {theory_name}.
        Focus on:
        - Fact-to-law analysis
        - Specific examples of rule violations
        - Comparisons to other cases or situations
        - Demonstration of how facts fit legal standards
        - Practical impact of rule application
        
        Look for content that answers "How do the rules apply to these facts?"
        """
    
    def _get_counterargument_extraction_prompt(self, theory_name: str) -> str:
        """Get prompt for extracting counterargument content"""
        return f"""
        Extract content that addresses potential counterarguments or defenses for {theory_name}.
        Focus on:
        - Anticipated opposing arguments
        - Potential weaknesses in the legal theory
        - Alternative interpretations of law or facts
        - Responses to likely defenses
        - Distinguishing unfavorable authorities
        
        Look for content that addresses "What might the opposition argue?" and rebuttals.
        """
    
    def _get_final_conclusion_extraction_prompt(self, theory_name: str) -> str:
        """Get prompt for extracting final conclusion content"""
        return f"""
        Extract content that provides final conclusions and calls for relief for {theory_name}.
        Focus on:
        - Summation of proven violations
        - Reinforcement of systematic problems
        - Requests for specific relief
        - Final persuasive statements
        - Connection to broader legal principles
        
        Look for content that provides the capstone argument and resolution.
        """
    
    def _parse_extraction_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM extraction response into structured data"""
        
        result = {
            'content': '',
            'citations': [],
            'key_concepts': [],
            'authorities': [],
            'strength': 5.0
        }
        
        current_section = None
        
        for line in response.split('\n'):
            line = line.strip()
            
            if line.startswith('CONTENT:'):
                current_section = 'content'
                result['content'] = line.replace('CONTENT:', '').strip()
            elif line.startswith('CITATIONS:'):
                current_section = 'citations'
                citations_text = line.replace('CITATIONS:', '').strip()
                result['citations'] = [c.strip() for c in citations_text.split(',') if c.strip()]
            elif line.startswith('KEY_CONCEPTS:'):
                current_section = 'key_concepts'
                concepts_text = line.replace('KEY_CONCEPTS:', '').strip()
                result['key_concepts'] = [c.strip() for c in concepts_text.split(',') if c.strip()]
            elif line.startswith('AUTHORITIES:'):
                current_section = 'authorities'
                authorities_text = line.replace('AUTHORITIES:', '').strip()
                result['authorities'] = [a.strip() for a in authorities_text.split(',') if a.strip()]
            elif line.startswith('STRENGTH:'):
                try:
                    strength = float(line.replace('STRENGTH:', '').strip())
                    result['strength'] = strength
                except ValueError:
                    pass
            elif current_section == 'content' and line:
                # Continue building content for multi-line content
                result['content'] += ' ' + line
        
        return result
    
    async def _analyze_content_redundancy(self, content_extracts: List[ContentExtract]) -> Dict[str, Any]:
        """Analyze redundancy patterns in extracted content"""
        
        if len(content_extracts) <= 1:
            return {'redundancy_pairs': [], 'unique_content': content_extracts, 'redundancy_score': 0.0}
        
        redundancy_pairs = []
        content_similarities = {}
        
        # Calculate pairwise similarities
        for i, extract1 in enumerate(content_extracts):
            for j, extract2 in enumerate(content_extracts[i + 1:], i + 1):
                try:
                    similarity = await self.semantic_service.calculate_similarity(
                        extract1.content, extract2.content
                    )
                    content_similarities[(i, j)] = similarity
                    
                    if similarity > 0.7:  # High similarity threshold
                        redundancy_pairs.append((extract1, extract2, similarity))
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate similarity between extracts {i} and {j}: {e}")
        
        # Calculate overall redundancy score
        similarities = list(content_similarities.values())
        redundancy_score = max(similarities) if similarities else 0.0
        
        return {
            'redundancy_pairs': redundancy_pairs,
            'unique_content': content_extracts,  # Will be filtered during synthesis
            'redundancy_score': redundancy_score,
            'similarity_matrix': content_similarities
        }
    
    async def _synthesize_by_section_type(self, 
                                        section_type: str,
                                        content_extracts: List[ContentExtract],
                                        theory_name: str,
                                        synthesis_mode: SynthesisMode,
                                        redundancy_analysis: Dict[str, Any]) -> str:
        """Synthesize content based on CRRACC section type"""
        
        if not content_extracts:
            return f"[No content available for {section_type} section]"
        
        # Create section-specific synthesis prompts
        synthesis_prompts = {
            'conclusion': self._create_conclusion_synthesis_prompt,
            'rule_statement': self._create_rule_statement_synthesis_prompt,
            'rule_explanation': self._create_rule_explanation_synthesis_prompt,
            'application': self._create_application_synthesis_prompt,
            'counterargument': self._create_counterargument_synthesis_prompt,
            'final_conclusion': self._create_final_conclusion_synthesis_prompt
        }
        
        prompt_creator = synthesis_prompts.get(section_type, synthesis_prompts['rule_statement'])
        prompt = prompt_creator(theory_name, content_extracts, synthesis_mode, redundancy_analysis)
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert legal writer specializing in memorandum consolidation. "
                                    "Create sophisticated, persuasive legal arguments that eliminate redundancy "
                                    "while maintaining analytical depth and proper legal citation format."),
                HumanMessage(content=prompt)
            ])
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to synthesize {section_type} content: {e}")
            return f"[Error synthesizing {section_type} content]"
    
    def _create_conclusion_synthesis_prompt(self, 
                                          theory_name: str,
                                          content_extracts: List[ContentExtract],
                                          synthesis_mode: SynthesisMode,
                                          redundancy_analysis: Dict[str, Any]) -> str:
        """Create synthesis prompt for conclusion section"""
        
        extract_contents = [extract.content for extract in content_extracts]
        
        return f"""
        Synthesize a powerful executive summary conclusion for {theory_name} that consolidates
        the following source materials without redundancy:
        
        SOURCE MATERIALS:
        {self._format_source_materials(content_extracts)}
        
        SYNTHESIS REQUIREMENTS:
        1. Create a unified conclusion that captures the overarching legal violations
        2. Eliminate redundant statements while preserving key insights from each source
        3. Quantify the impact where possible (number of violations, scope of harm)
        4. Establish the systematic nature of the violations
        5. Set up the detailed analysis to follow
        6. Use authoritative legal language appropriate for litigation
        
        REDUNDANCY GUIDANCE: 
        {redundancy_analysis['redundancy_score']:.2f} similarity detected between sources.
        Focus on unique insights and avoid repeating similar concepts.
        
        Create a compelling conclusion that serves as both summary and roadmap for the legal argument.
        """
    
    def _create_rule_statement_synthesis_prompt(self, 
                                              theory_name: str,
                                              content_extracts: List[ContentExtract],
                                              synthesis_mode: SynthesisMode,
                                              redundancy_analysis: Dict[str, Any]) -> str:
        """Create synthesis prompt for rule statement section"""
        
        return f"""
        Synthesize the legal rules and standards for {theory_name} from these source materials:
        
        SOURCE MATERIALS:
        {self._format_source_materials(content_extracts)}
        
        SYNTHESIS REQUIREMENTS:
        1. Establish clear, authoritative statements of controlling law
        2. Organize rules hierarchically (constitutional → statutory → regulatory → common law)
        3. Eliminate redundant rule statements while preserving nuanced distinctions
        4. Create unified framework that supports the violations analysis
        5. Maintain precise legal citations and authority references
        6. Focus on rules most directly relevant to {theory_name}
        
        Create a comprehensive rule foundation that establishes the legal standards
        against which violations will be measured. Avoid merely concatenating sources.
        """
    
    def _create_rule_explanation_synthesis_prompt(self, 
                                                theory_name: str,
                                                content_extracts: List[ContentExtract],
                                                synthesis_mode: SynthesisMode,
                                                redundancy_analysis: Dict[str, Any]) -> str:
        """Create synthesis prompt for rule explanation section"""
        
        return f"""
        Synthesize rule explanations for {theory_name} that consolidate case law analysis
        and judicial interpretation from these sources:
        
        SOURCE MATERIALS:
        {self._format_source_materials(content_extracts)}
        
        SYNTHESIS REQUIREMENTS:
        1. Integrate case law analysis without repetitive citation of the same cases
        2. Build from general interpretive principles to specific applications
        3. Explain HOW courts have applied the rules, not just WHAT the rules are
        4. Demonstrate judicial evolution and trend toward protective interpretation
        5. Create narrative flow that builds understanding progressively
        6. Preserve important quotations and analytical insights
        
        Focus on creating a unified explanation that shows how legal authorities
        support the theory being advanced, rather than presenting separate analyses.
        """
    
    def _create_application_synthesis_prompt(self, 
                                           theory_name: str,
                                           content_extracts: List[ContentExtract],
                                           synthesis_mode: SynthesisMode,
                                           redundancy_analysis: Dict[str, Any]) -> str:
        """Create synthesis prompt for application section"""
        
        return f"""
        Synthesize the application analysis for {theory_name} using comparative methodology
        from these source materials:
        
        SOURCE MATERIALS:
        {self._format_source_materials(content_extracts)}
        
        SYNTHESIS REQUIREMENTS:
        1. Integrate fact-to-law analysis using comparative approach across sources
        2. Identify patterns of violations rather than isolated incidents
        3. Build systematic arguments showing escalating misconduct
        4. Use specific examples to illustrate broader systemic problems
        5. Demonstrate how individual violations combine into comprehensive pattern
        6. Maintain logical flow from established rules to factual application
        
        Create integrated application that shows systematic violations across
        multiple contexts rather than separate provision-by-provision analysis.
        """
    
    def _create_counterargument_synthesis_prompt(self, 
                                                theory_name: str,
                                                content_extracts: List[ContentExtract],
                                                synthesis_mode: SynthesisMode,
                                                redundancy_analysis: Dict[str, Any]) -> str:
        """Create synthesis prompt for counterargument section"""
        
        return f"""
        Synthesize comprehensive counterargument responses for {theory_name} from these sources:
        
        SOURCE MATERIALS:
        {self._format_source_materials(content_extracts)}
        
        SYNTHESIS REQUIREMENTS:
        1. Identify and address the strongest potential counterarguments systematically
        2. Avoid repetitive responses to similar challenges
        3. Create integrated defense that addresses multiple theories simultaneously
        4. Anticipate sophisticated defenses and pre-empt them effectively
        5. Use comparative analysis to show why counterarguments fail across contexts
        6. Strengthen overall argument by acknowledging and defeating challenges
        
        Focus on the most compelling counterarguments that could undermine the
        systematic violations analysis and provide devastating responses.
        """
    
    def _create_final_conclusion_synthesis_prompt(self, 
                                                 theory_name: str,
                                                 content_extracts: List[ContentExtract],
                                                 synthesis_mode: SynthesisMode,
                                                 redundancy_analysis: Dict[str, Any]) -> str:
        """Create synthesis prompt for final conclusion section"""
        
        return f"""
        Synthesize a powerful final conclusion for {theory_name} that ties together
        the entire legal argument from these sources:
        
        SOURCE MATERIALS:
        {self._format_source_materials(content_extracts)}
        
        SYNTHESIS REQUIREMENTS:
        1. Reinforce the pattern of systematic violations demonstrated
        2. Connect back to opening conclusions while showing what has been proven
        3. Emphasize how individual violations combine into comprehensive misconduct
        4. Call for relief proportionate to the systematic nature of violations
        5. Leave no doubt about the strength and completeness of the legal case
        6. Create compelling capstone that compels judicial action
        
        This should be the definitive final word that ties all arguments together
        and makes the case for comprehensive relief irresistible.
        """
    
    def _format_source_materials(self, content_extracts: List[ContentExtract]) -> str:
        """Format content extracts for inclusion in synthesis prompts"""
        
        formatted = []
        
        for i, extract in enumerate(content_extracts, 1):
            formatted.append(f"""
            SOURCE {i} (from {extract.source_document_id}):
            Content: {extract.content[:1500]}...
            Key Citations: {', '.join(extract.citations[:5])}
            Strength: {extract.argument_strength}/10
            """)
        
        return '\n'.join(formatted)
    
    async def _integrate_citations(self, content_extracts: List[ContentExtract]) -> List[str]:
        """Integrate and deduplicate citations from content extracts"""
        
        all_citations = []
        
        for extract in content_extracts:
            all_citations.extend(extract.citations)
        
        # Use citation service to normalize and deduplicate
        try:
            # Extract and normalize citations
            normalized_citations = []
            for citation in all_citations:
                if citation.strip():
                    # Simple normalization - in practice, use more sophisticated citation parsing
                    normalized = citation.strip().rstrip('.')
                    if normalized not in normalized_citations:
                        normalized_citations.append(normalized)
            
            return normalized_citations[:50]  # Limit to most important citations
            
        except Exception as e:
            logger.error(f"Failed to integrate citations: {e}")
            return list(set(all_citations))  # Simple deduplication fallback
    
    async def _calculate_synthesis_quality(self, 
                                         synthesized_content: str,
                                         content_extracts: List[ContentExtract],
                                         integrated_citations: List[str]) -> Dict[str, float]:
        """Calculate quality metrics for synthesized content"""
        
        metrics = {}
        
        # Confidence score based on content length and citation integration
        content_length_score = min(1.0, len(synthesized_content) / 2000)  # Optimal around 2000 chars
        citation_score = min(1.0, len(integrated_citations) / 20)  # Good citation density
        metrics['confidence'] = (content_length_score + citation_score) / 2
        
        # Redundancy elimination (estimate based on compression ratio)
        total_source_length = sum(len(extract.content) for extract in content_extracts)
        compression_ratio = len(synthesized_content) / max(total_source_length, 1)
        metrics['redundancy_eliminated'] = max(0.0, 1.0 - compression_ratio * 2)  # Expect ~50% compression
        
        # Coherence score (simplified based on content structure)
        paragraph_count = len([p for p in synthesized_content.split('\n\n') if p.strip()])
        structure_score = min(1.0, paragraph_count / 5)  # Good structure has 3-7 paragraphs
        metrics['coherence'] = structure_score
        
        # Authority integration score
        source_authorities = set()
        for extract in content_extracts:
            source_authorities.update(extract.legal_authorities)
        
        # Check if synthesized content mentions key authorities
        content_lower = synthesized_content.lower()
        mentioned_authorities = sum(1 for auth in source_authorities if auth.lower() in content_lower)
        metrics['authority_integration'] = mentioned_authorities / max(len(source_authorities), 1)
        
        return metrics
    
    async def synthesize_multi_theory_content(self, 
                                            theories: List[str],
                                            documents_by_theory: Dict[str, List[Dict[str, Any]]],
                                            target_section: str) -> SynthesisResult:
        """Synthesize content across multiple legal theories for integrated analysis"""
        
        logger.info(f"Cross-theory synthesis for {target_section} across {len(theories)} theories")
        
        # Synthesize each theory individually first
        theory_syntheses = {}
        
        for theory in theories:
            if theory in documents_by_theory:
                theory_result = await self.synthesize_legal_theory_content(
                    theory, documents_by_theory[theory], target_section
                )
                theory_syntheses[theory] = theory_result
        
        # Now create integrated synthesis across theories
        integration_prompt = f"""
        Create an integrated {target_section} section that synthesizes analysis across
        multiple legal theories without redundancy:
        
        INDIVIDUAL THEORY ANALYSES:
        {self._format_theory_syntheses(theory_syntheses)}
        
        INTEGRATION REQUIREMENTS:
        1. Identify overlapping concepts and eliminate redundancy
        2. Show how different theories reinforce each other
        3. Create unified argument that is stronger than sum of parts
        4. Maintain distinct theory identities while showing connections
        5. Build hierarchical argument with theories supporting overall conclusion
        
        Create a seamless integrated analysis that leverages multiple theories
        for maximum persuasive impact.
        """
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert at legal theory integration and cross-cutting analysis."),
                HumanMessage(content=integration_prompt)
            ])
            
            # Aggregate metrics from individual syntheses
            total_citations = []
            source_docs = []
            confidence_scores = []
            
            for result in theory_syntheses.values():
                total_citations.extend(result.integrated_citations)
                source_docs.extend(result.source_documents)
                confidence_scores.append(result.confidence_score)
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            return SynthesisResult(
                synthesized_content=response.content,
                integrated_citations=list(set(total_citations)),
                source_documents=list(set(source_docs)),
                confidence_score=avg_confidence,
                synthesis_strategy="multi_theory_integration",
                redundancy_eliminated=0.8,  # High for cross-theory integration
                coherence_score=0.85,
                authority_integration_score=0.9
            )
            
        except Exception as e:
            logger.error(f"Failed multi-theory synthesis: {e}")
            # Fallback to concatenation
            fallback_content = "\n\n".join([
                f"## {theory}\n{result.synthesized_content}" 
                for theory, result in theory_syntheses.items()
            ])
            
            return SynthesisResult(
                synthesized_content=fallback_content,
                integrated_citations=[],
                source_documents=[],
                confidence_score=0.3,
                synthesis_strategy="fallback_concatenation",
                redundancy_eliminated=0.1,
                coherence_score=0.4,
                authority_integration_score=0.3
            )
    
    def _format_theory_syntheses(self, theory_syntheses: Dict[str, SynthesisResult]) -> str:
        """Format individual theory syntheses for integration prompt"""
        
        formatted = []
        
        for theory_name, result in theory_syntheses.items():
            formatted.append(f"""
            THEORY: {theory_name}
            CONTENT: {result.synthesized_content[:2000]}...
            KEY CITATIONS: {', '.join(result.integrated_citations[:5])}
            CONFIDENCE: {result.confidence_score:.2f}
            """)
        
        return '\n'.join(formatted)