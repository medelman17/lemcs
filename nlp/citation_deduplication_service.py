"""
Citation Preservation and Deduplication Service.

This service provides sophisticated citation management for legal document consolidation:
- Preserves all citations with accurate formatting and context
- Identifies and eliminates duplicate citations across multiple documents
- Normalizes citation formats for consistency
- Groups related citations and creates cross-references
- Maintains citation context and authority rankings
- Supports multiple citation formats (Bluebook, local rules, etc.)
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import difflib

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from nlp.citation_service import CitationService

logger = logging.getLogger(__name__)


class CitationType(Enum):
    """Types of legal citations"""
    CASE = "case"
    STATUTE = "statute"
    REGULATION = "regulation"
    CONSTITUTIONAL = "constitutional"
    SECONDARY = "secondary"
    UNKNOWN = "unknown"


class CitationFormat(Enum):
    """Citation format standards"""
    BLUEBOOK = "bluebook"
    NEW_JERSEY = "new_jersey"
    ALWD = "alwd"
    LOCAL = "local"


@dataclass
class CitationMatch:
    """Represents a match between citations"""
    citation1_id: str
    citation2_id: str
    similarity_score: float
    match_type: str  # exact, substantial, partial
    differences: List[str]


@dataclass
class NormalizedCitation:
    """Normalized citation with metadata"""
    original_text: str
    normalized_text: str
    citation_type: CitationType
    format_style: CitationFormat
    authority_level: int  # 1=highest (Supreme Court), 5=lowest (secondary)
    jurisdiction: str
    year: Optional[int] = None
    court: Optional[str] = None
    case_name: Optional[str] = None
    statute_section: Optional[str] = None
    page_numbers: Optional[str] = None
    parenthetical: Optional[str] = None
    contexts: List[str] = field(default_factory=list)  # Where citation appears
    source_documents: List[str] = field(default_factory=list)


@dataclass
class CitationCluster:
    """Group of related citations"""
    cluster_id: str
    primary_citation: NormalizedCitation
    alternative_citations: List[NormalizedCitation]
    citation_type: CitationType
    authority_ranking: int
    usage_frequency: int
    contexts: List[str]
    consolidated_format: str


@dataclass
class DeduplicationResult:
    """Result of citation deduplication process"""
    original_citation_count: int
    deduplicated_citation_count: int
    citation_clusters: List[CitationCluster]
    duplicate_pairs: List[CitationMatch]
    normalization_changes: Dict[str, str]
    authority_rankings: Dict[str, int]
    format_consistency_score: float
    preservation_rate: float


class CitationDeduplicationService:
    """
    Service for preserving and deduplicating citations in legal document consolidation.
    
    Provides comprehensive citation management including normalization, deduplication,
    authority ranking, and format consistency enforcement.
    """
    
    def __init__(self):
        self.citation_service = CitationService()
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            max_tokens=2000
        )
        
        # Citation pattern regex for basic identification
        self.citation_patterns = {
            'case': [
                r'\b\d+\s+N\.J\.\s+\d+',  # New Jersey Reports
                r'\b\d+\s+N\.J\.\s+Super\.\s+\d+',  # New Jersey Superior Court
                r'\b\d+\s+U\.S\.\s+\d+',  # U.S. Reports
                r'\b\d+\s+S\.\s*Ct\.\s+\d+',  # Supreme Court Reporter
                r'\b\d+\s+F\.\d+d\s+\d+',  # Federal Reporter
            ],
            'statute': [
                r'N\.J\.S\.A\.\s+[\d\w:\-\.]+',  # New Jersey Statutes
                r'\d+\s+U\.S\.C\.\s+§?\s*\d+',  # United States Code
                r'N\.J\.A\.C\.\s+[\d:\-\.]+',  # New Jersey Administrative Code
            ],
            'regulation': [
                r'N\.J\.A\.C\.\s+[\d:\-\.]+',
                r'\d+\s+C\.F\.R\.\s+§?\s*[\d\.]+',
            ]
        }
    
    async def deduplicate_citations(self, 
                                  source_documents: List[Dict[str, Any]],
                                  preserve_context: bool = True,
                                  normalize_format: CitationFormat = CitationFormat.NEW_JERSEY) -> DeduplicationResult:
        """
        Main deduplication process for citations across multiple documents.
        
        Args:
            source_documents: List of documents with content and metadata
            preserve_context: Whether to preserve citation context information
            normalize_format: Target citation format for normalization
            
        Returns:
            DeduplicationResult with deduplicated citations and metadata
        """
        logger.info(f"Starting citation deduplication for {len(source_documents)} documents")
        
        # Step 1: Extract all citations from source documents
        all_citations = await self._extract_all_citations(source_documents, preserve_context)
        logger.info(f"Extracted {len(all_citations)} total citations")
        
        # Step 2: Normalize citation formats
        normalized_citations = await self._normalize_citations(all_citations, normalize_format)
        logger.info(f"Normalized {len(normalized_citations)} citations")
        
        # Step 3: Identify duplicate and similar citations
        duplicate_pairs = await self._identify_duplicates(normalized_citations)
        logger.info(f"Found {len(duplicate_pairs)} duplicate pairs")
        
        # Step 4: Create citation clusters
        citation_clusters = await self._create_citation_clusters(normalized_citations, duplicate_pairs)
        logger.info(f"Created {len(citation_clusters)} citation clusters")
        
        # Step 5: Rank citations by authority
        authority_rankings = self._calculate_authority_rankings(citation_clusters)
        
        # Step 6: Calculate quality metrics
        quality_metrics = self._calculate_deduplication_quality(
            all_citations, citation_clusters, duplicate_pairs
        )
        
        return DeduplicationResult(
            original_citation_count=len(all_citations),
            deduplicated_citation_count=len(citation_clusters),
            citation_clusters=citation_clusters,
            duplicate_pairs=duplicate_pairs,
            normalization_changes={},  # Would track specific changes
            authority_rankings=authority_rankings,
            format_consistency_score=quality_metrics['format_consistency'],
            preservation_rate=quality_metrics['preservation_rate']
        )
    
    async def _extract_all_citations(self, 
                                   source_documents: List[Dict[str, Any]],
                                   preserve_context: bool) -> List[NormalizedCitation]:
        """Extract citations from all source documents with context"""
        
        all_citations = []
        
        for doc_idx, doc in enumerate(source_documents):
            content = doc.get('content', '')
            doc_id = doc.get('id', f"doc_{doc_idx}")
            
            # Extract citations using the citation service
            try:
                extracted_citations = await self.citation_service.extract_citations(content)
                
                for citation_data in extracted_citations:
                    # Create initial normalized citation
                    citation = NormalizedCitation(
                        original_text=citation_data.get('full_citation', ''),
                        normalized_text=citation_data.get('full_citation', ''),
                        citation_type=self._classify_citation_type(citation_data.get('full_citation', '')),
                        format_style=CitationFormat.UNKNOWN,
                        authority_level=5,  # Will be recalculated
                        jurisdiction="New Jersey",  # Default, will be refined
                        source_documents=[doc_id]
                    )
                    
                    # Extract additional metadata
                    citation = await self._enrich_citation_metadata(citation, citation_data)
                    
                    # Add context if requested
                    if preserve_context:
                        citation.contexts = self._extract_citation_context(content, citation.original_text)
                    
                    all_citations.append(citation)
                    
            except Exception as e:
                logger.error(f"Failed to extract citations from document {doc_id}: {e}")
        
        return all_citations
    
    def _classify_citation_type(self, citation_text: str) -> CitationType:
        """Classify citation by type based on text patterns"""
        
        citation_lower = citation_text.lower()
        
        # Check for statute patterns
        if any(pattern in citation_lower for pattern in ['n.j.s.a.', 'u.s.c.', 'c.f.r.']):
            return CitationType.STATUTE
        
        # Check for regulation patterns  
        if any(pattern in citation_lower for pattern in ['n.j.a.c.', 'c.f.r.']):
            return CitationType.REGULATION
        
        # Check for constitutional provisions
        if any(pattern in citation_lower for pattern in ['const.', 'constitution', 'article', 'amendment']):
            return CitationType.CONSTITUTIONAL
        
        # Check for case patterns
        if any(pattern in citation_lower for pattern in ['n.j.', 'u.s.', 'f.2d', 'f.3d', 'f.supp', 'a.2d']):
            return CitationType.CASE
        
        # Secondary sources
        if any(pattern in citation_lower for pattern in ['law review', 'journal', 'treatise', 'am jur', 'c.j.s.']):
            return CitationType.SECONDARY
        
        return CitationType.UNKNOWN
    
    async def _enrich_citation_metadata(self, 
                                      citation: NormalizedCitation, 
                                      citation_data: Dict[str, Any]) -> NormalizedCitation:
        """Enrich citation with additional metadata"""
        
        # Extract year from citation
        year_match = re.search(r'\b(19|20)\d{2}\b', citation.original_text)
        if year_match:
            citation.year = int(year_match.group())
        
        # Extract case name for case citations
        if citation.citation_type == CitationType.CASE:
            citation.case_name = citation_data.get('case_name', '')
            citation.court = self._extract_court_from_citation(citation.original_text)
        
        # Extract statute section for statutes
        if citation.citation_type == CitationType.STATUTE:
            citation.statute_section = self._extract_statute_section(citation.original_text)
        
        # Set authority level based on citation type and source
        citation.authority_level = self._calculate_initial_authority_level(citation)
        
        return citation
    
    def _extract_court_from_citation(self, citation_text: str) -> Optional[str]:
        """Extract court information from case citation"""
        
        court_patterns = {
            'N.J.': 'New Jersey Supreme Court',
            'N.J. Super.': 'New Jersey Superior Court',
            'U.S.': 'United States Supreme Court',
            'F.2d': 'Federal Court of Appeals',
            'F.3d': 'Federal Court of Appeals',
            'F.Supp': 'Federal District Court'
        }
        
        for pattern, court_name in court_patterns.items():
            if pattern in citation_text:
                return court_name
        
        return None
    
    def _extract_statute_section(self, citation_text: str) -> Optional[str]:
        """Extract statute section from statutory citation"""
        
        # Pattern for N.J.S.A. sections
        njsa_pattern = r'N\.J\.S\.A\.\s+([\d\w:\-\.]+)'
        match = re.search(njsa_pattern, citation_text)
        if match:
            return match.group(1)
        
        # Pattern for U.S.C. sections
        usc_pattern = r'(\d+)\s+U\.S\.C\.\s+§?\s*(\d+)'
        match = re.search(usc_pattern, citation_text)
        if match:
            return f"{match.group(1)} U.S.C. § {match.group(2)}"
        
        return None
    
    def _calculate_initial_authority_level(self, citation: NormalizedCitation) -> int:
        """Calculate initial authority level (1=highest, 5=lowest)"""
        
        citation_text = citation.original_text.lower()
        
        # Supreme Court cases = 1
        if 'u.s.' in citation_text and citation.citation_type == CitationType.CASE:
            return 1
        
        # Constitutional provisions = 1
        if citation.citation_type == CitationType.CONSTITUTIONAL:
            return 1
        
        # Federal statutes = 2
        if 'u.s.c.' in citation_text:
            return 2
        
        # State supreme court cases = 2
        if 'n.j.' in citation_text and 'super' not in citation_text:
            return 2
        
        # State statutes = 3  
        if 'n.j.s.a.' in citation_text:
            return 3
        
        # Lower court cases = 4
        if citation.citation_type == CitationType.CASE:
            return 4
        
        # Secondary sources = 5
        if citation.citation_type == CitationType.SECONDARY:
            return 5
        
        return 4  # Default
    
    def _extract_citation_context(self, content: str, citation_text: str) -> List[str]:
        """Extract context around citation appearances"""
        
        contexts = []
        
        # Find all occurrences of the citation
        for match in re.finditer(re.escape(citation_text), content, re.IGNORECASE):
            start = max(0, match.start() - 200)
            end = min(len(content), match.end() + 200)
            context = content[start:end].strip()
            
            # Clean up context
            context = re.sub(r'\s+', ' ', context)
            contexts.append(context)
        
        return contexts[:3]  # Limit to first 3 contexts
    
    async def _normalize_citations(self, 
                                 citations: List[NormalizedCitation],
                                 target_format: CitationFormat) -> List[NormalizedCitation]:
        """Normalize citation formats for consistency"""
        
        normalized = []
        
        for citation in citations:
            try:
                normalized_text = await self._normalize_single_citation(citation, target_format)
                citation.normalized_text = normalized_text
                citation.format_style = target_format
                normalized.append(citation)
            except Exception as e:
                logger.warning(f"Failed to normalize citation '{citation.original_text}': {e}")
                # Keep original if normalization fails
                normalized.append(citation)
        
        return normalized
    
    async def _normalize_single_citation(self, 
                                       citation: NormalizedCitation, 
                                       target_format: CitationFormat) -> str:
        """Normalize a single citation to target format"""
        
        if target_format == CitationFormat.NEW_JERSEY:
            return await self._normalize_to_nj_format(citation)
        elif target_format == CitationFormat.BLUEBOOK:
            return await self._normalize_to_bluebook_format(citation)
        else:
            return citation.original_text  # No normalization
    
    async def _normalize_to_nj_format(self, citation: NormalizedCitation) -> str:
        """Normalize citation to New Jersey court format"""
        
        # Use LLM for sophisticated normalization
        prompt = f"""
        Normalize this legal citation to proper New Jersey court format:
        
        Original citation: {citation.original_text}
        Citation type: {citation.citation_type.value}
        
        Follow these New Jersey citation rules:
        1. Case citations: Use proper N.J. or N.J. Super. format
        2. Statute citations: Use N.J.S.A. format with proper spacing
        3. Include parenthetical years where appropriate
        4. Maintain proper punctuation and spacing
        
        Return only the normalized citation text.
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM normalization failed: {e}")
            return citation.original_text
    
    async def _normalize_to_bluebook_format(self, citation: NormalizedCitation) -> str:
        """Normalize citation to Bluebook format"""
        
        # Simplified Bluebook normalization
        # In practice, this would be much more sophisticated
        
        text = citation.original_text
        
        # Basic Bluebook rules
        text = re.sub(r'N\.J\.S\.A\.', 'N.J. Stat. Ann.', text)
        text = re.sub(r'N\.J\.A\.C\.', 'N.J. Admin. Code', text)
        
        return text
    
    async def _identify_duplicates(self, citations: List[NormalizedCitation]) -> List[CitationMatch]:
        """Identify duplicate and similar citations"""
        
        duplicate_pairs = []
        
        for i, citation1 in enumerate(citations):
            for j, citation2 in enumerate(citations[i + 1:], i + 1):
                match = await self._compare_citations(citation1, citation2, i, j)
                if match:
                    duplicate_pairs.append(match)
        
        return duplicate_pairs
    
    async def _compare_citations(self, 
                               citation1: NormalizedCitation, 
                               citation2: NormalizedCitation,
                               idx1: int, 
                               idx2: int) -> Optional[CitationMatch]:
        """Compare two citations for similarity"""
        
        # Exact match
        if citation1.normalized_text.strip() == citation2.normalized_text.strip():
            return CitationMatch(
                citation1_id=f"cite_{idx1}",
                citation2_id=f"cite_{idx2}",
                similarity_score=1.0,
                match_type="exact",
                differences=[]
            )
        
        # Substantial similarity using string similarity
        similarity = difflib.SequenceMatcher(
            None, 
            citation1.normalized_text.lower(), 
            citation2.normalized_text.lower()
        ).ratio()
        
        if similarity > 0.8:
            differences = self._identify_citation_differences(citation1, citation2)
            return CitationMatch(
                citation1_id=f"cite_{idx1}",
                citation2_id=f"cite_{idx2}",
                similarity_score=similarity,
                match_type="substantial" if similarity > 0.9 else "partial",
                differences=differences
            )
        
        return None
    
    def _identify_citation_differences(self, 
                                     citation1: NormalizedCitation, 
                                     citation2: NormalizedCitation) -> List[str]:
        """Identify specific differences between similar citations"""
        
        differences = []
        
        # Compare years
        if citation1.year and citation2.year and citation1.year != citation2.year:
            differences.append(f"Different years: {citation1.year} vs {citation2.year}")
        
        # Compare page numbers (if we extracted them)
        text1_pages = re.search(r'\bat\s+(\d+)', citation1.original_text)
        text2_pages = re.search(r'\bat\s+(\d+)', citation2.original_text)
        
        if text1_pages and text2_pages and text1_pages.group(1) != text2_pages.group(1):
            differences.append(f"Different page numbers: {text1_pages.group(1)} vs {text2_pages.group(1)}")
        
        # Compare parentheticals
        text1_paren = re.search(r'\([^)]+\)$', citation1.original_text)
        text2_paren = re.search(r'\([^)]+\)$', citation2.original_text)
        
        if text1_paren and not text2_paren:
            differences.append("Additional parenthetical in first citation")
        elif text2_paren and not text1_paren:
            differences.append("Additional parenthetical in second citation")
        elif text1_paren and text2_paren and text1_paren.group() != text2_paren.group():
            differences.append("Different parentheticals")
        
        return differences
    
    async def _create_citation_clusters(self, 
                                      citations: List[NormalizedCitation],
                                      duplicate_pairs: List[CitationMatch]) -> List[CitationCluster]:
        """Create clusters of related citations"""
        
        # Build clusters based on duplicate relationships
        citation_clusters = []
        processed_indices = set()
        
        # Create mapping of indices to citations
        idx_to_citation = {i: citation for i, citation in enumerate(citations)}
        
        for i, citation in enumerate(citations):
            if i in processed_indices:
                continue
            
            # Find all citations that match this one
            cluster_citations = [citation]
            cluster_indices = {i}
            
            for match in duplicate_pairs:
                cite1_idx = int(match.citation1_id.split('_')[1])
                cite2_idx = int(match.citation2_id.split('_')[1])
                
                if cite1_idx == i and cite2_idx not in processed_indices:
                    cluster_citations.append(idx_to_citation[cite2_idx])
                    cluster_indices.add(cite2_idx)
                elif cite2_idx == i and cite1_idx not in processed_indices:
                    cluster_citations.append(idx_to_citation[cite1_idx])
                    cluster_indices.add(cite1_idx)
            
            # Mark all citations in this cluster as processed
            processed_indices.update(cluster_indices)
            
            # Choose primary citation (highest authority)
            primary_citation = min(cluster_citations, key=lambda c: c.authority_level)
            alternative_citations = [c for c in cluster_citations if c != primary_citation]
            
            # Create cluster
            cluster = CitationCluster(
                cluster_id=f"cluster_{len(citation_clusters)}",
                primary_citation=primary_citation,
                alternative_citations=alternative_citations,
                citation_type=primary_citation.citation_type,
                authority_ranking=primary_citation.authority_level,
                usage_frequency=len(cluster_citations),
                contexts=self._aggregate_contexts(cluster_citations),
                consolidated_format=primary_citation.normalized_text
            )
            
            citation_clusters.append(cluster)
        
        # Sort clusters by authority ranking and usage frequency
        citation_clusters.sort(key=lambda c: (c.authority_ranking, -c.usage_frequency))
        
        return citation_clusters
    
    def _aggregate_contexts(self, citations: List[NormalizedCitation]) -> List[str]:
        """Aggregate contexts from multiple citations"""
        
        all_contexts = []
        for citation in citations:
            all_contexts.extend(citation.contexts)
        
        # Remove duplicates and limit
        unique_contexts = []
        for context in all_contexts:
            if context not in unique_contexts:
                unique_contexts.append(context)
        
        return unique_contexts[:5]  # Limit to top 5 contexts
    
    def _calculate_authority_rankings(self, citation_clusters: List[CitationCluster]) -> Dict[str, int]:
        """Calculate final authority rankings for citation clusters"""
        
        rankings = {}
        
        # Sort by authority level and usage frequency
        sorted_clusters = sorted(citation_clusters, key=lambda c: (c.authority_ranking, -c.usage_frequency))
        
        for rank, cluster in enumerate(sorted_clusters, 1):
            rankings[cluster.cluster_id] = rank
        
        return rankings
    
    def _calculate_deduplication_quality(self, 
                                       original_citations: List[NormalizedCitation],
                                       citation_clusters: List[CitationCluster],
                                       duplicate_pairs: List[CitationMatch]) -> Dict[str, float]:
        """Calculate quality metrics for deduplication process"""
        
        metrics = {}
        
        # Deduplication efficiency
        reduction_rate = 1.0 - (len(citation_clusters) / max(len(original_citations), 1))
        metrics['reduction_rate'] = reduction_rate
        
        # Format consistency
        formatted_citations = sum(1 for cluster in citation_clusters 
                                if cluster.primary_citation.format_style != CitationFormat.UNKNOWN)
        metrics['format_consistency'] = formatted_citations / max(len(citation_clusters), 1)
        
        # Preservation rate (no citations lost)
        total_clustered = sum(1 + len(cluster.alternative_citations) for cluster in citation_clusters)
        metrics['preservation_rate'] = total_clustered / max(len(original_citations), 1)
        
        # Authority preservation (high-authority citations preserved)
        high_authority_preserved = sum(1 for cluster in citation_clusters 
                                     if cluster.authority_ranking <= 2)
        high_authority_original = sum(1 for citation in original_citations 
                                    if citation.authority_level <= 2)
        metrics['authority_preservation'] = (high_authority_preserved / max(high_authority_original, 1) 
                                           if high_authority_original > 0 else 1.0)
        
        return metrics
    
    async def format_deduplicated_citations(self, 
                                          citation_clusters: List[CitationCluster],
                                          format_style: str = "paragraph") -> str:
        """Format deduplicated citations for inclusion in consolidated document"""
        
        if format_style == "paragraph":
            return self._format_citations_paragraph(citation_clusters)
        elif format_style == "bibliography":
            return self._format_citations_bibliography(citation_clusters)
        elif format_style == "footnotes":
            return self._format_citations_footnotes(citation_clusters)
        else:
            return self._format_citations_inline(citation_clusters)
    
    def _format_citations_paragraph(self, citation_clusters: List[CitationCluster]) -> str:
        """Format citations as integrated paragraph references"""
        
        # Group by authority level
        authority_groups = defaultdict(list)
        for cluster in citation_clusters:
            authority_groups[cluster.authority_ranking].append(cluster)
        
        formatted_sections = []
        
        for authority_level in sorted(authority_groups.keys()):
            clusters = authority_groups[authority_level]
            
            # Create section for this authority level
            level_name = {1: "Primary Authorities", 2: "Controlling Authorities", 
                         3: "Statutory Authorities", 4: "Supporting Authorities", 
                         5: "Secondary Authorities"}.get(authority_level, "Other Authorities")
            
            citations_text = "; ".join(cluster.consolidated_format for cluster in clusters)
            formatted_sections.append(f"**{level_name}:** {citations_text}")
        
        return "\n\n".join(formatted_sections)
    
    def _format_citations_bibliography(self, citation_clusters: List[CitationCluster]) -> str:
        """Format citations as bibliography"""
        
        formatted_citations = []
        
        for i, cluster in enumerate(citation_clusters, 1):
            formatted_citations.append(f"{i}. {cluster.consolidated_format}")
        
        return "\n".join(formatted_citations)
    
    def _format_citations_footnotes(self, citation_clusters: List[CitationCluster]) -> str:
        """Format citations for footnote reference"""
        
        footnotes = []
        
        for i, cluster in enumerate(citation_clusters, 1):
            footnotes.append(f"[^{i}]: {cluster.consolidated_format}")
        
        return "\n".join(footnotes)
    
    def _format_citations_inline(self, citation_clusters: List[CitationCluster]) -> str:
        """Format citations for inline reference"""
        
        return "; ".join(cluster.consolidated_format for cluster in citation_clusters[:20])  # Limit for inline