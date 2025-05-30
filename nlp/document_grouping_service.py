"""
Document Grouping Service for organizing legal memoranda for consolidation.

This service provides sophisticated document grouping and organization capabilities
to support the CRRACC consolidation methodology by:
- Grouping documents by legal theory rather than individual provisions
- Analyzing semantic similarity between documents
- Organizing by argument strength and strategic priority
- Supporting multiple grouping strategies for different consolidation needs
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from nlp.semantic_similarity import SemanticSimilarityService
from nlp.citation_service import CitationService

logger = logging.getLogger(__name__)


class GroupingStrategy(Enum):
    """Different strategies for grouping documents"""
    LEGAL_THEORY = "legal_theory"  # Group by underlying legal theories
    STATUTE_TYPE = "statute_type"  # Group by statutes being violated
    VIOLATION_PATTERN = "violation_pattern"  # Group by type of violation
    FACTUAL_SIMILARITY = "factual_similarity"  # Group by factual patterns
    HYBRID = "hybrid"  # Combination approach


@dataclass
class DocumentCluster:
    """Represents a cluster of related documents"""
    cluster_id: str
    cluster_name: str
    primary_theory: str
    documents: List[Dict[str, Any]]
    similarity_score: float
    strength_score: float
    priority_rank: int
    supporting_citations: List[str]
    key_provisions: List[str]
    factual_patterns: List[str]


@dataclass
class GroupingResult:
    """Result of document grouping analysis"""
    strategy: GroupingStrategy
    clusters: List[DocumentCluster]
    ungrouped_documents: List[Dict[str, Any]]
    quality_metrics: Dict[str, float]
    recommended_consolidation_order: List[str]  # cluster_ids in order


class DocumentGroupingService:
    """
    Service for intelligent grouping and organization of legal documents
    to support systematic consolidation using the CRRACC method.
    """
    
    def __init__(self):
        self.semantic_service = SemanticSimilarityService()
        self.citation_service = CitationService()
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            max_tokens=3000
        )
        
    async def group_documents(self, documents: List[Dict[str, Any]], 
                            strategy: GroupingStrategy = GroupingStrategy.LEGAL_THEORY,
                            min_cluster_size: int = 2,
                            similarity_threshold: float = 0.7) -> GroupingResult:
        """
        Group documents using the specified strategy.
        
        Args:
            documents: List of document dictionaries with content, metadata, citations
            strategy: Grouping strategy to use
            min_cluster_size: Minimum documents required to form a cluster
            similarity_threshold: Similarity threshold for grouping
            
        Returns:
            GroupingResult with organized clusters and metadata
        """
        logger.info(f"Grouping {len(documents)} documents using {strategy.value} strategy")
        
        # Apply the selected grouping strategy
        if strategy == GroupingStrategy.LEGAL_THEORY:
            return await self._group_by_legal_theory(documents, min_cluster_size, similarity_threshold)
        elif strategy == GroupingStrategy.STATUTE_TYPE:
            return await self._group_by_statute_type(documents, min_cluster_size, similarity_threshold)
        elif strategy == GroupingStrategy.VIOLATION_PATTERN:
            return await self._group_by_violation_pattern(documents, min_cluster_size, similarity_threshold)
        elif strategy == GroupingStrategy.FACTUAL_SIMILARITY:
            return await self._group_by_factual_similarity(documents, min_cluster_size, similarity_threshold)
        elif strategy == GroupingStrategy.HYBRID:
            return await self._group_hybrid(documents, min_cluster_size, similarity_threshold)
        else:
            raise ValueError(f"Unsupported grouping strategy: {strategy}")
    
    async def _group_by_legal_theory(self, documents: List[Dict[str, Any]], 
                                   min_cluster_size: int, 
                                   similarity_threshold: float) -> GroupingResult:
        """Group documents by underlying legal theories they advance"""
        
        # Step 1: Extract legal theories from each document
        document_theories = []
        for doc_idx, doc in enumerate(documents):
            theories = await self._extract_document_theories(doc, doc_idx)
            document_theories.append(theories)
        
        # Step 2: Create theory-based clusters
        theory_clusters = defaultdict(list)
        theory_metadata = defaultdict(dict)
        
        for doc_idx, theories in enumerate(document_theories):
            doc = documents[doc_idx]
            doc['_index'] = doc_idx
            
            for theory_name, theory_data in theories.items():
                theory_clusters[theory_name].append(doc)
                
                # Aggregate theory metadata
                if theory_name not in theory_metadata:
                    theory_metadata[theory_name] = {
                        'citations': set(),
                        'provisions': set(),
                        'strength_scores': [],
                        'factual_patterns': set()
                    }
                
                theory_metadata[theory_name]['citations'].update(theory_data.get('citations', []))
                theory_metadata[theory_name]['provisions'].update(theory_data.get('provisions', []))
                theory_metadata[theory_name]['strength_scores'].append(theory_data.get('strength', 5.0))
                theory_metadata[theory_name]['factual_patterns'].update(theory_data.get('factual_patterns', []))
        
        # Step 3: Filter clusters by minimum size and create DocumentCluster objects
        clusters = []
        ungrouped_documents = []
        
        for theory_name, cluster_docs in theory_clusters.items():
            if len(cluster_docs) >= min_cluster_size:
                metadata = theory_metadata[theory_name]
                
                # Calculate semantic similarity within cluster
                cluster_similarity = await self._calculate_cluster_similarity(cluster_docs)
                
                # Calculate strength score
                strength_score = np.mean(metadata['strength_scores']) if metadata['strength_scores'] else 5.0
                
                cluster = DocumentCluster(
                    cluster_id=f"theory_{len(clusters)}",
                    cluster_name=theory_name,
                    primary_theory=theory_name,
                    documents=cluster_docs,
                    similarity_score=cluster_similarity,
                    strength_score=strength_score,
                    priority_rank=0,  # Will be set later
                    supporting_citations=list(metadata['citations']),
                    key_provisions=list(metadata['provisions']),
                    factual_patterns=list(metadata['factual_patterns'])
                )
                clusters.append(cluster)
            else:
                ungrouped_documents.extend(cluster_docs)
        
        # Step 4: Rank clusters by priority
        clusters = self._rank_clusters_by_priority(clusters)
        
        # Step 5: Calculate quality metrics
        quality_metrics = self._calculate_grouping_quality(clusters, ungrouped_documents, documents)
        
        # Step 6: Determine recommended consolidation order
        consolidation_order = [cluster.cluster_id for cluster in clusters]
        
        return GroupingResult(
            strategy=GroupingStrategy.LEGAL_THEORY,
            clusters=clusters,
            ungrouped_documents=ungrouped_documents,
            quality_metrics=quality_metrics,
            recommended_consolidation_order=consolidation_order
        )
    
    async def _extract_document_theories(self, document: Dict[str, Any], doc_idx: int) -> Dict[str, Dict[str, Any]]:
        """Extract legal theories from a single document using LLM analysis"""
        
        content = document.get('content', '')
        title = document.get('title', f'Document {doc_idx}')
        
        # Limit content length for LLM processing
        content_sample = content[:4000] if len(content) > 4000 else content
        
        prompt = f"""
        Analyze this legal memorandum and identify the distinct legal theories being argued.
        
        Document Title: {title}
        
        Content: {content_sample}
        
        For each legal theory, extract:
        1. Theory name (e.g., "Unconscionability", "Truth in Renting Act Violation", "Warranty of Habitability Breach")
        2. Supporting legal citations
        3. Related statutory provisions  
        4. Key factual patterns that support this theory
        5. Argument strength (1-10 scale)
        
        Focus on substantive legal theories, not procedural arguments.
        
        Format your response as:
        THEORY: [theory name]
        CITATIONS: [citation1], [citation2], [citation3]
        PROVISIONS: [provision1], [provision2]
        FACTUAL_PATTERNS: [pattern1], [pattern2]
        STRENGTH: [1-10]
        ---
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return self._parse_theory_extraction_response(response.content)
        except Exception as e:
            logger.error(f"Failed to extract theories from document {doc_idx}: {e}")
            return {}
    
    def _parse_theory_extraction_response(self, response: str) -> Dict[str, Dict[str, Any]]:
        """Parse LLM response into structured theory data"""
        theories = {}
        current_theory = None
        
        for line in response.split('\n'):
            line = line.strip()
            
            if line.startswith('THEORY:'):
                current_theory = line.replace('THEORY:', '').strip()
                theories[current_theory] = {
                    'citations': [],
                    'provisions': [],
                    'factual_patterns': [],
                    'strength': 5.0
                }
            elif line.startswith('CITATIONS:') and current_theory:
                citations_text = line.replace('CITATIONS:', '').strip()
                citations = [c.strip() for c in citations_text.split(',') if c.strip()]
                theories[current_theory]['citations'] = citations
            elif line.startswith('PROVISIONS:') and current_theory:
                provisions_text = line.replace('PROVISIONS:', '').strip()
                provisions = [p.strip() for p in provisions_text.split(',') if p.strip()]
                theories[current_theory]['provisions'] = provisions
            elif line.startswith('FACTUAL_PATTERNS:') and current_theory:
                patterns_text = line.replace('FACTUAL_PATTERNS:', '').strip()
                patterns = [p.strip() for p in patterns_text.split(',') if p.strip()]
                theories[current_theory]['factual_patterns'] = patterns
            elif line.startswith('STRENGTH:') and current_theory:
                try:
                    strength = float(line.replace('STRENGTH:', '').strip())
                    theories[current_theory]['strength'] = strength
                except ValueError:
                    pass
        
        return theories
    
    async def _group_by_statute_type(self, documents: List[Dict[str, Any]], 
                                   min_cluster_size: int, 
                                   similarity_threshold: float) -> GroupingResult:
        """Group documents by the types of statutes they address"""
        
        statute_clusters = defaultdict(list)
        
        for doc_idx, doc in enumerate(documents):
            doc['_index'] = doc_idx
            statutes = await self._identify_primary_statutes(doc)
            
            for statute in statutes:
                statute_clusters[statute].append(doc)
        
        # Convert to DocumentCluster objects
        clusters = []
        ungrouped_documents = []
        
        for statute_name, cluster_docs in statute_clusters.items():
            if len(cluster_docs) >= min_cluster_size:
                cluster_similarity = await self._calculate_cluster_similarity(cluster_docs)
                
                cluster = DocumentCluster(
                    cluster_id=f"statute_{len(clusters)}",
                    cluster_name=f"{statute_name} Violations",
                    primary_theory=statute_name,
                    documents=cluster_docs,
                    similarity_score=cluster_similarity,
                    strength_score=7.0,  # Default for statute-based grouping
                    priority_rank=0,
                    supporting_citations=[],
                    key_provisions=[statute_name],
                    factual_patterns=[]
                )
                clusters.append(cluster)
            else:
                ungrouped_documents.extend(cluster_docs)
        
        clusters = self._rank_clusters_by_priority(clusters)
        quality_metrics = self._calculate_grouping_quality(clusters, ungrouped_documents, documents)
        consolidation_order = [cluster.cluster_id for cluster in clusters]
        
        return GroupingResult(
            strategy=GroupingStrategy.STATUTE_TYPE,
            clusters=clusters,
            ungrouped_documents=ungrouped_documents,
            quality_metrics=quality_metrics,
            recommended_consolidation_order=consolidation_order
        )
    
    async def _identify_primary_statutes(self, document: Dict[str, Any]) -> List[str]:
        """Identify the primary statutes addressed in a document"""
        
        content = document.get('content', '')[:3000]  # Limit for efficiency
        
        # Common New Jersey tenant protection statutes
        statute_patterns = {
            'Truth in Renting Act': ['Truth in Renting', 'N.J.S.A. 46:8-48', '46:8-48'],
            'Anti-Eviction Act': ['Anti-Eviction', 'N.J.S.A. 2A:18-61', '2A:18-61'],
            'Foreclosure Fairness Act': ['Foreclosure Fairness', 'N.J.S.A. 2A:50-70', '2A:50-70'],
            'Unconscionability': ['N.J.S.A. 12A:2A-108', '12A:2A-108', 'unconscionable'],
            'Warranty of Habitability': ['warranty of habitability', 'Marini v. Ireland'],
            'Quiet Enjoyment': ['quiet enjoyment', 'Reste Realty']
        }
        
        identified_statutes = []
        
        for statute_name, patterns in statute_patterns.items():
            for pattern in patterns:
                if pattern.lower() in content.lower():
                    identified_statutes.append(statute_name)
                    break
        
        return identified_statutes if identified_statutes else ['General Tenant Rights']
    
    async def _group_by_violation_pattern(self, documents: List[Dict[str, Any]], 
                                        min_cluster_size: int, 
                                        similarity_threshold: float) -> GroupingResult:
        """Group documents by patterns of violations (procedural vs substantive, etc.)"""
        
        pattern_clusters = defaultdict(list)
        
        for doc_idx, doc in enumerate(documents):
            doc['_index'] = doc_idx
            violation_patterns = await self._identify_violation_patterns(doc)
            
            for pattern in violation_patterns:
                pattern_clusters[pattern].append(doc)
        
        # Convert to DocumentCluster objects
        clusters = []
        ungrouped_documents = []
        
        for pattern_name, cluster_docs in pattern_clusters.items():
            if len(cluster_docs) >= min_cluster_size:
                cluster_similarity = await self._calculate_cluster_similarity(cluster_docs)
                
                cluster = DocumentCluster(
                    cluster_id=f"pattern_{len(clusters)}",
                    cluster_name=f"{pattern_name} Violations",
                    primary_theory=pattern_name,
                    documents=cluster_docs,
                    similarity_score=cluster_similarity,
                    strength_score=6.5,
                    priority_rank=0,
                    supporting_citations=[],
                    key_provisions=[],
                    factual_patterns=[pattern_name]
                )
                clusters.append(cluster)
            else:
                ungrouped_documents.extend(cluster_docs)
        
        clusters = self._rank_clusters_by_priority(clusters)
        quality_metrics = self._calculate_grouping_quality(clusters, ungrouped_documents, documents)
        consolidation_order = [cluster.cluster_id for cluster in clusters]
        
        return GroupingResult(
            strategy=GroupingStrategy.VIOLATION_PATTERN,
            clusters=clusters,
            ungrouped_documents=ungrouped_documents,
            quality_metrics=quality_metrics,
            recommended_consolidation_order=consolidation_order
        )
    
    async def _identify_violation_patterns(self, document: Dict[str, Any]) -> List[str]:
        """Identify patterns of violations in a document"""
        
        content = document.get('content', '')[:3000]
        
        violation_patterns = []
        
        # Pattern detection based on content analysis
        if any(term in content.lower() for term in ['waiver', 'waives', 'waived']):
            violation_patterns.append('Rights Waiver')
        
        if any(term in content.lower() for term in ['unconscionable', 'adhesion', 'take it or leave it']):
            violation_patterns.append('Unconscionable Terms')
        
        if any(term in content.lower() for term in ['subordination', 'attorney-in-fact', 'foreclosure']):
            violation_patterns.append('Circumvention of Protections')
        
        if any(term in content.lower() for term in ['habitability', 'conditions', 'repairs']):
            violation_patterns.append('Habitability Violations')
        
        if any(term in content.lower() for term in ['jury trial', 'counterclaim', 'procedural']):
            violation_patterns.append('Procedural Rights Violations')
        
        return violation_patterns if violation_patterns else ['General Violations']
    
    async def _group_by_factual_similarity(self, documents: List[Dict[str, Any]], 
                                         min_cluster_size: int, 
                                         similarity_threshold: float) -> GroupingResult:
        """Group documents by factual similarity using semantic analysis"""
        
        # Extract factual content from each document
        factual_contents = []
        for doc in documents:
            factual_content = await self._extract_factual_content(doc)
            factual_contents.append(factual_content)
        
        # Use semantic similarity to cluster documents
        similarity_matrix = await self._build_similarity_matrix(factual_contents)
        clusters_indices = self._cluster_by_similarity(similarity_matrix, similarity_threshold, min_cluster_size)
        
        # Convert to DocumentCluster objects
        clusters = []
        ungrouped_documents = []
        
        for cluster_idx, doc_indices in enumerate(clusters_indices):
            if len(doc_indices) >= min_cluster_size:
                cluster_docs = [documents[i] for i in doc_indices]
                for doc in cluster_docs:
                    doc['_index'] = doc_indices[cluster_docs.index(doc)]
                
                cluster_similarity = np.mean([
                    similarity_matrix[i][j] for i in doc_indices for j in doc_indices if i != j
                ]) if len(doc_indices) > 1 else 1.0
                
                cluster = DocumentCluster(
                    cluster_id=f"factual_{cluster_idx}",
                    cluster_name=f"Factual Pattern Cluster {cluster_idx + 1}",
                    primary_theory="Factual Similarity",
                    documents=cluster_docs,
                    similarity_score=cluster_similarity,
                    strength_score=6.0,
                    priority_rank=0,
                    supporting_citations=[],
                    key_provisions=[],
                    factual_patterns=[f"Pattern {cluster_idx + 1}"]
                )
                clusters.append(cluster)
            else:
                ungrouped_documents.extend([documents[i] for i in doc_indices])
        
        clusters = self._rank_clusters_by_priority(clusters)
        quality_metrics = self._calculate_grouping_quality(clusters, ungrouped_documents, documents)
        consolidation_order = [cluster.cluster_id for cluster in clusters]
        
        return GroupingResult(
            strategy=GroupingStrategy.FACTUAL_SIMILARITY,
            clusters=clusters,
            ungrouped_documents=ungrouped_documents,
            quality_metrics=quality_metrics,
            recommended_consolidation_order=consolidation_order
        )
    
    async def _extract_factual_content(self, document: Dict[str, Any]) -> str:
        """Extract factual content from a document for similarity analysis"""
        
        content = document.get('content', '')
        
        # Look for common factual sections
        factual_sections = []
        
        # Split into sections and identify factual content
        sections = content.split('\n\n')
        for section in sections:
            section_lower = section.lower()
            
            # Skip pure legal analysis sections
            if any(term in section_lower for term in [
                'legal argument', 'questions presented', 'relief requested',
                'conclusion', 'memorandum of law', 'respectfully submitted'
            ]):
                continue
            
            # Include factual sections
            if any(term in section_lower for term in [
                'statement of facts', 'factual background', 'lease provision',
                'the provision', 'the clause', 'tenant', 'landlord', 'premises'
            ]):
                factual_sections.append(section)
        
        return '\n\n'.join(factual_sections[:3])  # Limit to first 3 factual sections
    
    async def _build_similarity_matrix(self, contents: List[str]) -> np.ndarray:
        """Build semantic similarity matrix for document contents"""
        
        n_docs = len(contents)
        similarity_matrix = np.eye(n_docs)  # Identity matrix as base
        
        # Calculate pairwise similarities
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                if contents[i] and contents[j]:  # Skip empty contents
                    try:
                        similarity = await self.semantic_service.calculate_similarity(contents[i], contents[j])
                        similarity_matrix[i][j] = similarity
                        similarity_matrix[j][i] = similarity
                    except Exception as e:
                        logger.warning(f"Failed to calculate similarity between docs {i} and {j}: {e}")
                        similarity_matrix[i][j] = 0.0
                        similarity_matrix[j][i] = 0.0
        
        return similarity_matrix
    
    def _cluster_by_similarity(self, similarity_matrix: np.ndarray, 
                             threshold: float, min_cluster_size: int) -> List[List[int]]:
        """Cluster documents based on similarity matrix using simple threshold clustering"""
        
        n_docs = similarity_matrix.shape[0]
        visited = set()
        clusters = []
        
        for i in range(n_docs):
            if i in visited:
                continue
            
            # Start new cluster
            cluster = [i]
            visited.add(i)
            
            # Add similar documents to cluster
            for j in range(n_docs):
                if j not in visited and similarity_matrix[i][j] >= threshold:
                    cluster.append(j)
                    visited.add(j)
            
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
            else:
                # Return individual documents as single-item clusters
                for doc_idx in cluster:
                    clusters.append([doc_idx])
        
        return clusters
    
    async def _group_hybrid(self, documents: List[Dict[str, Any]], 
                          min_cluster_size: int, 
                          similarity_threshold: float) -> GroupingResult:
        """Hybrid grouping combining multiple strategies"""
        
        # Get results from different strategies
        theory_result = await self._group_by_legal_theory(documents, min_cluster_size, similarity_threshold)
        statute_result = await self._group_by_statute_type(documents, min_cluster_size, similarity_threshold)
        
        # Combine and optimize clusters
        all_clusters = theory_result.clusters + statute_result.clusters
        
        # Remove duplicates and merge related clusters
        optimized_clusters = self._optimize_cluster_combination(all_clusters)
        
        # Determine ungrouped documents
        grouped_doc_indices = set()
        for cluster in optimized_clusters:
            for doc in cluster.documents:
                grouped_doc_indices.add(doc.get('_index', id(doc)))
        
        ungrouped_documents = [
            doc for i, doc in enumerate(documents) 
            if i not in grouped_doc_indices
        ]
        
        quality_metrics = self._calculate_grouping_quality(optimized_clusters, ungrouped_documents, documents)
        consolidation_order = [cluster.cluster_id for cluster in optimized_clusters]
        
        return GroupingResult(
            strategy=GroupingStrategy.HYBRID,
            clusters=optimized_clusters,
            ungrouped_documents=ungrouped_documents,
            quality_metrics=quality_metrics,
            recommended_consolidation_order=consolidation_order
        )
    
    def _optimize_cluster_combination(self, clusters: List[DocumentCluster]) -> List[DocumentCluster]:
        """Optimize cluster combination by removing duplicates and merging similar clusters"""
        
        # Simple deduplication based on document overlap
        unique_clusters = []
        
        for cluster in clusters:
            doc_ids = {id(doc) for doc in cluster.documents}
            
            # Check if this cluster significantly overlaps with existing ones
            is_duplicate = False
            for existing in unique_clusters:
                existing_doc_ids = {id(doc) for doc in existing.documents}
                overlap = len(doc_ids & existing_doc_ids) / len(doc_ids | existing_doc_ids)
                
                if overlap > 0.5:  # More than 50% overlap
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_clusters.append(cluster)
        
        return self._rank_clusters_by_priority(unique_clusters)
    
    async def _calculate_cluster_similarity(self, cluster_docs: List[Dict[str, Any]]) -> float:
        """Calculate average internal similarity within a cluster"""
        
        if len(cluster_docs) <= 1:
            return 1.0
        
        similarities = []
        
        for i, doc1 in enumerate(cluster_docs):
            for j, doc2 in enumerate(cluster_docs[i + 1:], i + 1):
                content1 = doc1.get('content', '')[:2000]
                content2 = doc2.get('content', '')[:2000]
                
                if content1 and content2:
                    try:
                        sim = await self.semantic_service.calculate_similarity(content1, content2)
                        similarities.append(sim)
                    except Exception as e:
                        logger.warning(f"Failed to calculate similarity within cluster: {e}")
        
        return np.mean(similarities) if similarities else 0.5
    
    def _rank_clusters_by_priority(self, clusters: List[DocumentCluster]) -> List[DocumentCluster]:
        """Rank clusters by priority for consolidation"""
        
        # Priority ranking criteria:
        # 1. Strength score (higher is better)
        # 2. Number of documents (more is better)
        # 3. Similarity score (higher is better)
        # 4. Number of citations (more is better)
        
        for i, cluster in enumerate(clusters):
            priority_score = (
                cluster.strength_score * 0.4 +
                len(cluster.documents) * 0.3 +
                cluster.similarity_score * 0.2 +
                len(cluster.supporting_citations) * 0.1
            )
            cluster.priority_rank = i
        
        # Sort by priority score (descending)
        sorted_clusters = sorted(clusters, key=lambda c: (
            c.strength_score * 0.4 +
            len(c.documents) * 0.3 +
            c.similarity_score * 0.2 +
            len(c.supporting_citations) * 0.1
        ), reverse=True)
        
        # Update ranks
        for i, cluster in enumerate(sorted_clusters):
            cluster.priority_rank = i + 1
        
        return sorted_clusters
    
    def _calculate_grouping_quality(self, clusters: List[DocumentCluster], 
                                  ungrouped_documents: List[Dict[str, Any]], 
                                  all_documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate quality metrics for the grouping result"""
        
        total_docs = len(all_documents)
        grouped_docs = sum(len(cluster.documents) for cluster in clusters)
        ungrouped_count = len(ungrouped_documents)
        
        metrics = {}
        
        # Coverage: percentage of documents successfully grouped
        metrics['coverage'] = grouped_docs / total_docs if total_docs > 0 else 0.0
        
        # Cluster quality: average similarity within clusters
        if clusters:
            cluster_similarities = [cluster.similarity_score for cluster in clusters]
            metrics['cluster_cohesion'] = np.mean(cluster_similarities)
        else:
            metrics['cluster_cohesion'] = 0.0
        
        # Distribution balance: how evenly documents are distributed across clusters
        if clusters:
            cluster_sizes = [len(cluster.documents) for cluster in clusters]
            size_variance = np.var(cluster_sizes) / np.mean(cluster_sizes) if cluster_sizes else 0
            metrics['distribution_balance'] = max(0.0, 1.0 - size_variance)
        else:
            metrics['distribution_balance'] = 0.0
        
        # Overall quality
        metrics['overall_quality'] = (
            metrics['coverage'] * 0.5 +
            metrics['cluster_cohesion'] * 0.3 +
            metrics['distribution_balance'] * 0.2
        )
        
        return metrics