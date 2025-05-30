"""
Semantic Similarity Service for Legal Citation Resolution.
Uses OpenAI embeddings to disambiguate and match ambiguous citations based on semantic proximity.
"""
import logging
import re
import asyncio
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from db.models import Citation
from nlp.openai_service import openai_service

logger = logging.getLogger(__name__)


@dataclass
class SemanticMatch:
    """Represents a semantic similarity match between citations"""
    source_citation_id: str
    target_citation_id: str
    similarity_score: float
    context_overlap: float
    combined_confidence: float
    match_reason: str
    semantic_features: Dict[str, Any]


@dataclass
class CitationContext:
    """Extracted context information for a citation"""
    citation_id: str
    citation_text: str
    surrounding_text: str
    extracted_entities: List[str]
    legal_concepts: List[str]
    case_names: List[str]
    statutory_references: List[str]
    position_in_document: float  # 0.0 to 1.0


class SemanticSimilarityService:
    """
    Service for performing semantic similarity analysis on legal citations.
    
    Uses OpenAI embeddings to:
    1. Generate semantic representations of citations and their context
    2. Calculate similarity scores between citations
    3. Identify semantic relationships beyond pattern matching
    4. Provide confidence scoring for ambiguous matches
    """
    
    def __init__(self):
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.context_cache: Dict[str, CitationContext] = {}
        self.cache_expiry = timedelta(hours=24)
        self.last_cache_cleanup = datetime.utcnow()
        
        # Legal concept patterns for enhanced matching
        self.legal_concept_patterns = {
            'procedural': [
                r'\b(motion|hearing|trial|appeal|jurisdiction|venue|standing)\b',
                r'\b(pleading|discovery|summary judgment|dismissal)\b',
                r'\b(res judicata|collateral estoppel|statute of limitations)\b'
            ],
            'constitutional': [
                r'\b(due process|equal protection|first amendment|fourth amendment)\b',
                r'\b(constitutional|unconstitutional|bill of rights)\b',
                r'\b(substantive due process|procedural due process)\b'
            ],
            'contract': [
                r'\b(contract|agreement|breach|damages|consideration)\b',
                r'\b(offer|acceptance|performance|breach|remedy)\b',
                r'\b(material breach|anticipatory repudiation)\b'
            ],
            'tort': [
                r'\b(negligence|duty|breach|causation|damages)\b',
                r'\b(intentional tort|strict liability|defamation)\b',
                r'\b(proximate cause|reasonable person|standard of care)\b'
            ],
            'criminal': [
                r'\b(guilty|conviction|sentence|probation|parole)\b',
                r'\b(mens rea|actus reus|intent|premeditation)\b',
                r'\b(beyond reasonable doubt|burden of proof)\b'
            ]
        }
    
    async def extract_citation_context(
        self, 
        citation: Citation, 
        document_text: str,
        context_window: int = 500
    ) -> CitationContext:
        """
        Extract semantic context around a citation for similarity analysis
        
        Args:
            citation: Citation object to analyze
            document_text: Full document text
            context_window: Number of characters around citation to extract
        """
        try:
            citation_id = str(citation.id)
            
            # Check cache first
            if citation_id in self.context_cache:
                cached_context = self.context_cache[citation_id]
                # Verify cache freshness
                if (datetime.utcnow() - self.last_cache_cleanup) < self.cache_expiry:
                    return cached_context
            
            # Extract surrounding text context
            start_pos = max(0, citation.position_start - context_window)
            end_pos = min(len(document_text), citation.position_end + context_window)
            surrounding_text = document_text[start_pos:end_pos]
            
            # Extract semantic features
            entities = self._extract_legal_entities(surrounding_text)
            concepts = self._extract_legal_concepts(surrounding_text)
            case_names = self._extract_case_names(surrounding_text)
            statutory_refs = self._extract_statutory_references(surrounding_text)
            
            # Calculate relative position in document
            position_ratio = citation.position_start / len(document_text) if document_text else 0.0
            
            context = CitationContext(
                citation_id=citation_id,
                citation_text=citation.citation_text,
                surrounding_text=surrounding_text,
                extracted_entities=entities,
                legal_concepts=concepts,
                case_names=case_names,
                statutory_references=statutory_refs,
                position_in_document=position_ratio
            )
            
            # Cache the result
            self.context_cache[citation_id] = context
            
            return context
            
        except Exception as e:
            logger.error(f"Error extracting context for citation {citation.id}: {e}")
            # Return minimal context
            return CitationContext(
                citation_id=str(citation.id),
                citation_text=citation.citation_text,
                surrounding_text=citation.citation_text,
                extracted_entities=[],
                legal_concepts=[],
                case_names=[],
                statutory_references=[],
                position_in_document=0.0
            )
    
    async def generate_citation_embedding(
        self, 
        citation_context: CitationContext,
        include_surrounding_context: bool = True
    ) -> np.ndarray:
        """
        Generate semantic embedding for a citation using OpenAI embeddings
        
        Args:
            citation_context: Extracted context for the citation
            include_surrounding_context: Whether to include surrounding text
        """
        try:
            # Check cache first
            cache_key = f"{citation_context.citation_id}_{include_surrounding_context}"
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
            
            # Prepare text for embedding
            if include_surrounding_context:
                # Combine citation with enhanced context
                embedding_text = self._prepare_embedding_text(citation_context)
            else:
                # Just the citation text
                embedding_text = citation_context.citation_text
            
            # Generate embedding using OpenAI service
            embedding = await openai_service.generate_embedding(embedding_text)
            
            # Convert to numpy array for efficient computation
            embedding_array = np.array(embedding, dtype=np.float32)
            
            # Cache the result
            self.embedding_cache[cache_key] = embedding_array
            
            return embedding_array
            
        except Exception as e:
            logger.error(f"Error generating embedding for citation {citation_context.citation_id}: {e}")
            # Return zero vector as fallback
            return np.zeros(1536, dtype=np.float32)  # OpenAI embedding dimension
    
    async def calculate_semantic_similarity(
        self,
        source_context: CitationContext,
        target_context: CitationContext,
        include_context: bool = True
    ) -> SemanticMatch:
        """
        Calculate semantic similarity between two citations
        
        Args:
            source_context: Context for source citation
            target_context: Context for target citation  
            include_context: Whether to include surrounding context in similarity
        """
        try:
            # Generate embeddings for both citations
            source_embedding = await self.generate_citation_embedding(
                source_context, include_context
            )
            target_embedding = await self.generate_citation_embedding(
                target_context, include_context
            )
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(
                source_embedding.reshape(1, -1),
                target_embedding.reshape(1, -1)
            )
            semantic_score = float(similarity_matrix[0, 0])
            
            # Calculate context overlap score
            context_score = self._calculate_context_overlap(source_context, target_context)
            
            # Determine match reason based on similarities
            match_reason = self._determine_match_reason(
                source_context, target_context, semantic_score, context_score
            )
            
            # Calculate combined confidence score
            combined_confidence = self._calculate_combined_confidence(
                semantic_score, context_score, source_context, target_context
            )
            
            # Extract semantic features for debugging/analysis
            semantic_features = {
                "semantic_similarity": semantic_score,
                "context_overlap": context_score,
                "shared_concepts": list(set(source_context.legal_concepts) & set(target_context.legal_concepts)),
                "shared_entities": list(set(source_context.extracted_entities) & set(target_context.extracted_entities)),
                "shared_case_names": list(set(source_context.case_names) & set(target_context.case_names)),
                "position_distance": abs(source_context.position_in_document - target_context.position_in_document)
            }
            
            return SemanticMatch(
                source_citation_id=source_context.citation_id,
                target_citation_id=target_context.citation_id,
                similarity_score=semantic_score,
                context_overlap=context_score,
                combined_confidence=combined_confidence,
                match_reason=match_reason,
                semantic_features=semantic_features
            )
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            # Return low-confidence match
            return SemanticMatch(
                source_citation_id=source_context.citation_id,
                target_citation_id=target_context.citation_id,
                similarity_score=0.0,
                context_overlap=0.0,
                combined_confidence=0.0,
                match_reason="error_fallback",
                semantic_features={"error": str(e)}
            )
    
    async def find_semantic_matches(
        self,
        source_citation: Citation,
        candidate_citations: List[Citation],
        document_text: str,
        threshold: float = 0.7,
        max_matches: int = 5
    ) -> List[SemanticMatch]:
        """
        Find the best semantic matches for a source citation among candidates
        
        Args:
            source_citation: Citation to find matches for
            candidate_citations: List of potential target citations
            document_text: Full document text for context extraction
            threshold: Minimum similarity threshold
            max_matches: Maximum number of matches to return
        """
        try:
            # Extract context for source citation
            source_context = await self.extract_citation_context(
                source_citation, document_text
            )
            
            matches = []
            
            # Calculate similarity with each candidate
            for candidate in candidate_citations:
                if str(candidate.id) == str(source_citation.id):
                    continue  # Skip self-matching
                
                # Extract context for candidate
                candidate_context = await self.extract_citation_context(
                    candidate, document_text
                )
                
                # Calculate semantic similarity
                semantic_match = await self.calculate_semantic_similarity(
                    source_context, candidate_context
                )
                
                # Only include matches above threshold
                if semantic_match.combined_confidence >= threshold:
                    matches.append(semantic_match)
            
            # Sort by combined confidence score (descending)
            matches.sort(key=lambda x: x.combined_confidence, reverse=True)
            
            # Return top matches
            return matches[:max_matches]
            
        except Exception as e:
            logger.error(f"Error finding semantic matches: {e}")
            return []
    
    def _prepare_embedding_text(self, context: CitationContext) -> str:
        """Prepare enhanced text for embedding generation"""
        try:
            components = [
                # Primary citation text
                context.citation_text,
                
                # Key legal concepts (weighted heavily)
                " ".join(context.legal_concepts),
                
                # Case names (important for legal citations)
                " ".join(context.case_names),
                
                # Extracted entities
                " ".join(context.extracted_entities),
                
                # Surrounding context (limited to avoid noise)
                context.surrounding_text[:200] if context.surrounding_text else ""
            ]
            
            # Combine non-empty components
            enhanced_text = " ".join([comp for comp in components if comp.strip()])
            
            # Clean and normalize
            enhanced_text = re.sub(r'\s+', ' ', enhanced_text).strip()
            
            return enhanced_text
            
        except Exception as e:
            logger.error(f"Error preparing embedding text: {e}")
            return context.citation_text
    
    def _extract_legal_entities(self, text: str) -> List[str]:
        """Extract legal entities from text"""
        entities = []
        
        # Court names
        court_patterns = [
            r'\b(?:Supreme Court|Court of Appeals|District Court|Circuit Court)\b',
            r'\b(?:U\.S\.|United States|State of \w+)\b',
        ]
        
        # Legal parties
        party_patterns = [
            r'\b([A-Z][a-z]+ v\.? [A-Z][a-z]+)\b',
            r'\b([A-Z][a-z]+ Corp\.?|[A-Z][a-z]+ Inc\.?)\b',
        ]
        
        all_patterns = court_patterns + party_patterns
        
        for pattern in all_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend([match if isinstance(match, str) else match[0] for match in matches])
        
        return list(set(entities))
    
    def _extract_legal_concepts(self, text: str) -> List[str]:
        """Extract legal concepts from text using pattern matching"""
        concepts = []
        
        for concept_type, patterns in self.legal_concept_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                concepts.extend(matches)
        
        return list(set(concepts))
    
    def _extract_case_names(self, text: str) -> List[str]:
        """Extract case names from text"""
        # Common case name patterns
        patterns = [
            r'\b([A-Z][a-z]+ v\.? [A-Z][a-z]+)\b',
            r'\b([A-Z][a-z]+ v\.? [A-Z][a-z]+ (?:Corp|Inc|LLC)\.?)\b',
            r'\b((?:In re|Ex parte) [A-Z][a-z]+)\b',
        ]
        
        case_names = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            case_names.extend(matches)
        
        return list(set(case_names))
    
    def _extract_statutory_references(self, text: str) -> List[str]:
        """Extract statutory references from text"""
        patterns = [
            r'\b(\d+\s+U\.S\.C\.?\s+§?\s*\d+)\b',
            r'\b(\d+\s+C\.F\.R\.?\s+§?\s*\d+)\b',
            r'\b(§\s*\d+(?:\.\d+)*)\b',
        ]
        
        references = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            references.extend(matches)
        
        return list(set(references))
    
    def _calculate_context_overlap(
        self, 
        source_context: CitationContext, 
        target_context: CitationContext
    ) -> float:
        """Calculate overlap score between citation contexts"""
        try:
            scores = []
            
            # Legal concepts overlap
            source_concepts = set(source_context.legal_concepts)
            target_concepts = set(target_context.legal_concepts)
            if source_concepts or target_concepts:
                concept_overlap = len(source_concepts & target_concepts) / len(source_concepts | target_concepts)
                scores.append(concept_overlap * 0.4)  # Weight: 40%
            
            # Case names overlap  
            source_cases = set(source_context.case_names)
            target_cases = set(target_context.case_names)
            if source_cases or target_cases:
                case_overlap = len(source_cases & target_cases) / len(source_cases | target_cases)
                scores.append(case_overlap * 0.3)  # Weight: 30%
            
            # Entity overlap
            source_entities = set(source_context.extracted_entities)
            target_entities = set(target_context.extracted_entities)
            if source_entities or target_entities:
                entity_overlap = len(source_entities & target_entities) / len(source_entities | target_entities)
                scores.append(entity_overlap * 0.2)  # Weight: 20%
            
            # Position proximity (closer citations are more likely related)
            position_distance = abs(source_context.position_in_document - target_context.position_in_document)
            position_score = max(0, 1.0 - position_distance * 2)  # Penalty for distance
            scores.append(position_score * 0.1)  # Weight: 10%
            
            return sum(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating context overlap: {e}")
            return 0.0
    
    def _determine_match_reason(
        self,
        source_context: CitationContext,
        target_context: CitationContext, 
        semantic_score: float,
        context_score: float
    ) -> str:
        """Determine the primary reason for the match"""
        try:
            if semantic_score > 0.9:
                return "high_semantic_similarity"
            elif context_score > 0.8:
                return "strong_context_overlap"
            elif len(set(source_context.case_names) & set(target_context.case_names)) > 0:
                return "shared_case_references"
            elif len(set(source_context.legal_concepts) & set(target_context.legal_concepts)) > 2:
                return "shared_legal_concepts"
            elif abs(source_context.position_in_document - target_context.position_in_document) < 0.1:
                return "proximity_in_document"
            elif semantic_score > 0.7:
                return "moderate_semantic_similarity"
            else:
                return "weak_similarity"
                
        except Exception as e:
            logger.error(f"Error determining match reason: {e}")
            return "unknown"
    
    def _calculate_combined_confidence(
        self,
        semantic_score: float,
        context_score: float,
        source_context: CitationContext,
        target_context: CitationContext
    ) -> float:
        """Calculate combined confidence score using multiple signals"""
        try:
            # Base scores with weights
            weighted_semantic = semantic_score * 0.6  # 60% weight on semantic similarity
            weighted_context = context_score * 0.3   # 30% weight on context overlap
            
            # Bonus factors
            bonus = 0.0
            
            # Case name match bonus
            if set(source_context.case_names) & set(target_context.case_names):
                bonus += 0.1
            
            # Same document section bonus
            if abs(source_context.position_in_document - target_context.position_in_document) < 0.05:
                bonus += 0.05
            
            # Multiple legal concept matches bonus
            shared_concepts = len(set(source_context.legal_concepts) & set(target_context.legal_concepts))
            if shared_concepts > 2:
                bonus += min(0.05, shared_concepts * 0.01)
            
            # Combine scores
            combined = weighted_semantic + weighted_context + bonus
            
            # Ensure score is in valid range [0, 1]
            return max(0.0, min(1.0, combined))
            
        except Exception as e:
            logger.error(f"Error calculating combined confidence: {e}")
            return semantic_score * 0.8  # Fallback to reduced semantic score
    
    def cleanup_cache(self):
        """Clean up expired cache entries"""
        try:
            current_time = datetime.utcnow()
            if (current_time - self.last_cache_cleanup) > self.cache_expiry:
                self.embedding_cache.clear()
                self.context_cache.clear()
                self.last_cache_cleanup = current_time
                logger.info("Cleaned up semantic similarity cache")
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")


# Global service instance
semantic_similarity_service = SemanticSimilarityService() 