"""
Citation extraction service using eyecite for legal document analysis.
Provides high-performance citation extraction, resolution, and analysis.
"""
import logging
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from eyecite import get_citations, resolve_citations
from eyecite.models import (
    FullCaseCitation, ShortCaseCitation, SupraCitation, 
    IdCitation, CitationBase as EyeciteCitation
)

from db.models import Citation, Document, CitationEmbedding, CitationRelationship
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class CitationExtractionResult:
    """Result of citation extraction with metadata"""
    citations: List[Citation]
    extraction_stats: Dict[str, Any]
    errors: List[str]
    processing_time_ms: int


@dataclass
class CitationAnalysis:
    """Analysis result for a citation's precedential authority"""
    citation: Citation
    court_level: Optional[str] = None
    jurisdiction: Optional[str] = None
    precedential_strength: Optional[str] = None
    authority_score: Optional[float] = None
    confidence_score: Optional[float] = None


class CitationExtractionService:
    """Service for extracting and analyzing legal citations from documents"""
    
    def __init__(self):
        self.extraction_stats = {
            "total_processed": 0,
            "total_citations": 0,
            "total_errors": 0,
            "processing_time_ms": 0
        }
    
    async def extract_citations_from_document(
        self, 
        document: Document, 
        db_session: AsyncSession,
        resolve_references: bool = True,
        create_embeddings: bool = False
    ) -> CitationExtractionResult:
        """
        Extract all citations from a document and store in database
        
        Args:
            document: Document to process
            db_session: Database session
            resolve_references: Whether to resolve supra/id citations
            create_embeddings: Whether to generate embeddings for citations
            
        Returns:
            CitationExtractionResult with extracted citations and metadata
        """
        start_time = datetime.now()
        errors = []
        
        try:
            # Extract citations using eyecite
            raw_citations = await self._extract_raw_citations(document.extracted_text)
            
            # Resolve references if requested
            if resolve_references:
                raw_citations = await self._resolve_citation_references(raw_citations)
            
            # Convert to database objects
            db_citations = []
            for raw_cite in raw_citations:
                try:
                    db_citation = await self._convert_to_db_citation(raw_cite, document.id)
                    db_citations.append(db_citation)
                except Exception as e:
                    errors.append(f"Failed to convert citation {raw_cite}: {e}")
                    logger.warning(f"Citation conversion error: {e}")
            
            # Save to database
            for citation in db_citations:
                db_session.add(citation)
            
            await db_session.commit()
            
            # Create embeddings if requested
            if create_embeddings:
                try:
                    await self._create_citation_embeddings(db_citations, db_session)
                except Exception as e:
                    errors.append(f"Failed to create embeddings: {e}")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update stats
            self.extraction_stats["total_processed"] += 1
            self.extraction_stats["total_citations"] += len(db_citations)
            self.extraction_stats["total_errors"] += len(errors)
            self.extraction_stats["processing_time_ms"] += processing_time
            
            # Create extraction stats
            extraction_stats = {
                "total_citations": len(db_citations),
                "citation_types": self._analyze_citation_types(raw_citations),
                "resolution_count": len([c for c in raw_citations if hasattr(c, 'antecedent_guess')]),
                "error_count": len(errors),
                "processing_time_ms": processing_time
            }
            
            return CitationExtractionResult(
                citations=db_citations,
                extraction_stats=extraction_stats,
                errors=errors,
                processing_time_ms=int(processing_time)
            )
            
        except Exception as e:
            logger.error(f"Citation extraction failed for document {document.id}: {e}")
            errors.append(f"Extraction failed: {e}")
            
            return CitationExtractionResult(
                citations=[],
                extraction_stats={"error": str(e)},
                errors=errors,
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
    
    async def _extract_raw_citations(self, text: str) -> List[EyeciteCitation]:
        """Extract raw citations using eyecite"""
        try:
            # Run eyecite extraction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            citations = await loop.run_in_executor(None, get_citations, text)
            return citations
        except Exception as e:
            logger.error(f"Raw citation extraction failed: {e}")
            raise
    
    async def _resolve_citation_references(self, citations: List[EyeciteCitation]) -> List[EyeciteCitation]:
        """Resolve supra and id citations to their antecedents"""
        try:
            # Run resolve_citations in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            resolved_citations = await loop.run_in_executor(None, resolve_citations, citations)
            return resolved_citations
        except Exception as e:
            logger.warning(f"Citation resolution failed: {e}")
            return citations  # Return unresolved citations if resolution fails
    
    async def _convert_to_db_citation(self, raw_citation: EyeciteCitation, document_id: str) -> Citation:
        """Convert eyecite citation to database model"""
        
        # Extract basic information
        citation_type = type(raw_citation).__name__
        reporter = getattr(raw_citation, 'reporter', None)
        volume = getattr(raw_citation, 'volume', None)
        page = getattr(raw_citation, 'page', None)
        
        # Get position in text
        span = raw_citation.span()
        position_start = span[0] if span else None
        position_end = span[1] if span else None
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(raw_citation)
        
        # Get corrected citation text
        try:
            corrected_citation = raw_citation.corrected_citation()
        except:
            corrected_citation = str(raw_citation)
            
        metadata = {
            "original_text": str(raw_citation),
            "corrected_citation": corrected_citation,
            "citation_class": citation_type,
            "groups": getattr(raw_citation, 'groups', {}),
            "court": getattr(raw_citation, 'court', None),
            "year": getattr(raw_citation, 'year', None)
        }
        
        # Add antecedent information if available
        if hasattr(raw_citation, 'antecedent_guess') and raw_citation.antecedent_guess:
            metadata["antecedent"] = {
                "text": str(raw_citation.antecedent_guess),
                "score": getattr(raw_citation.antecedent_guess, 'score', None)
            }
        
        return Citation(
            document_id=document_id,
            citation_text=corrected_citation,
            citation_type=citation_type,
            reporter=str(reporter) if reporter else None,
            volume=str(volume) if volume else None,
            page=str(page) if page else None,
            position_start=position_start,
            position_end=position_end,
            confidence_score=confidence_score,
            doc_metadata=metadata,
            created_at=datetime.utcnow()
        )
    
    def _calculate_confidence_score(self, citation: EyeciteCitation) -> float:
        """Calculate confidence score for citation quality"""
        score = 0.8  # Base score
        
        # Increase score for full citations
        if isinstance(citation, FullCaseCitation):
            score += 0.15
        
        # Increase score if volume and page are present
        if hasattr(citation, 'volume') and citation.volume:
            score += 0.05
        if hasattr(citation, 'page') and citation.page:
            score += 0.05
        
        # Decrease score for uncertain patterns
        if isinstance(citation, (ShortCaseCitation, SupraCitation, IdCitation)):
            score -= 0.1
        
        # Adjust based on reporter recognition
        if hasattr(citation, 'reporter') and citation.reporter:
            score += 0.05
        
        return min(1.0, max(0.1, score))
    
    def _analyze_citation_types(self, citations: List[EyeciteCitation]) -> Dict[str, int]:
        """Analyze the types of citations found"""
        type_counts = {}
        for citation in citations:
            citation_type = type(citation).__name__
            type_counts[citation_type] = type_counts.get(citation_type, 0) + 1
        return type_counts
    
    async def _create_citation_embeddings(self, citations: List[Citation], db_session: AsyncSession):
        """Create vector embeddings for citations using OpenAI"""
        if not citations:
            return
        
        # Check if OpenAI is configured
        if not settings.OPENAI_API_KEY:
            logger.info("OpenAI API key not configured, skipping embedding creation")
            return
        
        try:
            # Import OpenAI service (lazy import to avoid dependency issues)
            from nlp.openai_service import openai_service
            
            # Create embeddings for all citations
            embedding_results = await openai_service.create_citation_embeddings(
                citations=citations,
                include_context=True
            )
            
            # Create CitationEmbedding records
            citation_embeddings = []
            for citation, embedding_result in zip(citations, embedding_results):
                citation_embedding = CitationEmbedding(
                    citation_id=citation.id,
                    embedding=embedding_result.embedding,
                    created_at=datetime.utcnow()
                )
                citation_embeddings.append(citation_embedding)
                db_session.add(citation_embedding)
            
            # Commit the embeddings
            await db_session.commit()
            
            logger.info(f"Created embeddings for {len(citation_embeddings)} citations "
                       f"(tokens: {sum(r.tokens_used for r in embedding_results)}, "
                       f"cached: {sum(1 for r in embedding_results if r.cached)})")
            
        except ImportError as e:
            logger.error(f"Failed to import OpenAI service: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create citation embeddings: {e}")
            # Don't re-raise - embeddings are optional
    
    async def analyze_citation_authority(self, citation: Citation) -> CitationAnalysis:
        """Analyze the precedential authority of a citation"""
        try:
            # Extract court information from reporter
            court_level = self._determine_court_level(citation.reporter, citation.doc_metadata)
            jurisdiction = self._determine_jurisdiction(citation.reporter, citation.doc_metadata)
            precedential_strength = self._assess_precedential_strength(court_level, jurisdiction)
            authority_score = self._calculate_authority_score(court_level, precedential_strength)
            
            return CitationAnalysis(
                citation=citation,
                court_level=court_level,
                jurisdiction=jurisdiction,
                precedential_strength=precedential_strength,
                authority_score=authority_score,
                confidence_score=citation.confidence_score
            )
            
        except Exception as e:
            logger.warning(f"Authority analysis failed for citation {citation.id}: {e}")
            return CitationAnalysis(citation=citation)
    
    def _determine_court_level(self, reporter: Optional[str], metadata: Dict) -> Optional[str]:
        """Determine the court level from reporter information"""
        if not reporter:
            return None
        
        reporter_lower = reporter.lower()
        
        # Supreme Court
        if any(pattern in reporter_lower for pattern in ['u.s.', 'u. s.', 'supreme']):
            return "supreme"
        
        # Federal Circuit Courts
        if any(pattern in reporter_lower for pattern in ['f.2d', 'f.3d', 'f.4th', 'fed.']):
            return "circuit"
        
        # Federal District Courts
        if any(pattern in reporter_lower for pattern in ['f.supp', 'f. supp']):
            return "district"
        
        # State Supreme Courts (common patterns)
        if any(pattern in reporter_lower for pattern in ['n.e.', 'n.w.', 's.e.', 's.w.', 'so.', 'a.2d', 'p.2d']):
            return "state_high"
        
        return "unknown"
    
    def _determine_jurisdiction(self, reporter: Optional[str], metadata: Dict) -> Optional[str]:
        """Determine jurisdiction from citation information"""
        if not reporter:
            return None
        
        # Federal courts
        if any(pattern in reporter.lower() for pattern in ['u.s.', 'f.2d', 'f.3d', 'f.supp', 'fed.']):
            return "federal"
        
        # State courts (would need more sophisticated mapping)
        return "state"
    
    def _assess_precedential_strength(self, court_level: Optional[str], jurisdiction: Optional[str]) -> Optional[str]:
        """Assess precedential strength based on court hierarchy"""
        if court_level == "supreme":
            return "binding_nationwide"
        elif court_level == "circuit":
            return "binding_circuit"
        elif court_level == "district":
            return "persuasive"
        elif court_level == "state_high":
            return "binding_state"
        else:
            return "unknown"
    
    def _calculate_authority_score(self, court_level: Optional[str], precedential_strength: Optional[str]) -> Optional[float]:
        """Calculate numerical authority score (0.0 to 1.0)"""
        if not court_level or not precedential_strength:
            return None
        
        authority_scores = {
            "binding_nationwide": 1.0,
            "binding_circuit": 0.8,
            "binding_state": 0.7,
            "persuasive": 0.4,
            "unknown": 0.2
        }
        
        return authority_scores.get(precedential_strength, 0.2)
    
    async def create_citation_relationships(
        self, 
        citations: List[Citation], 
        db_session: AsyncSession
    ) -> List[CitationRelationship]:
        """Create relationships between related citations"""
        relationships = []
        
        for citation in citations:
            if citation.doc_metadata and "antecedent" in citation.doc_metadata:
                # Find the antecedent citation in our database
                antecedent_text = citation.doc_metadata["antecedent"]["text"]
                confidence = citation.doc_metadata["antecedent"].get("score", 0.5)
                
                # TODO: Find matching citation in database
                # This would require a more sophisticated matching algorithm
                
                # For now, create a placeholder relationship
                relationship = CitationRelationship(
                    source_citation_id=citation.id,
                    target_citation_id=None,  # Would be filled when antecedent is found
                    relationship_type="references",
                    confidence_score=confidence,
                    created_at=datetime.utcnow()
                )
                relationships.append(relationship)
        
        return relationships
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get overall extraction statistics"""
        stats = self.extraction_stats.copy()
        
        if stats["total_processed"] > 0:
            stats["avg_citations_per_document"] = stats["total_citations"] / stats["total_processed"]
            stats["avg_processing_time_ms"] = stats["processing_time_ms"] / stats["total_processed"]
            stats["error_rate"] = stats["total_errors"] / stats["total_processed"]
        
        return stats


# Global service instance
citation_service = CitationExtractionService()