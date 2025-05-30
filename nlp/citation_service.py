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

# Add import for the new semantic similarity service
from nlp.semantic_similarity import semantic_similarity_service

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
        create_embeddings: bool = False,
        create_relationships: bool = True
    ) -> CitationExtractionResult:
        """
        Extract all citations from a document and store in database
        
        Args:
            document: Document to process
            db_session: Database session
            resolve_references: Whether to resolve supra/id citations
            create_embeddings: Whether to generate embeddings for citations
            create_relationships: Whether to create citation relationships
            
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
            
            # Create citation relationships if requested
            relationships_created = 0
            if create_relationships and db_citations:
                try:
                    relationships = await self.create_citation_relationships(db_citations, db_session)
                    # Save relationships to database
                    for relationship in relationships:
                        db_session.add(relationship)
                    await db_session.commit()
                    relationships_created = len(relationships)
                    logger.info(f"Created {relationships_created} citation relationships")
                except Exception as e:
                    errors.append(f"Failed to create citation relationships: {e}")
                    logger.warning(f"Citation relationship creation error: {e}")
            
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
                "relationships_created": relationships_created,
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
        """
        Create relationships between related citations using sophisticated matching algorithms.
        
        This method resolves reference citations (id, supra, short forms) to their antecedent
        full citations using multiple strategies:
        1. Position-based matching for id citations
        2. Name-based matching for supra citations  
        3. Reporter/volume matching for short citations
        4. Semantic similarity for ambiguous cases
        
        Args:
            citations: List of Citation objects from the document
            db_session: Database session for operations
            
        Returns:
            List of CitationRelationship objects with resolved target citations
        """
        if not citations:
            return []
        
        relationships = []
        
        try:
            # Sort citations by position for sequential processing
            sorted_citations = sorted(citations, key=lambda c: c.position_start or 0)
            
            # Build citation index for efficient lookup
            citation_index = self._build_citation_index(sorted_citations)
            
            # Process each citation for potential relationships
            for i, citation in enumerate(sorted_citations):
                # Skip if not a reference citation
                if not self._is_reference_citation(citation):
                    continue
                
                # Find antecedent using appropriate strategy
                target_citation = await self._resolve_citation_antecedent(
                    citation, 
                    sorted_citations[:i],  # Only consider preceding citations
                    citation_index,
                    db_session
                )
                
                if target_citation:
                    relationship_type = self._determine_relationship_type(citation, target_citation)
                    confidence_score = self._calculate_relationship_confidence(citation, target_citation)
                    
                    relationship = CitationRelationship(
                        source_citation_id=citation.id,
                        target_citation_id=target_citation.id,
                        relationship_type=relationship_type,
                        confidence_score=confidence_score,
                        created_at=datetime.utcnow()
                    )
                    relationships.append(relationship)
                    
                    logger.debug(f"Resolved {citation.citation_type} '{citation.citation_text}' "
                               f"to '{target_citation.citation_text}' (confidence: {confidence_score:.2f})")
        
        except Exception as e:
            logger.error(f"Error creating citation relationships: {e}")
            # Don't re-raise - relationships are optional
        
        return relationships
    
    def _build_citation_index(self, citations: List[Citation]) -> Dict[str, Any]:
        """Build an index of citations for efficient lookup by different criteria"""
        index = {
            "by_position": {},
            "by_reporter": {},
            "by_case_name": {},
            "full_citations": [],
            "reference_citations": []
        }
        
        for citation in citations:
            # Index by position
            if citation.position_start:
                index["by_position"][citation.position_start] = citation
            
            # Index full citations by reporter/volume
            if citation.citation_type == "FullCaseCitation" and citation.reporter and citation.volume:
                reporter_key = f"{citation.reporter}_{citation.volume}"
                if reporter_key not in index["by_reporter"]:
                    index["by_reporter"][reporter_key] = []
                index["by_reporter"][reporter_key].append(citation)
                index["full_citations"].append(citation)
                
                # Extract case name for supra matching
                case_name = self._extract_case_name(citation)
                if case_name:
                    case_key = case_name.lower().strip()
                    if case_key not in index["by_case_name"]:
                        index["by_case_name"][case_key] = []
                    index["by_case_name"][case_key].append(citation)
            else:
                index["reference_citations"].append(citation)
        
        return index
    
    def _is_reference_citation(self, citation: Citation) -> bool:
        """Check if citation is a reference type that needs resolution"""
        reference_types = ["IdCitation", "SupraCitation", "ShortCaseCitation"]
        return citation.citation_type in reference_types
    
    async def _resolve_citation_antecedent(
        self,
        citation: Citation,
        preceding_citations: List[Citation],
        citation_index: Dict[str, Any],
        db_session: AsyncSession
    ) -> Optional[Citation]:
        """
        Resolve a reference citation to its antecedent using multiple strategies
        """
        citation_type = citation.citation_type
        
        # Strategy 1: Use eyecite's antecedent guess if available
        if citation.doc_metadata and "antecedent" in citation.doc_metadata:
            eyecite_match = await self._match_eyecite_antecedent(
                citation, citation_index, preceding_citations
            )
            if eyecite_match:
                return eyecite_match
        
        # Strategy 2: Type-specific matching
        if citation_type == "IdCitation":
            return self._resolve_id_citation(citation, preceding_citations)
        elif citation_type == "SupraCitation":
            return self._resolve_supra_citation(citation, citation_index)
        elif citation_type == "ShortCaseCitation":
            return self._resolve_short_citation(citation, citation_index)
        
        # Strategy 3: Semantic similarity fallback
        if settings.OPENAI_API_KEY:
            return await self._resolve_semantic_similarity(
                citation, citation_index["full_citations"], db_session
            )
        
        return None
    
    async def _match_eyecite_antecedent(
        self,
        citation: Citation,
        citation_index: Dict[str, Any],
        preceding_citations: List[Citation]
    ) -> Optional[Citation]:
        """Match using eyecite's antecedent guess"""
        try:
            antecedent_text = citation.doc_metadata["antecedent"]["text"]
            
            # Look for exact or partial match in preceding citations
            for candidate in reversed(preceding_citations):  # Start with most recent
                if candidate.citation_type == "FullCaseCitation":
                    # Check for text similarity
                    if self._citations_match(antecedent_text, candidate.citation_text):
                        return candidate
            
            return None
        except (KeyError, TypeError):
            return None
    
    def _resolve_id_citation(self, citation: Citation, preceding_citations: List[Citation]) -> Optional[Citation]:
        """Resolve 'Id.' citations to the immediately preceding full citation"""
        # Id citations refer to the immediately preceding full case citation
        for candidate in reversed(preceding_citations):
            if candidate.citation_type == "FullCaseCitation":
                return candidate
        return None
    
    def _resolve_supra_citation(self, citation: Citation, citation_index: Dict[str, Any]) -> Optional[Citation]:
        """Resolve 'supra' citations by matching case names"""
        # Extract case name from supra citation
        supra_case_name = self._extract_case_name_from_supra(citation.citation_text)
        if not supra_case_name:
            return None
        
        case_key = supra_case_name.lower().strip()
        
        # Look for matching case names in index
        if case_key in citation_index["by_case_name"]:
            # Return the most recent full citation with this case name
            candidates = citation_index["by_case_name"][case_key]
            return candidates[-1] if candidates else None
        
        # Fuzzy matching for partial case names
        for indexed_case_name, candidates in citation_index["by_case_name"].items():
            if (supra_case_name.lower() in indexed_case_name or 
                indexed_case_name in supra_case_name.lower()):
                return candidates[-1] if candidates else None
        
        return None
    
    def _resolve_short_citation(self, citation: Citation, citation_index: Dict[str, Any]) -> Optional[Citation]:
        """Resolve short citations by matching reporter and volume"""
        # Extract reporter info from short citation
        if not citation.reporter or not citation.volume:
            return None
        
        reporter_key = f"{citation.reporter}_{citation.volume}"
        
        if reporter_key in citation_index["by_reporter"]:
            candidates = citation_index["by_reporter"][reporter_key]
            # Return the most recent matching citation
            return candidates[-1] if candidates else None
        
        return None
    
    async def _resolve_semantic_similarity(
        self,
        citation: Citation,
        full_citations: List[Citation],
        db_session: AsyncSession
    ) -> Optional[Citation]:
        """
        Use advanced semantic similarity to resolve ambiguous citations.
        
        This enhanced method uses OpenAI embeddings combined with legal context analysis
        to match reference citations to their antecedents when pattern matching fails.
        """
        try:
            # Use the comprehensive semantic similarity service
            from nlp.semantic_similarity import semantic_similarity_service
            
            # Get document text for context extraction
            # Note: In a real implementation, you'd get this from the document
            # For now, we'll use the citations themselves as a proxy
            document_text = self._reconstruct_document_text(citation, full_citations)
            
            # Find semantic matches using the advanced service
            semantic_matches = await semantic_similarity_service.find_semantic_matches(
                source_citation=citation,
                candidate_citations=full_citations,
                document_text=document_text,
                threshold=0.7,  # Adjusted threshold for legal accuracy
                max_matches=3
            )
            
            if semantic_matches:
                # Get the best match
                best_match = semantic_matches[0]
                
                # Log detailed match information for debugging
                logger.info(f"Semantic match found for {citation.citation_type} '{citation.citation_text[:50]}...'")
                logger.info(f"  → Target: '{best_match.target_citation_id}' (confidence: {best_match.combined_confidence:.3f})")
                logger.info(f"  → Reason: {best_match.match_reason}")
                logger.info(f"  → Semantic score: {best_match.similarity_score:.3f}")
                logger.info(f"  → Context overlap: {best_match.context_overlap:.3f}")
                
                if best_match.semantic_features.get("shared_case_names"):
                    logger.info(f"  → Shared case names: {best_match.semantic_features['shared_case_names']}")
                if best_match.semantic_features.get("shared_concepts"):
                    logger.info(f"  → Shared concepts: {best_match.semantic_features['shared_concepts'][:3]}")
                
                # Find and return the actual Citation object
                target_citation_id = best_match.target_citation_id
                for candidate in full_citations:
                    if str(candidate.id) == target_citation_id:
                        return candidate
            
            logger.debug(f"No semantic match found for {citation.citation_type} '{citation.citation_text}'")
            return None
            
        except Exception as e:
            logger.warning(f"Enhanced semantic similarity resolution failed: {e}")
            return None
    
    def _reconstruct_document_text(self, citation: Citation, all_citations: List[Citation]) -> str:
        """
        Reconstruct document text context from available citations.
        
        In a real implementation, this would get the actual document text.
        For now, we'll create a reasonable approximation from citations.
        """
        try:
            # Sort citations by position
            sorted_citations = sorted(
                [c for c in all_citations + [citation] if c.position_start], 
                key=lambda x: x.position_start
            )
            
            # Create a pseudo-document by concatenating citation texts with context
            document_parts = []
            for i, cit in enumerate(sorted_citations):
                # Add some legal context around each citation
                if i == 0:
                    document_parts.append("The court considers the following precedential authority. ")
                elif i == len(sorted_citations) - 1:
                    document_parts.append("In conclusion, ")
                else:
                    document_parts.append("Furthermore, ")
                
                document_parts.append(cit.citation_text)
                document_parts.append(". ")
            
            return "".join(document_parts)
            
        except Exception as e:
            logger.warning(f"Error reconstructing document text: {e}")
            # Fallback: just concatenate citation texts
            return ". ".join([c.citation_text for c in all_citations + [citation]])
    
    def _extract_case_name(self, citation: Citation) -> Optional[str]:
        """Extract case name from a full citation"""
        try:
            # Try to get case name from metadata first
            if citation.doc_metadata and "groups" in citation.doc_metadata:
                groups = citation.doc_metadata["groups"]
                if isinstance(groups, dict) and "case_name" in groups:
                    return groups["case_name"]
            
            # Fallback: extract from citation text
            citation_text = citation.citation_text
            # Case names typically come before the reporter
            if citation.reporter and citation.reporter in citation_text:
                name_part = citation_text.split(citation.reporter)[0].strip()
                # Remove common prefixes and clean up
                name_part = name_part.replace(",", "").strip()
                return name_part if name_part else None
            
            return None
        except Exception:
            return None
    
    def _extract_case_name_from_supra(self, supra_text: str) -> Optional[str]:
        """Extract case name from supra citation text"""
        try:
            # Common patterns: "Case Name, supra" or "Case Name supra"
            supra_text = supra_text.strip()
            
            # Remove "supra" and everything after it
            if "supra" in supra_text.lower():
                case_part = supra_text.lower().split("supra")[0].strip()
                # Remove trailing comma
                case_part = case_part.rstrip(",").strip()
                return case_part if case_part else None
            
            return None
        except Exception:
            return None
    
    def _citations_match(self, text1: str, text2: str) -> bool:
        """Check if two citation texts are similar enough to be the same case"""
        if not text1 or not text2:
            return False
        
        # Normalize texts
        norm1 = text1.lower().strip()
        norm2 = text2.lower().strip()
        
        # Exact match
        if norm1 == norm2:
            return True
        
        # Partial match (one contains the other)
        if norm1 in norm2 or norm2 in norm1:
            return True
        
        # Extract key components and compare
        # This is a simple heuristic - could be made more sophisticated
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        # If they share significant words, consider it a match
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) > 0:
            similarity = len(intersection) / len(union)
            return similarity > 0.6  # 60% word overlap threshold
        
        return False
    
    def _determine_relationship_type(self, source: Citation, target: Citation) -> str:
        """Determine the specific type of relationship between citations"""
        source_type = source.citation_type
        
        if source_type == "IdCitation":
            return "id_reference"
        elif source_type == "SupraCitation":
            return "supra_reference"
        elif source_type == "ShortCaseCitation":
            return "short_form_reference"
        else:
            return "general_reference"
    
    def _calculate_relationship_confidence(self, source: Citation, target: Citation) -> float:
        """Calculate confidence score for the relationship"""
        base_confidence = 0.7
        
        # Increase confidence for exact matches
        if source.reporter == target.reporter and source.volume == target.volume:
            base_confidence += 0.2
        
        # Increase confidence based on citation type certainty
        if source.citation_type == "IdCitation":
            base_confidence += 0.1  # Id citations are usually clear
        elif source.citation_type == "SupraCitation":
            base_confidence += 0.05  # Supra citations can be ambiguous
        
        # Factor in original confidence scores
        if source.confidence_score:
            base_confidence *= source.confidence_score
        
        return min(1.0, max(0.1, base_confidence))
    
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