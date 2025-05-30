"""
Legal NLP service using spaCy for entity extraction and text analysis.
Replaces LexNLP functionality with Python 3.12 compatible libraries.
"""

import spacy
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LegalNLPService:
    """Service for legal document NLP processing using spaCy."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the legal NLP service.
        
        Args:
            model_name: spaCy model to use for processing
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"Failed to load spaCy model: {model_name}")
            raise
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of entity dictionaries with text, label, start, end positions
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_),
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": getattr(ent, "_.confidence", None)
            })
        
        return entities
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract legal-specific entities from text.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary of entity types and their values
        """
        doc = self.nlp(text)
        legal_entities = {
            "organizations": [],
            "persons": [],
            "locations": [],
            "dates": [],
            "money": [],
            "laws": [],
            "contracts": []
        }
        
        for ent in doc.ents:
            if ent.label_ == "ORG":
                legal_entities["organizations"].append(ent.text)
            elif ent.label_ == "PERSON":
                legal_entities["persons"].append(ent.text)
            elif ent.label_ in ["GPE", "LOC"]:
                legal_entities["locations"].append(ent.text)
            elif ent.label_ == "DATE":
                legal_entities["dates"].append(ent.text)
            elif ent.label_ == "MONEY":
                legal_entities["money"].append(ent.text)
            elif ent.label_ in ["LAW", "STATUTE"]:
                legal_entities["laws"].append(ent.text)
        
        # Custom patterns for legal documents
        legal_entities.update(self._extract_legal_patterns(text))
        
        return legal_entities
    
    def _extract_legal_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        Extract legal-specific patterns using regex and spaCy patterns.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary of additional legal entities
        """
        import re
        
        patterns = {
            "case_citations": [],
            "statutes": [],
            "contract_terms": [],
            "legal_concepts": []
        }
        
        # Case citation patterns (simplified)
        case_pattern = r'\b\d+\s+[A-Z][a-z]*\.?\s*\d+[a-z]*\b'
        patterns["case_citations"] = re.findall(case_pattern, text)
        
        # Statute patterns
        statute_pattern = r'\b\d+\s+U\.?S\.?C\.?\s*§?\s*\d+\b'
        patterns["statutes"] = re.findall(statute_pattern, text)
        
        # Common legal terms
        legal_terms = [
            "plaintiff", "defendant", "contract", "breach", "damages",
            "liability", "negligence", "warranty", "lease", "tenant",
            "landlord", "rent", "eviction", "notice", "default"
        ]
        
        for term in legal_terms:
            if term.lower() in text.lower():
                patterns["legal_concepts"].append(term)
        
        return patterns
    
    def analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze document structure and extract key sections.
        
        Args:
            text: Input document text
            
        Returns:
            Dictionary with document structure analysis
        """
        doc = self.nlp(text)
        
        analysis = {
            "sentence_count": len(list(doc.sents)),
            "word_count": len([token for token in doc if not token.is_space]),
            "paragraph_count": text.count('\n\n') + 1,
            "key_phrases": self._extract_key_phrases(doc),
            "document_type": self._classify_document_type(text),
            "complexity_score": self._calculate_complexity(doc)
        }
        
        return analysis
    
    def _extract_key_phrases(self, doc) -> List[str]:
        """Extract key phrases using noun chunks and named entities."""
        key_phrases = []
        
        # Add noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Multi-word phrases
                key_phrases.append(chunk.text)
        
        # Add named entities
        for ent in doc.ents:
            key_phrases.append(ent.text)
        
        # Remove duplicates and sort by frequency
        return list(set(key_phrases))[:20]  # Top 20 phrases
    
    def _classify_document_type(self, text: str) -> str:
        """Classify the type of legal document based on content."""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ["lease", "rental", "tenant", "landlord"]):
            return "lease_agreement"
        elif any(term in text_lower for term in ["complaint", "plaintiff", "defendant"]):
            return "legal_complaint"
        elif any(term in text_lower for term in ["memorandum", "memo", "brief"]):
            return "legal_memorandum"
        elif any(term in text_lower for term in ["contract", "agreement", "party"]):
            return "contract"
        else:
            return "unknown"
    
    def _calculate_complexity(self, doc) -> float:
        """Calculate document complexity score based on various factors."""
        if not doc:
            return 0.0
        
        # Factors: sentence length, word complexity, legal terminology
        avg_sentence_length = len([token for token in doc if not token.is_space]) / len(list(doc.sents))
        complex_words = len([token for token in doc if len(token.text) > 6])
        total_words = len([token for token in doc if not token.is_space])
        
        complexity = (avg_sentence_length * 0.4) + (complex_words / total_words * 0.6)
        return min(complexity, 10.0)  # Cap at 10