"""
Hybrid Legal NLP service combining spaCy and Hugging Face LEGAL-BERT for optimal performance.
"""

from typing import List, Dict, Any, Optional
import logging
from .legal_nlp_service import LegalNLPService
from .legal_bert_service import LegalBERTService

logger = logging.getLogger(__name__)


class HybridLegalNLP:
    """
    Hybrid service combining spaCy and LEGAL-BERT for comprehensive legal NLP.
    Uses spaCy for fast entity recognition and LEGAL-BERT for semantic analysis.
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm", 
                 bert_model: str = "nlpaueb/legal-bert-base-uncased"):
        """
        Initialize the hybrid legal NLP service.
        
        Args:
            spacy_model: spaCy model name
            bert_model: Hugging Face LEGAL-BERT model name
        """
        try:
            self.spacy_service = LegalNLPService(spacy_model)
            self.bert_service = LegalBERTService(bert_model)
            logger.info("Initialized hybrid legal NLP service")
        except Exception as e:
            logger.error(f"Failed to initialize hybrid service: {e}")
            raise
    
    def comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive legal document analysis using both models.
        
        Args:
            text: Input legal text
            
        Returns:
            Combined analysis results
        """
        try:
            # Get spaCy analysis (fast entity recognition and structure)
            spacy_entities = self.spacy_service.extract_entities(text)
            spacy_legal_entities = self.spacy_service.extract_legal_entities(text)
            spacy_structure = self.spacy_service.analyze_document_structure(text)
            
            # Get LEGAL-BERT analysis (semantic understanding)
            bert_classification = self.bert_service.classify_legal_text(text)
            bert_concepts = self.bert_service.extract_legal_concepts(text)
            bert_entities = self.bert_service.extract_legal_entities_bert(text)
            
            # Combine results
            combined_analysis = {
                "document_analysis": {
                    "type": bert_classification.get("document_type", spacy_structure.get("document_type")),
                    "confidence": bert_classification.get("confidence", 0.0),
                    "legal_domain": bert_classification.get("legal_domain", []),
                    "complexity_score": spacy_structure.get("complexity_score", 0.0),
                    "sentence_count": spacy_structure.get("sentence_count", 0),
                    "word_count": spacy_structure.get("word_count", 0)
                },
                "entities": {
                    "spacy_entities": spacy_entities,
                    "bert_entities": bert_entities,
                    "legal_entities": spacy_legal_entities,
                    "legal_concepts": bert_concepts
                },
                "semantic_analysis": {
                    "key_phrases": spacy_structure.get("key_phrases", []),
                    "complexity_indicators": bert_classification.get("complexity_indicators", []),
                    "embedding_available": True
                },
                "model_info": {
                    "spacy_model": self.spacy_service.nlp.meta.get("name", "unknown"),
                    "bert_model": self.bert_service.get_model_info()
                }
            }
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {"error": str(e)}
    
    def extract_all_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract entities using both spaCy and LEGAL-BERT.
        
        Args:
            text: Input text
            
        Returns:
            Combined entity extraction results
        """
        entities = {
            "spacy_entities": [],
            "bert_entities": [],
            "combined_unique": []
        }
        
        try:
            # Get entities from both models
            entities["spacy_entities"] = self.spacy_service.extract_entities(text)
            entities["bert_entities"] = self.bert_service.extract_legal_entities_bert(text)
            
            # Combine and deduplicate entities
            seen_entities = set()
            combined = []
            
            for entity in entities["spacy_entities"] + entities["bert_entities"]:
                entity_key = (entity.get("text", "").lower(), entity.get("label", ""))
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    combined.append(entity)
            
            entities["combined_unique"] = combined
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
        
        return entities
    
    def compare_models(self, text: str) -> Dict[str, Any]:
        """
        Compare performance and results of spaCy vs LEGAL-BERT models.
        
        Args:
            text: Input text for comparison
            
        Returns:
            Comparison results
        """
        import time
        
        comparison = {
            "spacy_results": {},
            "bert_results": {},
            "performance": {},
            "recommendations": []
        }
        
        try:
            # Time spaCy analysis
            start_time = time.time()
            spacy_entities = self.spacy_service.extract_entities(text)
            spacy_structure = self.spacy_service.analyze_document_structure(text)
            spacy_time = time.time() - start_time
            
            # Time LEGAL-BERT analysis
            start_time = time.time()
            bert_classification = self.bert_service.classify_legal_text(text)
            bert_concepts = self.bert_service.extract_legal_concepts(text)
            bert_time = time.time() - start_time
            
            comparison["spacy_results"] = {
                "entity_count": len(spacy_entities),
                "document_type": spacy_structure.get("document_type"),
                "complexity_score": spacy_structure.get("complexity_score")
            }
            
            comparison["bert_results"] = {
                "document_type": bert_classification.get("document_type"),
                "confidence": bert_classification.get("confidence"),
                "legal_concepts_count": sum(len(v) for v in bert_concepts.values())
            }
            
            comparison["performance"] = {
                "spacy_time_seconds": spacy_time,
                "bert_time_seconds": bert_time,
                "speed_ratio": bert_time / spacy_time if spacy_time > 0 else 0
            }
            
            # Generate recommendations
            if spacy_time < bert_time * 0.1:
                comparison["recommendations"].append("Use spaCy for real-time processing")
            if bert_classification.get("confidence", 0) > 0.8:
                comparison["recommendations"].append("LEGAL-BERT provides high-confidence classification")
            if len(spacy_entities) > 10:
                comparison["recommendations"].append("spaCy effective for entity-rich documents")
            
        except Exception as e:
            comparison["error"] = str(e)
            logger.error(f"Error in model comparison: {e}")
        
        return comparison
    
    def get_text_embedding(self, text: str):
        """Get LEGAL-BERT embedding for semantic similarity."""
        return self.bert_service.encode_text(text)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using LEGAL-BERT."""
        return self.bert_service.calculate_text_similarity(text1, text2)
    
    def get_service_status(self) -> Dict[str, bool]:
        """Check the status of both underlying services."""
        return {
            "spacy_available": self.spacy_service.nlp is not None,
            "bert_available": self.bert_service.model is not None,
            "bert_tokenizer_available": self.bert_service.tokenizer is not None,
            "bert_ner_available": self.bert_service.ner_pipeline is not None
        }