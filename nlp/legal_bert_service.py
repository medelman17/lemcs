"""
Legal NLP service using Hugging Face LEGAL-BERT models for enhanced legal document processing.
"""

from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LegalBERTService:
    """Service for legal document processing using LEGAL-BERT models."""
    
    def __init__(self, model_name: str = "nlpaueb/legal-bert-base-uncased"):
        """
        Initialize the Legal BERT service.
        
        Args:
            model_name: Hugging Face model to use for processing
        """
        try:
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Try to initialize NER pipeline (if available)
            self.ner_pipeline = None
            try:
                self.ner_pipeline = pipeline("ner", 
                                            model="opennyaiorg/en_legal_ner_trf", 
                                            aggregation_strategy="simple")
                logger.info("Loaded legal NER pipeline")
            except Exception as e:
                logger.warning(f"Could not load legal NER pipeline: {e}")
            
            logger.info(f"Loaded Legal BERT model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load Legal BERT model: {model_name}, error: {e}")
            raise
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text using Legal BERT to get contextualized embeddings.
        
        Args:
            text: Input text to encode
            
        Returns:
            Tensor of embeddings
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                               padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding for sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def extract_legal_entities_bert(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract legal entities using Legal BERT NER pipeline.
        
        Args:
            text: Input text to process
            
        Returns:
            List of entity dictionaries
        """
        if not self.ner_pipeline:
            logger.warning("Legal NER pipeline not available")
            return []
        
        try:
            entities = self.ner_pipeline(text)
            
            # Process and standardize entity format
            processed_entities = []
            for entity in entities:
                processed_entities.append({
                    "text": entity["word"],
                    "label": entity["entity_group"],
                    "confidence": entity["score"],
                    "start": entity["start"],
                    "end": entity["end"]
                })
            
            return processed_entities
        except Exception as e:
            logger.error(f"Error in legal entity extraction: {e}")
            return []
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two legal texts using Legal BERT.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        try:
            embedding1 = self.encode_text(text1)
            embedding2 = self.encode_text(text2)
            
            # Calculate cosine similarity
            cosine_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2)
            return float(cosine_sim.item())
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def classify_legal_text(self, text: str) -> Dict[str, Any]:
        """
        Classify legal text type and extract key characteristics.
        
        Args:
            text: Input legal text
            
        Returns:
            Dictionary with classification results
        """
        # Get embeddings for the text
        embeddings = self.encode_text(text)
        
        # Simple classification based on keywords and patterns
        text_lower = text.lower()
        
        classification = {
            "document_type": "unknown",
            "confidence": 0.0,
            "legal_domain": [],
            "complexity_indicators": [],
            "embedding_shape": embeddings.shape
        }
        
        # Document type classification (order matters - most specific first)
        if any(term in text_lower for term in ["memorandum", "memo", "brief"]):
            classification["document_type"] = "legal_memorandum"
            classification["confidence"] = 0.85
            classification["legal_domain"].append("analysis")
        elif any(term in text_lower for term in ["lease", "rental", "tenant", "landlord"]):
            classification["document_type"] = "lease_agreement"
            classification["confidence"] = 0.9
            classification["legal_domain"].append("real_estate")
        elif any(term in text_lower for term in ["complaint", "plaintiff", "defendant"]):
            classification["document_type"] = "legal_complaint"
            classification["confidence"] = 0.85
            classification["legal_domain"].append("litigation")
        elif any(term in text_lower for term in ["contract", "agreement", "party"]):
            classification["document_type"] = "contract"
            classification["confidence"] = 0.8
            classification["legal_domain"].append("commercial")
        
        # Complexity indicators
        if len(text.split()) > 1000:
            classification["complexity_indicators"].append("long_document")
        if text.count("§") > 0 or text.count("USC") > 0:
            classification["complexity_indicators"].append("statutory_references")
        if text.count("v.") > 0 or text.count("F.") > 0:
            classification["complexity_indicators"].append("case_citations")
        
        return classification
    
    def extract_legal_concepts(self, text: str) -> Dict[str, List[str]]:
        """
        Extract legal concepts and terminology from text.
        
        Args:
            text: Input legal text
            
        Returns:
            Dictionary of extracted legal concepts
        """
        concepts = {
            "legal_terms": [],
            "parties": [],
            "dates": [],
            "amounts": [],
            "citations": [],
            "statutes": []
        }
        
        # Get Legal BERT entities if available
        if self.ner_pipeline:
            bert_entities = self.extract_legal_entities_bert(text)
            for entity in bert_entities:
                label = entity["label"].upper()
                if "PERSON" in label or "PARTY" in label:
                    concepts["parties"].append(entity["text"])
                elif "DATE" in label:
                    concepts["dates"].append(entity["text"])
                elif "STATUTE" in label or "PROVISION" in label:
                    concepts["statutes"].append(entity["text"])
        
        # Enhanced pattern matching using Legal BERT context
        import re
        
        # Legal terminology patterns
        legal_terms_patterns = [
            r'\b(plaintiff|defendant|appellant|appellee|petitioner|respondent)\b',
            r'\b(contract|agreement|lease|covenant|warranty|breach)\b',
            r'\b(damages|liability|negligence|fraud|misrepresentation)\b',
            r'\b(jurisdiction|venue|standing|cause of action)\b'
        ]
        
        for pattern in legal_terms_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts["legal_terms"].extend(matches)
        
        # Citation patterns
        citation_pattern = r'\b\d+\s+[A-Z][a-z]*\.?\s*\d+[a-z]*\b'
        concepts["citations"] = re.findall(citation_pattern, text)
        
        # Statute patterns
        statute_pattern = r'\b\d+\s+U\.?S\.?C\.?\s*§?\s*\d+\b'
        concepts["statutes"].extend(re.findall(statute_pattern, text))
        
        # Remove duplicates
        for key in concepts:
            concepts[key] = list(set(concepts[key]))
        
        return concepts
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "tokenizer_vocab_size": self.tokenizer.vocab_size,
            "model_type": self.model.config.model_type if hasattr(self.model, 'config') else "unknown",
            "ner_available": self.ner_pipeline is not None
        }