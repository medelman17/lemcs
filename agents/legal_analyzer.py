from typing import Dict, Any
import spacy
from langchain_core.messages import SystemMessage

from agents.base import AgentState


class LegalAnalyzerAgent:
    def __init__(self):
        self.name = "LegalAnalyzer"
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        try:
            document_text = state.get("current_document", "")
            
            legal_issues = []
            
            keywords = [
                "breach", "violation", "damages", "liability", "negligence",
                "contract", "warranty", "habitability", "discrimination",
                "retaliation", "harassment", "injury", "harm"
            ]
            
            for keyword in keywords:
                if keyword.lower() in document_text.lower():
                    legal_issues.append(keyword)
            
            if self.nlp and document_text:
                doc = self.nlp(document_text[:1000000])  # Limit for performance
                
                for ent in doc.ents:
                    if ent.label_ in ["ORG", "PERSON", "LAW"]:
                        legal_issues.append(f"{ent.label_}: {ent.text}")
            
            return {
                "legal_issues": list(set(legal_issues)),
                "messages": [
                    SystemMessage(
                        content=f"Identified {len(legal_issues)} potential legal issues"
                    )
                ]
            }
        
        except Exception as e:
            return {
                "error": f"Legal analysis failed: {str(e)}",
                "messages": [
                    SystemMessage(content=f"Error in legal analysis: {str(e)}")
                ]
            }