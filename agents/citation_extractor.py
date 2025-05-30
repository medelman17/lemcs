from typing import Dict, Any
import eyecite
from langchain_core.messages import SystemMessage

from agents.base import AgentState


class CitationExtractorAgent:
    def __init__(self):
        self.name = "CitationExtractor"
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        try:
            document_text = state.get("current_document", "")
            
            citations = list(eyecite.get_citations(document_text))
            
            extracted_citations = []
            for citation in citations:
                extracted_citations.append({
                    "text": citation.matched_text(),
                    "type": citation.__class__.__name__,
                    "metadata": {
                        "reporter": getattr(citation, "reporter", None),
                        "page": getattr(citation, "page", None),
                        "volume": getattr(citation, "volume", None)
                    }
                })
            
            return {
                "extracted_citations": extracted_citations,
                "messages": [
                    SystemMessage(
                        content=f"Extracted {len(extracted_citations)} citations from document"
                    )
                ]
            }
        
        except Exception as e:
            return {
                "error": f"Citation extraction failed: {str(e)}",
                "messages": [
                    SystemMessage(content=f"Error in citation extraction: {str(e)}")
                ]
            }