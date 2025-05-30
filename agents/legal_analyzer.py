from typing import Dict, Any
import spacy
from langchain_core.messages import SystemMessage
import logging

from agents.base import AgentState
from agents.workflow_tracking_mixin import WorkflowTrackingMixin
from db.models import AgentType

logger = logging.getLogger(__name__)


class LegalAnalyzerAgent(WorkflowTrackingMixin):
    def __init__(self):
        self.name = "LegalAnalyzer"
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
    
    async def process(self, state: AgentState) -> Dict[str, Any]:
        """Process document for legal analysis with workflow tracking"""
        async with self.track_workflow("legal_analysis", AgentType.ANALYZER, state):
            try:
                document_text = state.get("current_document", "")
                legal_issues = []
                
                # Track keyword analysis task
                async with self.track_task("keyword_analysis", state):
                    keywords = [
                        "breach", "violation", "damages", "liability", "negligence",
                        "contract", "warranty", "habitability", "discrimination",
                        "retaliation", "harassment", "injury", "harm"
                    ]
                    
                    keyword_matches = []
                    for keyword in keywords:
                        if keyword.lower() in document_text.lower():
                            keyword_matches.append(keyword)
                            legal_issues.append(keyword)
                    
                    state["keyword_analysis_output"] = {
                        "keywords_checked": len(keywords),
                        "matches_found": len(keyword_matches),
                        "matched_keywords": keyword_matches
                    }
                
                # Track entity extraction task
                if self.nlp and document_text:
                    async with self.track_task("entity_extraction", state):
                        doc = self.nlp(document_text[:1000000])  # Limit for performance
                        
                        entities_found = []
                        for ent in doc.ents:
                            if ent.label_ in ["ORG", "PERSON", "LAW"]:
                                entity_info = f"{ent.label_}: {ent.text}"
                                entities_found.append(entity_info)
                                legal_issues.append(entity_info)
                        
                        state["entity_extraction_output"] = {
                            "entities_found": len(entities_found),
                            "entity_types": list(set([e.split(":")[0] for e in entities_found])),
                            "entities": entities_found[:20]  # Limit for storage
                        }
                
                # Prepare final output
                unique_issues = list(set(legal_issues))
                state["output_data"] = {
                    "legal_issues_count": len(unique_issues),
                    "legal_issues": unique_issues[:50]  # Limit for storage
                }
                
                return {
                    "legal_issues": unique_issues,
                    "messages": [
                        SystemMessage(
                            content=f"Identified {len(unique_issues)} potential legal issues"
                        )
                    ]
                }
            
            except Exception as e:
                logger.error(f"Legal analysis failed: {e}")
                return {
                    "error": f"Legal analysis failed: {str(e)}",
                    "messages": [
                        SystemMessage(content=f"Error in legal analysis: {str(e)}")
                    ]
                }
    
    def _calculate_quality_score(self, state: Dict[str, Any]) -> float:
        """Calculate quality score for legal analysis"""
        try:
            output_data = state.get("output_data", {})
            issues_count = output_data.get("legal_issues_count", 0)
            
            # Base score
            score = 0.5
            
            # Increase score based on issues found
            if issues_count > 0:
                score += min(0.3, issues_count * 0.03)  # Up to 0.3 for 10+ issues
            
            # Increase score if entities were extracted
            entity_output = state.get("entity_extraction_output", {})
            if entity_output.get("entities_found", 0) > 0:
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Failed to calculate quality score: {e}")
            return 0.5