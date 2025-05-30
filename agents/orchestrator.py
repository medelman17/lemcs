from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import AgentState, BaseAgent
from agents.citation_extractor import CitationExtractorAgent
from agents.legal_analyzer import LegalAnalyzerAgent
from agents.synthesis_agent import SynthesisAgent


class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("Orchestrator")
        self.citation_extractor = CitationExtractorAgent()
        self.legal_analyzer = LegalAnalyzerAgent()
        self.synthesis_agent = SynthesisAgent()
    
    def route_next_step(self, state: AgentState) -> str:
        if state.get("error"):
            return "handle_error"
        
        if not state.get("extracted_citations"):
            return "extract_citations"
        
        if not state.get("legal_issues"):
            return "analyze_legal_issues"
        
        if not state.get("synthesis_results"):
            return "synthesize_content"
        
        return "finalize_output"
    
    def initialize_process(self, state: AgentState) -> Dict[str, Any]:
        return {
            "messages": [
                SystemMessage(content="Starting legal memoranda consolidation process")
            ]
        }
    
    def handle_error(self, state: AgentState) -> Dict[str, Any]:
        return {
            "messages": [
                SystemMessage(content=f"Error occurred: {state.get('error')}")
            ]
        }
    
    def finalize_output(self, state: AgentState) -> Dict[str, Any]:
        return {
            "final_output": "Consolidation complete",
            "messages": [
                SystemMessage(content="Legal memoranda consolidation completed successfully")
            ]
        }
    
    def build_graph(self):
        self.graph.add_node("initialize", self.initialize_process)
        self.graph.add_node("extract_citations", self.citation_extractor.process)
        self.graph.add_node("analyze_legal_issues", self.legal_analyzer.process)
        self.graph.add_node("synthesize_content", self.synthesis_agent.process)
        self.graph.add_node("finalize_output", self.finalize_output)
        self.graph.add_node("handle_error", self.handle_error)
        
        self.graph.set_entry_point("initialize")
        
        self.graph.add_conditional_edges(
            "initialize",
            self.route_next_step,
            {
                "extract_citations": "extract_citations",
                "handle_error": "handle_error"
            }
        )
        
        self.graph.add_conditional_edges(
            "extract_citations",
            self.route_next_step,
            {
                "analyze_legal_issues": "analyze_legal_issues",
                "handle_error": "handle_error"
            }
        )
        
        self.graph.add_conditional_edges(
            "analyze_legal_issues",
            self.route_next_step,
            {
                "synthesize_content": "synthesize_content",
                "handle_error": "handle_error"
            }
        )
        
        self.graph.add_conditional_edges(
            "synthesize_content",
            self.route_next_step,
            {
                "finalize_output": "finalize_output",
                "handle_error": "handle_error"
            }
        )
        
        self.graph.add_edge("finalize_output", END)
        self.graph.add_edge("handle_error", END)
        
        return self