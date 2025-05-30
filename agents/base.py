from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_document: Optional[str]
    extracted_citations: List[dict]
    legal_issues: List[str]
    synthesis_results: Optional[dict]
    final_output: Optional[str]
    error: Optional[str]


class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.graph = StateGraph(AgentState)
    
    def build_graph(self):
        raise NotImplementedError("Subclasses must implement build_graph method")
    
    def compile(self):
        return self.graph.compile()