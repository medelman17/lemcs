"""
Citation Extractor Agent for LangGraph multi-agent workflow.
Handles citation extraction, resolution, and analysis within the legal document processing pipeline.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langgraph.graph import Graph, END
from sqlalchemy.ext.asyncio import AsyncSession

from nlp.citation_service import citation_service, CitationExtractionResult, CitationAnalysis
from db.models import Document, Citation, AgentWorkflow, AgentTask, WorkflowStatus, AgentType
from config.settings import settings

logger = logging.getLogger(__name__)


class CitationExtractorAgent:
    """
    LangGraph agent specialized in legal citation extraction and analysis.
    
    Responsibilities:
    - Extract citations from legal documents using eyecite
    - Resolve reference citations (supra, id.) to their antecedents
    - Analyze citation authority and precedential strength
    - Create citation embeddings for semantic search
    - Track extraction metrics and quality
    """
    
    def __init__(self):
        self.agent_type = AgentType.CITATION_EXTRACTOR
        self.graph = self._build_graph()
    
    def _build_graph(self) -> Graph:
        """Build the LangGraph workflow for citation extraction"""
        graph = Graph()
        
        # Add nodes for each step of citation processing
        graph.add_node("extract_citations", self.extract_citations)
        graph.add_node("resolve_references", self.resolve_references)
        graph.add_node("analyze_authority", self.analyze_authority)
        graph.add_node("create_embeddings", self.create_embeddings)
        graph.add_node("finalize_results", self.finalize_results)
        
        # Define the workflow sequence
        graph.add_edge("extract_citations", "resolve_references")
        graph.add_edge("resolve_references", "analyze_authority")
        graph.add_edge("analyze_authority", "create_embeddings")
        graph.add_edge("create_embeddings", "finalize_results")
        graph.add_edge("finalize_results", END)
        
        # Set entry point
        graph.set_entry_point("extract_citations")
        
        return graph
    
    async def process_document(
        self, 
        document: Document, 
        db_session: AsyncSession,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing a document through the citation extraction workflow
        
        Args:
            document: Document to process
            db_session: Database session
            options: Processing options (resolve_references, create_embeddings, etc.)
            
        Returns:
            Complete workflow results with extracted citations and analysis
        """
        workflow_id = await self._start_workflow(document.id, db_session)
        
        try:
            # Initialize state
            initial_state = {
                "document": document,
                "db_session": db_session,
                "workflow_id": workflow_id,
                "options": options or {},
                "citations": [],
                "extraction_result": None,
                "authority_analysis": [],
                "embeddings_created": False,
                "errors": [],
                "metrics": {}
            }
            
            # Run the graph
            compiled_graph = self.graph.compile()
            final_state = await compiled_graph.ainvoke(initial_state)
            
            # Mark workflow as completed
            await self._complete_workflow(workflow_id, final_state, db_session)
            
            return final_state
            
        except Exception as e:
            logger.error(f"Citation extraction workflow failed: {e}")
            await self._fail_workflow(workflow_id, str(e), db_session)
            raise
    
    async def extract_citations(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all citations from the document text"""
        await self._start_task("extract_citations", state)
        
        try:
            document = state["document"]
            db_session = state["db_session"]
            options = state["options"]
            
            # Extract citations using the citation service
            extraction_result = await citation_service.extract_citations_from_document(
                document=document,
                db_session=db_session,
                resolve_references=options.get("resolve_references", True),
                create_embeddings=False,  # Handle separately for better control
                create_relationships=options.get("create_relationships", True)
            )
            
            # Update state
            state["citations"] = extraction_result.citations
            state["extraction_result"] = extraction_result
            state["metrics"]["extraction_time_ms"] = extraction_result.processing_time_ms
            state["metrics"]["total_citations"] = len(extraction_result.citations)
            state["metrics"]["relationships_created"] = extraction_result.extraction_stats.get("relationships_created", 0)
            
            if extraction_result.errors:
                state["errors"].extend(extraction_result.errors)
            
            await self._complete_task("extract_citations", state, {
                "citations_found": len(extraction_result.citations),
                "relationships_created": state["metrics"]["relationships_created"],
                "extraction_stats": extraction_result.extraction_stats
            })
            
            logger.info(f"Extracted {len(extraction_result.citations)} citations "
                       f"and created {state['metrics']['relationships_created']} relationships "
                       f"from document {document.id}")
            
            return state
            
        except Exception as e:
            await self._fail_task("extract_citations", state, str(e))
            raise
    
    async def resolve_references(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve reference citations to their antecedents"""
        await self._start_task("resolve_references", state)
        
        try:
            citations = state["citations"]
            
            # Count citations with antecedent information
            resolved_count = 0
            for citation in citations:
                if citation.doc_metadata and "antecedent" in citation.doc_metadata:
                    resolved_count += 1
            
            state["metrics"]["resolved_citations"] = resolved_count
            state["metrics"]["resolution_rate"] = resolved_count / len(citations) if citations else 0
            
            await self._complete_task("resolve_references", state, {
                "resolved_count": resolved_count,
                "total_citations": len(citations)
            })
            
            logger.info(f"Resolved {resolved_count} reference citations")
            
            return state
            
        except Exception as e:
            await self._fail_task("resolve_references", state, str(e))
            raise
    
    async def analyze_authority(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the precedential authority of extracted citations"""
        await self._start_task("analyze_authority", state)
        
        try:
            citations = state["citations"]
            authority_analysis = []
            
            # Analyze each citation's authority
            for citation in citations:
                analysis = await citation_service.analyze_citation_authority(citation)
                authority_analysis.append(analysis)
            
            # Calculate authority metrics
            authority_scores = [a.authority_score for a in authority_analysis if a.authority_score]
            avg_authority = sum(authority_scores) / len(authority_scores) if authority_scores else 0
            
            high_authority_count = len([s for s in authority_scores if s and s > 0.7])
            
            state["authority_analysis"] = authority_analysis
            state["metrics"]["avg_authority_score"] = avg_authority
            state["metrics"]["high_authority_citations"] = high_authority_count
            
            await self._complete_task("analyze_authority", state, {
                "analyzed_citations": len(authority_analysis),
                "avg_authority_score": avg_authority,
                "high_authority_count": high_authority_count
            })
            
            logger.info(f"Analyzed authority for {len(authority_analysis)} citations")
            
            return state
            
        except Exception as e:
            await self._fail_task("analyze_authority", state, str(e))
            raise
    
    async def create_embeddings(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create vector embeddings for citations (when OpenAI is configured)"""
        await self._start_task("create_embeddings", state)
        
        try:
            citations = state["citations"]
            options = state["options"]
            
            if options.get("create_embeddings", False) and settings.OPENAI_API_KEY:
                # Import OpenAI service and create embeddings
                from nlp.openai_service import openai_service
                
                embedding_results = await openai_service.create_citation_embeddings(
                    citations=citations,
                    include_context=True
                )
                
                # Create CitationEmbedding records in database
                db_session = state["db_session"]
                citation_embeddings = []
                
                for citation, embedding_result in zip(citations, embedding_results):
                    from db.models import CitationEmbedding
                    citation_embedding = CitationEmbedding(
                        citation_id=citation.id,
                        embedding=embedding_result.embedding,
                        created_at=datetime.utcnow()
                    )
                    citation_embeddings.append(citation_embedding)
                    db_session.add(citation_embedding)
                
                await db_session.commit()
                
                # Update state with embedding information
                state["embeddings_created"] = True
                state["embedding_count"] = len(citation_embeddings)
                state["embedding_stats"] = {
                    "total_tokens": sum(r.tokens_used for r in embedding_results),
                    "cached_count": sum(1 for r in embedding_results if r.cached),
                    "total_cost_usd": sum(getattr(r, 'cost_usd', 0) for r in embedding_results)
                }
                
                logger.info(f"Created embeddings for {len(citation_embeddings)} citations")
                
            else:
                logger.info("Embedding creation skipped (not requested or API key not configured)")
                state["embeddings_created"] = False
                state["embedding_count"] = 0
            
            await self._complete_task("create_embeddings", state, {
                "embeddings_created": state["embeddings_created"],
                "citations_count": len(citations),
                "embedding_count": state.get("embedding_count", 0)
            })
            
            return state
            
        except Exception as e:
            await self._fail_task("create_embeddings", state, str(e))
            raise
    
    async def finalize_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize the citation extraction results and prepare output"""
        await self._start_task("finalize_results", state)
        
        try:
            # Compile final results
            final_results = {
                "document_id": str(state["document"].id),
                "citations": [
                    {
                        "id": str(c.id),
                        "text": c.citation_text,
                        "type": c.citation_type,
                        "reporter": c.reporter,
                        "volume": c.volume,
                        "page": c.page,
                        "position": [c.position_start, c.position_end],
                        "confidence": c.confidence_score
                    }
                    for c in state["citations"]
                ],
                "authority_analysis": [
                    {
                        "citation_id": str(a.citation.id),
                        "court_level": a.court_level,
                        "jurisdiction": a.jurisdiction,
                        "precedential_strength": a.precedential_strength,
                        "authority_score": a.authority_score
                    }
                    for a in state["authority_analysis"]
                ],
                "metrics": state["metrics"],
                "errors": state["errors"]
            }
            
            state["final_results"] = final_results
            
            await self._complete_task("finalize_results", state, final_results)
            
            logger.info(f"Citation extraction completed for document {state['document'].id}")
            
            return state
            
        except Exception as e:
            await self._fail_task("finalize_results", state, str(e))
            raise
    
    async def _start_workflow(self, document_id: str, db_session: AsyncSession) -> str:
        """Start a new workflow tracking record"""
        workflow = AgentWorkflow(
            consolidation_job_id=None,  # Will be set when part of larger job
            agent_type=AgentType.CITATION_EXTRACTOR,
            status=WorkflowStatus.RUNNING,
            started_at=datetime.utcnow(),
            input_data={"document_id": str(document_id)},
            retry_count=0
        )
        
        db_session.add(workflow)
        await db_session.commit()
        await db_session.refresh(workflow)
        
        return workflow.id
    
    async def _complete_workflow(self, workflow_id: str, final_state: Dict[str, Any], db_session: AsyncSession):
        """Mark workflow as completed"""
        workflow = await db_session.get(AgentWorkflow, workflow_id)
        if workflow:
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.utcnow()
            workflow.execution_time_ms = int(
                (workflow.completed_at - workflow.started_at).total_seconds() * 1000
            )
            workflow.output_data = final_state.get("final_results", {})
            workflow.quality_score = self._calculate_quality_score(final_state)
            
            await db_session.commit()
    
    async def _fail_workflow(self, workflow_id: str, error_message: str, db_session: AsyncSession):
        """Mark workflow as failed"""
        workflow = await db_session.get(AgentWorkflow, workflow_id)
        if workflow:
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.utcnow()
            workflow.error_message = error_message
            
            await db_session.commit()
    
    async def _start_task(self, task_type: str, state: Dict[str, Any]) -> Optional[str]:
        """Start tracking a specific task"""
        workflow_id = state.get("workflow_id")
        if not workflow_id:
            logger.warning(f"No workflow_id in state, skipping task tracking for {task_type}")
            return None
        
        try:
            from db.database import get_session
            from db.models import AgentTask, AgentTaskStatus
            
            async with get_session() as db_session:
                # Get parent task if exists
                parent_task_id = state.get("current_task_id")
                
                task = AgentTask(
                    workflow_id=workflow_id,
                    task_type=task_type,
                    parent_task_id=parent_task_id,
                    status=AgentTaskStatus.RUNNING,
                    started_at=datetime.utcnow(),
                    input_data={
                        "document_id": state.get("document_id"),
                        "page_count": state.get("page_count", 0),
                        "current_page": state.get("current_page", 0)
                    }
                )
                
                db_session.add(task)
                await db_session.commit()
                await db_session.refresh(task)
                
                # Store task ID in state for later reference
                state[f"{task_type}_task_id"] = task.id
                logger.debug(f"Started task: {task_type} (ID: {task.id})")
                return task.id
                
        except Exception as e:
            logger.error(f"Failed to create task record: {e}")
            return None
    
    async def _complete_task(self, task_type: str, state: Dict[str, Any], output_data: Dict[str, Any]):
        """Complete a specific task"""
        task_id = state.get(f"{task_type}_task_id")
        if not task_id:
            logger.debug(f"No task ID for {task_type}, skipping task completion tracking")
            return
        
        try:
            from db.database import get_session
            from db.models import AgentTask, AgentTaskStatus
            
            async with get_session() as db_session:
                task = await db_session.get(AgentTask, task_id)
                if task:
                    task.status = AgentTaskStatus.COMPLETED
                    task.completed_at = datetime.utcnow()
                    task.output_data = output_data
                    
                    # Calculate execution time
                    if task.started_at:
                        task.execution_time_ms = int((task.completed_at - task.started_at).total_seconds() * 1000)
                    
                    await db_session.commit()
                    logger.debug(f"Completed task: {task_type} (ID: {task.id})")
                    
        except Exception as e:
            logger.error(f"Failed to update task record: {e}")
    
    async def _fail_task(self, task_type: str, state: Dict[str, Any], error_message: str):
        """Mark a task as failed"""
        task_id = state.get(f"{task_type}_task_id")
        if not task_id:
            logger.error(f"Failed task: {task_type} - {error_message}")
            return
        
        try:
            from db.database import get_session
            from db.models import AgentTask, AgentTaskStatus
            
            async with get_session() as db_session:
                task = await db_session.get(AgentTask, task_id)
                if task:
                    task.status = AgentTaskStatus.FAILED
                    task.completed_at = datetime.utcnow()
                    task.error_message = error_message
                    
                    # Calculate execution time
                    if task.started_at:
                        task.execution_time_ms = int((task.completed_at - task.started_at).total_seconds() * 1000)
                    
                    await db_session.commit()
                    logger.error(f"Failed task: {task_type} (ID: {task.id}) - {error_message}")
                    
        except Exception as e:
            logger.error(f"Failed to update task record: {e}")
    
    def _calculate_quality_score(self, final_state: Dict[str, Any]) -> float:
        """Calculate overall quality score for the extraction"""
        try:
            metrics = final_state.get("metrics", {})
            
            # Base score
            score = 0.7
            
            # Increase score based on citation count
            citation_count = metrics.get("total_citations", 0)
            if citation_count > 0:
                score += 0.1
            if citation_count > 5:
                score += 0.1
            
            # Increase score for high authority citations
            high_authority_count = metrics.get("high_authority_citations", 0)
            if high_authority_count > 0:
                score += 0.1
            
            # Decrease score for errors
            error_count = len(final_state.get("errors", []))
            if error_count > 0:
                score -= 0.1 * error_count
            
            return max(0.1, min(1.0, score))
            
        except Exception:
            return 0.5  # Default score if calculation fails


# Global agent instance
citation_extractor_agent = CitationExtractorAgent()