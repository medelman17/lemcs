"""
Workflow tracking mixin for agents.

Provides standardized workflow and task tracking functionality that can be 
mixed into any agent class for consistent monitoring and debugging.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from db.database import get_db
from db.models import (
    AgentWorkflow, AgentTask, WorkflowStatus, 
    AgentType
)

logger = logging.getLogger(__name__)


class WorkflowTrackingMixin:
    """
    Mixin class that provides workflow and task tracking capabilities to agents.
    
    Usage:
        class MyAgent(BaseAgent, WorkflowTrackingMixin):
            async def process(self, ...):
                async with self.track_workflow("my_workflow", AgentType.ANALYZER, state):
                    await self.track_task("parse_document", state):
                        # Do work here
                        pass
    """
    
    @asynccontextmanager
    async def track_workflow(self, workflow_type: str, agent_type: AgentType, 
                           state: Dict[str, Any], input_data: Optional[Dict] = None):
        """
        Context manager for tracking an entire workflow.
        
        Automatically handles workflow creation, completion, and failure.
        """
        workflow_id = None
        
        try:
            # Create workflow record
            async with get_db() as db_session:
                workflow = AgentWorkflow(
                    workflow_type=workflow_type,
                    agent_type=agent_type,
                    status=WorkflowStatus.RUNNING,
                    started_at=datetime.utcnow(),
                    input_data=input_data or {}
                )
                
                db_session.add(workflow)
                await db_session.commit()
                await db_session.refresh(workflow)
                
                workflow_id = workflow.id
                state["workflow_id"] = workflow_id
                logger.info(f"Started workflow: {workflow_type} (ID: {workflow_id})")
            
            # Yield control back to the agent
            yield workflow_id
            
            # Mark workflow as completed
            async with get_db() as db_session:
                workflow = await db_session.get(AgentWorkflow, workflow_id)
                if workflow:
                    workflow.status = WorkflowStatus.COMPLETED
                    workflow.completed_at = datetime.utcnow()
                    workflow.execution_time_ms = int(
                        (workflow.completed_at - workflow.started_at).total_seconds() * 1000
                    )
                    workflow.quality_score = self._calculate_quality_score(state) if hasattr(self, '_calculate_quality_score') else None
                    workflow.output_data = state.get("output_data", {})
                    
                    await db_session.commit()
                    logger.info(f"Completed workflow: {workflow_type} (ID: {workflow_id})")
                    
        except Exception as e:
            # Mark workflow as failed
            if workflow_id:
                try:
                    async with get_db() as db_session:
                        workflow = await db_session.get(AgentWorkflow, workflow_id)
                        if workflow:
                            workflow.status = WorkflowStatus.FAILED
                            workflow.completed_at = datetime.utcnow()
                            workflow.execution_time_ms = int(
                                (workflow.completed_at - workflow.started_at).total_seconds() * 1000
                            )
                            workflow.error_message = str(e)
                            
                            await db_session.commit()
                            logger.error(f"Failed workflow: {workflow_type} (ID: {workflow_id}) - {e}")
                except Exception as db_error:
                    logger.error(f"Failed to update workflow status: {db_error}")
            
            # Re-raise the original exception
            raise
    
    @asynccontextmanager
    async def track_task(self, task_type: str, state: Dict[str, Any], 
                        input_data: Optional[Dict] = None):
        """
        Context manager for tracking individual tasks within a workflow.
        
        Automatically handles task creation, completion, and failure.
        """
        workflow_id = state.get("workflow_id")
        if not workflow_id:
            logger.warning(f"No workflow_id in state, skipping task tracking for {task_type}")
            yield None
            return
        
        task_id = None
        
        try:
            # Create task record
            async with get_db() as db_session:
                # Get parent task if exists
                parent_task_id = state.get("current_task_id")
                
                task = AgentTask(
                    workflow_id=workflow_id,
                    task_type=task_type,
                    parent_task_id=parent_task_id,
                    status=WorkflowStatus.RUNNING,
                    started_at=datetime.utcnow(),
                    input_data=input_data or {}
                )
                
                db_session.add(task)
                await db_session.commit()
                await db_session.refresh(task)
                
                task_id = task.id
                # Store current task ID for potential subtasks
                old_task_id = state.get("current_task_id")
                state["current_task_id"] = task_id
                state[f"{task_type}_task_id"] = task_id
                logger.debug(f"Started task: {task_type} (ID: {task_id})")
            
            # Broadcast update if available
            try:
                from api.routes.agent_workflows import broadcast_task_update
                await broadcast_task_update(task_id, workflow_id, "RUNNING", task_type)
            except:
                pass  # Broadcasting is optional
            
            # Yield control back to the agent
            yield task_id
            
            # Mark task as completed
            async with get_db() as db_session:
                task = await db_session.get(AgentTask, task_id)
                if task:
                    task.status = WorkflowStatus.COMPLETED
                    task.completed_at = datetime.utcnow()
                    task.execution_time_ms = int(
                        (task.completed_at - task.started_at).total_seconds() * 1000
                    )
                    task.output_data = state.get(f"{task_type}_output", {})
                    
                    await db_session.commit()
                    logger.debug(f"Completed task: {task_type} (ID: {task_id})")
            
            # Restore previous task ID
            if old_task_id is not None:
                state["current_task_id"] = old_task_id
            else:
                state.pop("current_task_id", None)
            
            # Broadcast update if available
            try:
                from api.routes.agent_workflows import broadcast_task_update
                await broadcast_task_update(task_id, workflow_id, "COMPLETED", task_type)
            except:
                pass  # Broadcasting is optional
                
        except Exception as e:
            # Mark task as failed
            if task_id:
                try:
                    async with get_db() as db_session:
                        task = await db_session.get(AgentTask, task_id)
                        if task:
                            task.status = WorkflowStatus.FAILED
                            task.completed_at = datetime.utcnow()
                            task.execution_time_ms = int(
                                (task.completed_at - task.started_at).total_seconds() * 1000
                            )
                            task.error_message = str(e)
                            
                            await db_session.commit()
                            logger.error(f"Failed task: {task_type} (ID: {task_id}) - {e}")
                    
                    # Broadcast update if available
                    try:
                        from api.routes.agent_workflows import broadcast_task_update
                        await broadcast_task_update(task_id, workflow_id, "FAILED", task_type)
                    except:
                        pass  # Broadcasting is optional
                        
                except Exception as db_error:
                    logger.error(f"Failed to update task status: {db_error}")
            
            # Re-raise the original exception
            raise
    
    async def update_workflow_progress(self, workflow_id: int, progress_data: Dict[str, Any]):
        """Update workflow with progress information"""
        try:
            async with get_db() as db_session:
                workflow = await db_session.get(AgentWorkflow, workflow_id)
                if workflow:
                    # Merge progress data into output_data
                    if not workflow.output_data:
                        workflow.output_data = {}
                    workflow.output_data.update(progress_data)
                    
                    await db_session.commit()
                    logger.debug(f"Updated workflow progress: {workflow_id}")
                    
        except Exception as e:
            logger.error(f"Failed to update workflow progress: {e}")
    
    async def increment_workflow_retry(self, workflow_id: int) -> bool:
        """
        Increment workflow retry count and check if more retries are allowed.
        
        Returns True if retry is allowed, False otherwise.
        """
        try:
            async with get_db() as db_session:
                workflow = await db_session.get(AgentWorkflow, workflow_id)
                if workflow:
                    workflow.retry_count += 1
                    can_retry = workflow.retry_count < workflow.max_retries
                    
                    if can_retry:
                        workflow.status = WorkflowStatus.RUNNING
                        logger.info(f"Retrying workflow {workflow_id} (attempt {workflow.retry_count + 1}/{workflow.max_retries})")
                    
                    await db_session.commit()
                    return can_retry
                    
        except Exception as e:
            logger.error(f"Failed to update workflow retry count: {e}")
        
        return False