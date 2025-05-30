"""
Agent Workflow API endpoints for monitoring and managing agent tasks.

This module provides REST endpoints for:
- Listing and querying agent workflows
- Retrieving workflow details and tasks
- Monitoring agent performance metrics
- Real-time workflow updates (WebSocket)
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, Field
import asyncio
import json

from db.database import get_session
from db.models import (
    AgentWorkflow, AgentTask, AgentWorkflowStatus, 
    AgentTaskStatus, AgentType
)

router = APIRouter(prefix="/workflows", tags=["Agent Workflows"])

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Connection might be closed, ignore
                pass

manager = ConnectionManager()

# Pydantic models
class WorkflowSummary(BaseModel):
    """Summary information for a workflow"""
    id: int
    workflow_type: str
    agent_type: AgentType
    status: AgentWorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime]
    execution_time_ms: Optional[int]
    quality_score: Optional[float]
    error_message: Optional[str]
    task_count: int = 0
    completed_task_count: int = 0

class TaskDetail(BaseModel):
    """Detailed information about a task"""
    id: int
    workflow_id: int
    task_type: str
    status: AgentTaskStatus
    started_at: datetime
    completed_at: Optional[datetime]
    execution_time_ms: Optional[int]
    parent_task_id: Optional[int]
    error_message: Optional[str]
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)

class WorkflowDetail(BaseModel):
    """Detailed workflow information including tasks"""
    id: int
    workflow_type: str
    agent_type: AgentType
    status: AgentWorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime]
    execution_time_ms: Optional[int]
    quality_score: Optional[float]
    error_message: Optional[str]
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    tasks: List[TaskDetail] = Field(default_factory=list)

class AgentMetrics(BaseModel):
    """Performance metrics for agents"""
    agent_type: AgentType
    total_workflows: int
    completed_workflows: int
    failed_workflows: int
    avg_execution_time_ms: float
    avg_quality_score: float
    success_rate: float
    total_tasks: int
    avg_tasks_per_workflow: float

class WorkflowFilter(BaseModel):
    """Filter parameters for workflow queries"""
    agent_type: Optional[AgentType] = None
    status: Optional[AgentWorkflowStatus] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_quality_score: Optional[float] = None
    max_quality_score: Optional[float] = None

# API Endpoints
@router.get("/", response_model=List[WorkflowSummary])
async def list_workflows(
    agent_type: Optional[AgentType] = Query(None, description="Filter by agent type"),
    status: Optional[AgentWorkflowStatus] = Query(None, description="Filter by status"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date (after)"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date (before)"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: AsyncSession = Depends(get_session)
) -> List[WorkflowSummary]:
    """
    List agent workflows with optional filtering.
    
    Returns a paginated list of workflow summaries with task counts.
    """
    # Build query with filters
    query = select(AgentWorkflow)
    
    if agent_type:
        query = query.where(AgentWorkflow.agent_type == agent_type)
    if status:
        query = query.where(AgentWorkflow.status == status)
    if start_date:
        query = query.where(AgentWorkflow.started_at >= start_date)
    if end_date:
        query = query.where(AgentWorkflow.started_at <= end_date)
    
    # Order by most recent first
    query = query.order_by(AgentWorkflow.started_at.desc())
    query = query.limit(limit).offset(offset)
    
    # Execute query
    result = await db.execute(query)
    workflows = result.scalars().all()
    
    # Get task counts for each workflow
    workflow_ids = [w.id for w in workflows]
    if workflow_ids:
        task_counts_query = select(
            AgentTask.workflow_id,
            func.count(AgentTask.id).label('total_count'),
            func.sum(
                func.cast(AgentTask.status == AgentTaskStatus.COMPLETED, type_=func.Integer)
            ).label('completed_count')
        ).where(
            AgentTask.workflow_id.in_(workflow_ids)
        ).group_by(AgentTask.workflow_id)
        
        task_counts_result = await db.execute(task_counts_query)
        task_counts = {
            row.workflow_id: {
                'total': row.total_count,
                'completed': row.completed_count or 0
            }
            for row in task_counts_result
        }
    else:
        task_counts = {}
    
    # Build response
    summaries = []
    for workflow in workflows:
        counts = task_counts.get(workflow.id, {'total': 0, 'completed': 0})
        summaries.append(WorkflowSummary(
            id=workflow.id,
            workflow_type=workflow.workflow_type,
            agent_type=workflow.agent_type,
            status=workflow.status,
            started_at=workflow.started_at,
            completed_at=workflow.completed_at,
            execution_time_ms=workflow.execution_time_ms,
            quality_score=workflow.quality_score,
            error_message=workflow.error_message,
            task_count=counts['total'],
            completed_task_count=counts['completed']
        ))
    
    return summaries

@router.get("/{workflow_id}", response_model=WorkflowDetail)
async def get_workflow_detail(
    workflow_id: int,
    include_tasks: bool = Query(True, description="Include task details"),
    db: AsyncSession = Depends(get_session)
) -> WorkflowDetail:
    """
    Get detailed information about a specific workflow.
    
    Includes all workflow data and optionally all associated tasks.
    """
    # Get workflow with tasks if requested
    query = select(AgentWorkflow).where(AgentWorkflow.id == workflow_id)
    if include_tasks:
        query = query.options(selectinload(AgentWorkflow.tasks))
    
    result = await db.execute(query)
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Build response
    detail = WorkflowDetail(
        id=workflow.id,
        workflow_type=workflow.workflow_type,
        agent_type=workflow.agent_type,
        status=workflow.status,
        started_at=workflow.started_at,
        completed_at=workflow.completed_at,
        execution_time_ms=workflow.execution_time_ms,
        quality_score=workflow.quality_score,
        error_message=workflow.error_message,
        input_data=workflow.input_data or {},
        output_data=workflow.output_data or {},
        retry_count=workflow.retry_count,
        max_retries=workflow.max_retries
    )
    
    if include_tasks and hasattr(workflow, 'tasks'):
        detail.tasks = [
            TaskDetail(
                id=task.id,
                workflow_id=task.workflow_id,
                task_type=task.task_type,
                status=task.status,
                started_at=task.started_at,
                completed_at=task.completed_at,
                execution_time_ms=task.execution_time_ms,
                parent_task_id=task.parent_task_id,
                error_message=task.error_message,
                input_data=task.input_data or {},
                output_data=task.output_data or {}
            )
            for task in workflow.tasks
        ]
    
    return detail

@router.get("/{workflow_id}/tasks", response_model=List[TaskDetail])
async def get_workflow_tasks(
    workflow_id: int,
    status: Optional[AgentTaskStatus] = Query(None, description="Filter by task status"),
    db: AsyncSession = Depends(get_session)
) -> List[TaskDetail]:
    """
    Get all tasks for a specific workflow.
    
    Returns tasks in execution order (by started_at).
    """
    # Build query
    query = select(AgentTask).where(AgentTask.workflow_id == workflow_id)
    
    if status:
        query = query.where(AgentTask.status == status)
    
    query = query.order_by(AgentTask.started_at)
    
    # Execute query
    result = await db.execute(query)
    tasks = result.scalars().all()
    
    # Build response
    return [
        TaskDetail(
            id=task.id,
            workflow_id=task.workflow_id,
            task_type=task.task_type,
            status=task.status,
            started_at=task.started_at,
            completed_at=task.completed_at,
            execution_time_ms=task.execution_time_ms,
            parent_task_id=task.parent_task_id,
            error_message=task.error_message,
            input_data=task.input_data or {},
            output_data=task.output_data or {}
        )
        for task in tasks
    ]

@router.get("/metrics/summary", response_model=List[AgentMetrics])
async def get_agent_metrics(
    start_date: Optional[datetime] = Query(None, description="Start date for metrics"),
    end_date: Optional[datetime] = Query(None, description="End date for metrics"),
    db: AsyncSession = Depends(get_session)
) -> List[AgentMetrics]:
    """
    Get performance metrics for all agent types.
    
    Returns aggregated metrics including success rates, execution times, and quality scores.
    """
    # Build base query
    conditions = []
    if start_date:
        conditions.append(AgentWorkflow.started_at >= start_date)
    if end_date:
        conditions.append(AgentWorkflow.started_at <= end_date)
    
    # Get workflow metrics by agent type
    workflow_query = select(
        AgentWorkflow.agent_type,
        func.count(AgentWorkflow.id).label('total'),
        func.sum(
            func.cast(AgentWorkflow.status == AgentWorkflowStatus.COMPLETED, type_=func.Integer)
        ).label('completed'),
        func.sum(
            func.cast(AgentWorkflow.status == AgentWorkflowStatus.FAILED, type_=func.Integer)
        ).label('failed'),
        func.avg(AgentWorkflow.execution_time_ms).label('avg_time'),
        func.avg(AgentWorkflow.quality_score).label('avg_quality')
    ).group_by(AgentWorkflow.agent_type)
    
    if conditions:
        workflow_query = workflow_query.where(and_(*conditions))
    
    workflow_result = await db.execute(workflow_query)
    workflow_metrics = {
        row.agent_type: {
            'total': row.total,
            'completed': row.completed or 0,
            'failed': row.failed or 0,
            'avg_time': row.avg_time or 0,
            'avg_quality': row.avg_quality or 0
        }
        for row in workflow_result
    }
    
    # Get task metrics by agent type
    task_query = select(
        AgentWorkflow.agent_type,
        func.count(AgentTask.id).label('total_tasks')
    ).join(
        AgentTask, AgentTask.workflow_id == AgentWorkflow.id
    ).group_by(AgentWorkflow.agent_type)
    
    if conditions:
        task_query = task_query.where(and_(*conditions))
    
    task_result = await db.execute(task_query)
    task_metrics = {
        row.agent_type: row.total_tasks
        for row in task_result
    }
    
    # Build response
    metrics = []
    for agent_type in AgentType:
        wf_metrics = workflow_metrics.get(agent_type, {
            'total': 0, 'completed': 0, 'failed': 0, 'avg_time': 0, 'avg_quality': 0
        })
        total_tasks = task_metrics.get(agent_type, 0)
        
        metrics.append(AgentMetrics(
            agent_type=agent_type,
            total_workflows=wf_metrics['total'],
            completed_workflows=wf_metrics['completed'],
            failed_workflows=wf_metrics['failed'],
            avg_execution_time_ms=float(wf_metrics['avg_time']),
            avg_quality_score=float(wf_metrics['avg_quality']),
            success_rate=float(wf_metrics['completed'] / wf_metrics['total']) if wf_metrics['total'] > 0 else 0,
            total_tasks=total_tasks,
            avg_tasks_per_workflow=float(total_tasks / wf_metrics['total']) if wf_metrics['total'] > 0 else 0
        ))
    
    return metrics

@router.websocket("/ws")
async def workflow_updates(websocket: WebSocket):
    """
    WebSocket endpoint for real-time workflow updates.
    
    Clients can connect to receive live updates about workflow and task status changes.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Helper function to broadcast workflow updates
async def broadcast_workflow_update(workflow_id: int, status: str, message: str = ""):
    """Broadcast workflow status update to all connected WebSocket clients"""
    update = {
        "type": "workflow_update",
        "workflow_id": workflow_id,
        "status": status,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast(json.dumps(update))

# Helper function to broadcast task updates
async def broadcast_task_update(task_id: int, workflow_id: int, status: str, task_type: str):
    """Broadcast task status update to all connected WebSocket clients"""
    update = {
        "type": "task_update",
        "task_id": task_id,
        "workflow_id": workflow_id,
        "task_type": task_type,
        "status": status,
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast(json.dumps(update))