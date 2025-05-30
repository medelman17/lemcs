from sqlalchemy import Column, String, DateTime, Integer, Text, JSON, ForeignKey, Float, Boolean, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid
import enum

Base = declarative_base()


class DocumentType(enum.Enum):
    MEMORANDUM = "memorandum"
    LEASE_AGREEMENT = "lease_agreement"
    CONSOLIDATED = "consolidated"
    TEMPLATE = "template"


class DocumentStatus(enum.Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    CONSOLIDATED = "consolidated"
    ERROR = "error"


class AgentType(enum.Enum):
    ORCHESTRATOR = "orchestrator"
    LEGAL_ANALYZER = "legal_analyzer"
    CITATION_EXTRACTOR = "citation_extractor"
    SYNTHESIS_AGENT = "synthesis_agent"


class WorkflowStatus(enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    content_hash = Column(String(64))
    document_type = Column(Enum(DocumentType), default=DocumentType.MEMORANDUM)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.UPLOADED)
    
    # Enhanced metadata
    title = Column(String(500))
    author = Column(String(255))
    jurisdiction = Column(String(100))
    legal_area = Column(String(100))
    
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    
    doc_metadata = Column(JSON)
    extracted_text = Column(Text)
    
    # Relationships
    citations = relationship("Citation", back_populates="document")
    consolidation_documents = relationship("ConsolidationDocument", back_populates="document")
    provisions = relationship("DocumentProvision", back_populates="document")
    embeddings = relationship("DocumentEmbedding", back_populates="document")
    analysis_results = relationship("AnalysisResult", back_populates="document")


class Citation(Base):
    __tablename__ = "citations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    
    citation_text = Column(Text, nullable=False)
    citation_type = Column(String(50))
    reporter = Column(String(100))
    volume = Column(String(20))
    page = Column(String(20))
    
    position_start = Column(Integer)
    position_end = Column(Integer)
    confidence_score = Column(Float)
    
    doc_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    document = relationship("Document", back_populates="citations")


class ConsolidationJob(Base):
    __tablename__ = "consolidation_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String(50), default="queued")
    consolidation_method = Column(String(50), default="CRRACC")
    output_format = Column(String(20), default="docx")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    progress = Column(Integer, default=0)
    current_stage = Column(String(100))
    error_message = Column(Text)
    
    result_file_path = Column(String(500))
    options = Column(JSON)
    
    documents = relationship("ConsolidationDocument", back_populates="job")
    audit_logs = relationship("AuditLog", back_populates="consolidation_job")


class ConsolidationDocument(Base):
    __tablename__ = "consolidation_documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("consolidation_jobs.id"))
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    
    document_order = Column(Integer)
    included_in_final = Column(String(5), default="true")
    
    job = relationship("ConsolidationJob", back_populates="documents")
    document = relationship("Document", back_populates="consolidation_documents")


class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(UUID(as_uuid=True))
    
    consolidation_job_id = Column(UUID(as_uuid=True), ForeignKey("consolidation_jobs.id"))
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    details = Column(JSON)
    
    consolidation_job = relationship("ConsolidationJob", back_populates="audit_logs")


class DocumentProvision(Base):
    """Individual lease provisions extracted from documents"""
    __tablename__ = "document_provisions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    
    provision_text = Column(Text, nullable=False)
    provision_number = Column(String(20))
    provision_title = Column(String(255))
    provision_type = Column(String(100))  # e.g., "habitability", "rent", "termination"
    
    position_start = Column(Integer)
    position_end = Column(Integer)
    confidence_score = Column(Float)
    
    legal_theories = Column(JSON)  # Associated legal theories
    violation_indicators = Column(JSON)  # Potential legal violations
    
    created_at = Column(DateTime, default=datetime.utcnow)
    doc_metadata = Column(JSON)
    
    document = relationship("Document", back_populates="provisions")
    provision_mappings = relationship("ProvisionMapping", back_populates="provision")


class DocumentEmbedding(Base):
    """Vector embeddings for semantic search"""
    __tablename__ = "document_embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_start = Column(Integer)
    chunk_end = Column(Integer)
    
    embedding = Column(Vector(1536))  # OpenAI text-embedding-3-small dimensions
    
    created_at = Column(DateTime, default=datetime.utcnow)
    doc_metadata = Column(JSON)
    
    document = relationship("Document", back_populates="embeddings")


class LegalTheory(Base):
    """Legal theories and frameworks for consolidation"""
    __tablename__ = "legal_theories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    
    # Hierarchy support
    parent_theory_id = Column(UUID(as_uuid=True), ForeignKey("legal_theories.id"))
    
    # CRRACC components
    rule_statement = Column(Text)
    rule_explanation = Column(Text)
    
    jurisdiction = Column(String(100))
    precedent_strength = Column(String(50))  # "strong", "moderate", "weak"
    
    created_at = Column(DateTime, default=datetime.utcnow)
    doc_metadata = Column(JSON)
    
    # Self-referential relationship for hierarchy
    parent = relationship("LegalTheory", remote_side=[id], backref="children")
    provision_mappings = relationship("ProvisionMapping", back_populates="legal_theory")


class AgentWorkflow(Base):
    """Multi-agent workflow execution tracking"""
    __tablename__ = "agent_workflows"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    consolidation_job_id = Column(UUID(as_uuid=True), ForeignKey("consolidation_jobs.id"))
    
    agent_type = Column(Enum(AgentType), nullable=False)
    status = Column(Enum(WorkflowStatus), default=WorkflowStatus.QUEUED)
    
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    execution_time_ms = Column(Integer)
    
    input_data = Column(JSON)
    output_data = Column(JSON)
    error_message = Column(Text)
    
    # Performance metrics
    token_usage = Column(JSON)
    quality_score = Column(Float)
    retry_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    consolidation_job = relationship("ConsolidationJob")
    tasks = relationship("AgentTask", back_populates="workflow")


class AgentTask(Base):
    """Granular task tracking within agent workflows"""
    __tablename__ = "agent_tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("agent_workflows.id"))
    
    task_type = Column(String(100), nullable=False)  # "document_analysis", "citation_extraction", etc.
    task_name = Column(String(255))
    status = Column(Enum(WorkflowStatus), default=WorkflowStatus.QUEUED)
    
    # Task ordering and dependencies
    sequence_number = Column(Integer)
    depends_on_task_id = Column(UUID(as_uuid=True), ForeignKey("agent_tasks.id"))
    
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    input_data = Column(JSON)
    output_data = Column(JSON)
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    workflow = relationship("AgentWorkflow", back_populates="tasks")
    depends_on = relationship("AgentTask", remote_side=[id])


class AnalysisResult(Base):
    """Structured analysis results from AI agents"""
    __tablename__ = "analysis_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("agent_workflows.id"))
    
    analysis_type = Column(String(100), nullable=False)  # "crracc", "citation_extraction", etc.
    
    # CRRACC components
    conclusion = Column(Text)
    rule_statement = Column(Text)
    rule_explanation = Column(Text)
    application = Column(Text)
    counterargument = Column(Text)
    final_conclusion = Column(Text)
    
    confidence_score = Column(Float)
    human_reviewed = Column(Boolean, default=False)
    reviewer_notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    doc_metadata = Column(JSON)
    
    document = relationship("Document", back_populates="analysis_results")
    workflow = relationship("AgentWorkflow")


class ProvisionMapping(Base):
    """Map provisions to legal theories and violations"""
    __tablename__ = "provision_mappings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provision_id = Column(UUID(as_uuid=True), ForeignKey("document_provisions.id"))
    legal_theory_id = Column(UUID(as_uuid=True), ForeignKey("legal_theories.id"))
    
    violation_type = Column(String(255))
    severity = Column(String(50))  # "high", "medium", "low"
    remedies = Column(JSON)
    
    # Class action support
    commonality_factors = Column(JSON)
    pattern_strength = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    doc_metadata = Column(JSON)
    
    provision = relationship("DocumentProvision", back_populates="provision_mappings")
    legal_theory = relationship("LegalTheory", back_populates="provision_mappings")


class CitationEmbedding(Base):
    """Vector embeddings for legal citations"""
    __tablename__ = "citation_embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    citation_id = Column(UUID(as_uuid=True), ForeignKey("citations.id"))
    
    embedding = Column(Vector(1536))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    citation = relationship("Citation")


class CitationRelationship(Base):
    """Relationships between citations across documents"""
    __tablename__ = "citation_relationships"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_citation_id = Column(UUID(as_uuid=True), ForeignKey("citations.id"))
    target_citation_id = Column(UUID(as_uuid=True), ForeignKey("citations.id"))
    
    relationship_type = Column(String(50))  # "overrules", "distinguishes", "follows", "cites_approvingly"
    confidence_score = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    source_citation = relationship("Citation", foreign_keys=[source_citation_id])
    target_citation = relationship("Citation", foreign_keys=[target_citation_id])