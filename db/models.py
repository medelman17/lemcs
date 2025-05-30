from sqlalchemy import Column, String, DateTime, Integer, Text, JSON, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    content_hash = Column(String(64))
    document_type = Column(String(50))
    status = Column(String(50), default="uploaded")
    
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    
    metadata = Column(JSON)
    extracted_text = Column(Text)
    
    citations = relationship("Citation", back_populates="document")
    consolidation_documents = relationship("ConsolidationDocument", back_populates="document")


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
    
    metadata = Column(JSON)
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