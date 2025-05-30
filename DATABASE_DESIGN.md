# LeMCS Database Design Documentation

## Overview

This document outlines the comprehensive database schema for the Legal Memoranda Consolidation System (LeMCS), designed to support AI-powered consolidation of legal memoranda using the CRRACC methodology with semantic search capabilities via PostgreSQL and pgvector.

## Core Design Principles

1. **Multi-Agent Workflow Support**: Track LangGraph agent execution and coordination
2. **Semantic Search**: Vector embeddings for legal concept similarity and citation matching
3. **CRRACC Methodology**: Support for Conclusion-Rule-Rule explanation-Application-Counterargument-Conclusion structure
4. **Citation Integrity**: Preserve legal citation accuracy and cross-references
5. **Audit Trail**: Complete traceability for legal compliance
6. **Scalability**: Handle 25+ memoranda with 73+ provisions efficiently

## Enhanced Data Model

### Core Document Management

#### Documents Table (Enhanced)
- **Purpose**: Store legal memoranda and consolidated documents
- **Key Features**: Content hashing, semantic embeddings, legal document classification
- **Relationships**: One-to-many with citations, provisions, and analysis results

#### Document Provisions Table (New)
- **Purpose**: Extract and track individual lease provisions across memoranda
- **Key Features**: Provision text, classification, cross-references
- **CRRACC Support**: Link provisions to rule statements and applications

#### Legal Theories Table (New)
- **Purpose**: Support theory-based organization (habitability, Truth in Renting Act, etc.)
- **Key Features**: Theory taxonomy, precedent tracking
- **Integration**: Connect multiple provisions under unified legal frameworks

### Semantic Search Infrastructure

#### Document Embeddings Table (New)
- **Purpose**: Store vector embeddings for semantic similarity search
- **Technology**: pgvector extension for efficient similarity queries
- **Use Cases**: Find similar provisions, identify redundant content, suggest consolidation patterns

#### Citation Embeddings Table (New)
- **Purpose**: Vector representations of legal citations for similarity matching
- **Benefits**: Identify related cases across memoranda, detect citation patterns

### Enhanced Citation Management

#### Citations Table (Enhanced)
- **New Fields**: Jurisdiction, court level, precedential value, embedding vectors
- **Cross-References**: Links to related citations across documents
- **Validation**: Automated citation format checking and verification

#### Citation Relationships Table (New)
- **Purpose**: Track citation dependencies and hierarchies
- **Types**: "overrules", "distinguishes", "follows", "cites_approvingly"
- **Analysis**: Support comprehensive legal authority mapping

### Multi-Agent Workflow Tracking

#### Agent Workflows Table (New)
- **Purpose**: Track LangGraph multi-agent execution
- **Agents**: Orchestrator, Legal Analyzer, Citation Extractor, Synthesis Agent
- **Status**: Queued, running, completed, failed, retry
- **Performance**: Execution time, token usage, quality metrics

#### Agent Tasks Table (New)
- **Purpose**: Granular task tracking within agent workflows
- **Task Types**: document_analysis, citation_extraction, provision_identification, synthesis
- **Dependencies**: Task execution order and prerequisites
- **Results**: Structured outputs from each agent

### CRRACC Methodology Support

#### Analysis Frameworks Table (New)
- **Purpose**: Store CRRACC structural analysis
- **Components**: Conclusion, Rule, Rule Explanation, Application, Counterargument
- **Linking**: Connect framework components to source documents and provisions

#### Synthesis Results Table (New)
- **Purpose**: Store consolidated analysis outputs
- **Structure**: Theory-based organization with cross-provision integration
- **Quality**: Confidence scores, human review status, revision tracking

### Advanced Features

#### Legal Concepts Table (New)
- **Purpose**: Extract and categorize legal concepts (habitability, unconscionability, etc.)
- **NLP Integration**: Named entity recognition for legal terms
- **Relationships**: Concept hierarchies and relationships

#### Provision Mappings Table (New)
- **Purpose**: Map lease provisions to legal violations and remedies
- **Class Action Support**: Track commonality factors across provisions
- **Pattern Detection**: Identify systematic violations across documents

## PostgreSQL with pgvector Implementation

### Extension Setup
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
```

### Vector Search Optimization
- **Embedding Dimensions**: 1536 (OpenAI text-embedding-3-small)
- **Index Types**: HNSW and IVFFlat for different query patterns
- **Distance Metrics**: Cosine similarity for legal text comparison

### Performance Considerations
- **Chunking Strategy**: 512-token chunks with 50-token overlap
- **Batch Processing**: Vectorize documents in parallel
- **Index Maintenance**: Automatic reindexing on bulk updates

## Security and Compliance

### Data Protection
- **Encryption**: AES-256 for sensitive legal content
- **Access Control**: Role-based permissions (attorney, paralegal, admin)
- **Audit Logging**: Complete action trail for legal compliance

### GDPR/CCPA Compliance
- **Data Minimization**: Store only necessary legal analysis data
- **Right to Deletion**: Automated data purging workflows
- **Consent Tracking**: Document processing permissions

## Performance Requirements

### Scalability Targets
- **Document Volume**: 1000+ memoranda per consolidation job
- **Concurrent Users**: 50+ simultaneous legal professionals
- **Response Time**: <2 seconds for semantic search queries
- **Batch Processing**: 25 memoranda in <10 minutes

### Optimization Strategies
- **Connection Pooling**: PgBouncer for connection management
- **Read Replicas**: Query distribution for report generation
- **Partitioning**: Time-based partitioning for audit logs
- **Caching**: Redis integration for frequently accessed data

## Migration Strategy

### Phase 1: Core Enhancement
1. Add pgvector extension
2. Enhance existing tables with new fields
3. Create embedding infrastructure

### Phase 2: Advanced Features
1. Implement multi-agent workflow tracking
2. Add CRRACC methodology support
3. Create provision mapping system

### Phase 3: Optimization
1. Performance tuning and indexing
2. Advanced semantic search features
3. AI-powered consolidation suggestions

## Monitoring and Maintenance

### Health Metrics
- **Query Performance**: Slow query identification and optimization
- **Vector Search Accuracy**: Embedding quality and relevance scoring
- **Agent Performance**: Workflow execution time and success rates

### Backup Strategy
- **Daily Snapshots**: Full database backups with point-in-time recovery
- **Vector Index Backup**: Specialized backup for embedding indexes
- **Cross-Region Replication**: Disaster recovery capabilities