---
description: Global coding standards and architecture principles for LeMCS legal document processing system
alwaysApply: true
---

# LeMCS Global Standards

LeMCS is an AI-powered legal document processing platform that consolidates multiple legal memoranda into comprehensive omnibus documents using a multi-agent architecture.

## Core Architecture Principles

### Multi-Agent System (LangGraph)
- Use LangGraph for orchestrating agent workflows
- Each agent has specific legal domain expertise
- Agents communicate through structured message passing
- Maintain state consistency across agent interactions

### Database Architecture (PostgreSQL + pgvector)
- Use AsyncPG for database connections
- pgvector extension for semantic search_files capabilities
- 15-table schema for comprehensive legal document storage
- Embed legal document vectors using LEGAL-BERT embeddings

### API Design (FastAPI)
- Async/await patterns throughout
- Pydantic models for data validation
- Comprehensive error handling with legal-specific exceptions
- OpenAPI documentation with legal domain examples

## Development Standards

### Code Quality
- Follow Python 3.12+ best practices
- Use type hints throughout
- Async/await for all I/O operations
- Black for code formatting
- Never commit secrets or API keys
- Use environment variables for sensitive configuration

### Performance Targets
- Process 25 memoranda in under 10 minutes
- >99% citation extraction accuracy (achieved with eyecite)
- Sub-second entity extraction per page
- Support concurrent processing of 10+ consolidation jobs

### Security Requirements
- Implement proper input validation for legal documents
- Audit logging for all legal document operations
- GDPR/CCPA compliance for PII handling
- Graceful error handling for production legal systems

## File Organization
```
/nlp/                     # Legal NLP services
/agents/                  # LangGraph agents
/api/                     # FastAPI endpoints
/db/                      # Database models and operations
/tests/                   # Comprehensive testing
/mcp_server/              # Model Context Protocol server
```

Remember: This is a production legal system - accuracy and reliability are paramount!