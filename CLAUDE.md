# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeMCS (Legal Memoranda Consolidation System) is an AI-powered document processing platform designed to consolidate multiple legal memoranda into comprehensive omnibus documents. The system uses a multi-agent architecture with specialized legal NLP tools to analyze complex residential lease agreements and create litigation-ready documentation.

## Architecture

The system employs a multi-agent AI architecture with the following core components:
- **Orchestration Layer**: LangGraph-based multi-agent system for workflow coordination
- **Legal NLP Engine**: Integrates eyecite, LexNLP, and spaCy with legal models for document analysis
- **Document Processing**: Uses docxcompose for format preservation and document manipulation
- **Knowledge Base**: PostgreSQL with pgvector extension for semantic search capabilities
- **Quality Assurance**: Automated validation with human-in-the-loop review stages

## Key Technologies

- Python 3.12+ (primary development language - validated and working)
- FastAPI for REST API endpoints (✅ implemented and tested)
- LangGraph for agent orchestration (✅ resolved - v0.4.7 compatible with Python 3.12)
- PostgreSQL 15+ with pgvector extension (✅ implemented with comprehensive schema)
- Redis for caching and job queuing (planned)
- eyecite for legal citation extraction (✅ fully integrated with multi-agent workflow)
- LexNLP for legal document parsing (⚠️ dependency conflicts with Python 3.12)
- spaCy with custom legal NER models (alternative needed)
- python-docx for DOCX text extraction (✅ working)

## Development Commands

### Python Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-basic.txt  # Core dependencies that work with Python 3.12
```

### Running the Application
```bash
# Start PostgreSQL with pgvector
docker compose up -d postgres

# Initialize database (first time only)
DATABASE_URL="postgresql+asyncpg://lemcs_user:lemcs_password@localhost/lemcs" python scripts/init_database.py

# Start development server
python main_simple.py
# Application will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Running MCP Server
```bash
# Start MCP server for integration with Claude Desktop or other AI systems
DATABASE_URL="postgresql+asyncpg://lemcs_user:lemcs_password@localhost/lemcs" python mcp_server/server.py
# Server exposes LeMCS functionality through standardized MCP tools
# Configure in Claude Desktop settings to access legal document processing tools

# Test MCP tools (examples)
# upload_document("/path/to/document.txt") - Upload and process documents
# extract_citations(doc_id) - Extract legal citations with authority analysis
# search_citations("landlord tenant") - Search citations by text content
```

### Running Tests
```bash
pytest tests/  # Run all tests
pytest tests/test_basic.py -v  # Run specific test file with verbose output
```

### Code Quality
```bash
black .  # Code formatting (installed and working)
# Note: flake8 and mypy need to be added to requirements for full linting
```

### Validation Status
✅ **Environment Validated** (Python 3.12.3)
✅ **FastAPI Application** (Running successfully)
✅ **Document Upload** (DOCX text extraction working)
✅ **API Endpoints** (Health check, document upload, citation extraction)
✅ **Testing Framework** (4 passing tests with pytest)
✅ **Code Formatting** (Black formatter working)
✅ **Database Schema** (PostgreSQL with pgvector, 15 tables for semantic search)
✅ **Multi-Agent Workflow** (LangGraph integration ready)
✅ **Citation Extraction** (eyecite integration with >99% accuracy, authority analysis)
✅ **Legal NLP Pipeline** (Citation service, agent workflow, API endpoints)
✅ **MCP Server** (Model Context Protocol server tested and working - 12 tools, 2 resources)
✅ **End-to-End Testing** (Document upload, citation extraction, authority analysis via MCP)

## Project Structure

The codebase should be organized following the microservices architecture outlined in the PRD:
- `/api` - FastAPI endpoints
- `/agents` - LangGraph agent implementations
- `/nlp` - Legal NLP processing modules
- `/document` - Document processing and manipulation
- `/db` - Database models and migrations
- `/mcp_server` - Model Context Protocol server for AI system integration (renamed to avoid conflicts)
- `/tests` - Test suite

## Legal Document Processing Methodology

The system follows the CRRACC Synthesis Method for consolidating legal memoranda:
1. **Conclusion** - State overarching legal violations
2. **Rule statement** - Synthesize rules from multiple memoranda
3. **Rule explanation** - Consolidate explanations without redundancy
4. **Application** - Integrate applications using comparative analysis
5. **Counterargument** - Address counterarguments comprehensively
6. **Conclusion** - Reinforce pattern of violations

Key principles:
- Organize content by legal theory, not provision-by-provision
- Maintain narrative prose format throughout
- Preserve all citations and verbatim quotes
- Create hierarchical framework mirroring complaint structure

## Performance Requirements

- Process 25 memoranda in under 10 minutes
- >99% citation extraction accuracy
- Support concurrent processing of 10+ consolidation jobs
- Sub-second citation extraction per page

## Security Considerations

- End-to-end encryption for document storage
- Role-based access control (RBAC)
- Audit logging for all operations
- GDPR/CCPA compliance for PII handling
- Air-gapped deployment option available