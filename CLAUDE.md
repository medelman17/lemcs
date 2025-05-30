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

- Python 3.11+ (primary development language)
- LangGraph for agent orchestration
- FastAPI for REST API endpoints
- PostgreSQL 15+ with pgvector extension
- Redis for caching and job queuing
- eyecite for legal citation extraction
- LexNLP for legal document parsing
- spaCy with custom legal NER models
- docxcompose for DOCX manipulation

## Development Commands

Since this is a new project without existing build configuration, the following commands will need to be established:

### Python Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt  # Once created
```

### Running Tests
```bash
pytest tests/  # Standard Python testing
pytest tests/test_specific.py::test_function  # Run single test
```

### Code Quality
```bash
black .  # Code formatting
flake8 .  # Linting
mypy .  # Type checking
```

## Project Structure

The codebase should be organized following the microservices architecture outlined in the PRD:
- `/api` - FastAPI endpoints
- `/agents` - LangGraph agent implementations
- `/nlp` - Legal NLP processing modules
- `/document` - Document processing and manipulation
- `/db` - Database models and migrations
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