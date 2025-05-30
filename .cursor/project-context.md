# LeMCS Project Context for Cursor AI

## Project Status Overview (Current)

### ✅ Completed & Production Ready
- **FastAPI Application**: Running at http://localhost:8000 with comprehensive API
- **PostgreSQL Database**: 15-table schema with pgvector for semantic search
- **Legal NLP Pipeline**: Hybrid spaCy + LEGAL-BERT system (27 passing tests)
- **Citation Extraction**: eyecite integration with >99% accuracy
- **MCP Server**: 12 tools, 2 resources for AI system integration
- **Testing Framework**: Comprehensive test coverage across all components
- **Documentation**: README, API docs, development guides

### 🔧 Technology Stack (All Python 3.12 Compatible)
- **Web**: FastAPI 0.109.0 + Uvicorn
- **Database**: PostgreSQL 15+ with pgvector, AsyncPG
- **AI/ML**: LangGraph 0.0.26, OpenAI, Anthropic APIs
- **Legal NLP**: spaCy 3.8.7 + transformers 4.52.3 (LEGAL-BERT)
- **Citation**: eyecite 2.7.5 (legal citation extraction)
- **Documents**: python-docx, docxcompose for document manipulation
- **Testing**: pytest with async support

### 📁 Current Project Structure
```
/aidrafter/
├── agents/                 # LangGraph multi-agent system
│   ├── base.py            # Base agent class
│   ├── citation_extractor.py  # Citation extraction agent
│   ├── legal_analyzer.py     # Legal document analysis
│   ├── orchestrator.py       # Agent coordination
│   └── synthesis_agent.py    # Document consolidation
├── api/                   # FastAPI endpoints
│   └── routes/           # API route definitions
├── config/               # Application configuration
├── db/                   # Database models and operations
├── nlp/                  # ⭐ LEGAL NLP SERVICES (MAIN FOCUS)
│   ├── hybrid_legal_nlp.py    # 🎯 PRIMARY SERVICE (use this!)
│   ├── legal_bert_service.py  # LEGAL-BERT transformer service
│   ├── legal_nlp_service.py   # spaCy-based fast processing
│   └── citation_service.py    # eyecite integration
├── mcp_server/           # Model Context Protocol server
├── tests/                # Comprehensive testing
│   ├── test_hybrid_nlp.py     # Hybrid service tests
│   ├── test_legal_bert.py     # LEGAL-BERT tests
│   └── test_legal_nlp.py      # spaCy service tests
└── .cursor/              # 📋 Cursor AI rules and guides
```

### 🎯 Primary Development Focus: Legal NLP System

The core of LeMCS is the legal NLP pipeline. **Always use `HybridLegalNLP`** as your primary interface:

```python
from nlp.hybrid_legal_nlp import HybridLegalNLP

# Initialize once (expensive operation)
nlp = HybridLegalNLP()

# Comprehensive analysis (recommended for all legal documents)
analysis = nlp.comprehensive_analysis(legal_document_text)

# Key outputs:
doc_type = analysis['document_analysis']['type']  # lease_agreement, legal_complaint, etc.
entities = analysis['entities']['combined_unique']  # All entities from both models
confidence = analysis['document_analysis']['confidence']  # Classification confidence
complexity = analysis['document_analysis']['complexity_score']  # Document complexity
```

### 📊 Performance Characteristics (Measured)
- **spaCy Processing**: ~0.027s (fast entity extraction)
- **LEGAL-BERT Processing**: ~0.052s (semantic analysis)
- **Hybrid Approach**: Optimal balance of speed + accuracy
- **Citation Extraction**: >99% accuracy with eyecite
- **Document Classification**: High confidence (>80%) for known legal types

### 🏗️ Legal Document Types Supported
1. **lease_agreement**: Residential/commercial leases
2. **legal_complaint**: Litigation documents
3. **legal_memorandum**: Legal analysis documents  
4. **contract**: Commercial agreements
5. **unknown**: Unclassified documents

### 🔍 Entity Types Extracted
- **Standard**: PERSON, ORG, GPE, DATE, MONEY (via spaCy)
- **Legal-Specific**: Case citations, statutes, legal concepts (via LEGAL-BERT)
- **Combined**: Deduplicated entities from both models

### 🚀 Recent Major Achievement: LexNLP Replacement
**Problem Solved**: LexNLP was incompatible with Python 3.12 due to dependency conflicts
**Solution**: Implemented hybrid spaCy + LEGAL-BERT system that provides:
- ✅ Python 3.12 compatibility
- ✅ Superior semantic understanding (LEGAL-BERT trained on 12GB legal text)
- ✅ Faster processing (spaCy for real-time needs)
- ✅ Comprehensive test coverage (27 tests passing)

### 🎯 Current Development Priorities

#### High Priority
1. **Legal NLP Enhancement**: Expand entity types, improve classification accuracy
2. **Agent Workflow Integration**: Connect legal NLP with LangGraph agents
3. **Semantic Search**: Implement pgvector-based document similarity search
4. **Performance Optimization**: Cache models, batch processing

#### Medium Priority  
1. **API Enhancement**: Add more legal document processing endpoints
2. **Testing Expansion**: More real-world legal document tests
3. **MCP Tool Enhancement**: Expose more legal NLP capabilities
4. **Documentation**: Legal domain-specific examples

#### Low Priority
1. **UI Development**: Web interface for legal document processing
2. **Advanced Analytics**: Legal document insights and reporting
3. **Integration**: Third-party legal service integrations

### 🔧 Development Environment Notes

#### Required Environment Variables
```bash
DATABASE_URL="postgresql+asyncpg://lemcs_user:lemcs_password@localhost/lemcs"
```

#### Quick Development Commands
```bash
# Start development
source venv/bin/activate
python main_simple.py  # Web app at http://localhost:8000

# Run tests
pytest tests/ -v

# Format code
black .

# Start database
docker compose up -d postgres
```

### ⚠️ Important Development Notes

1. **Legal Accuracy is Critical**: This is a production legal system - accuracy over speed
2. **Always Test with Real Legal Documents**: Use actual lease agreements, complaints, etc.
3. **Model Loading is Expensive**: Cache HybridLegalNLP instances when possible
4. **Error Handling**: Legal processing should never crash - graceful degradation
5. **Documentation**: Legal domain complexity requires clear documentation

### 🎯 Next Logical Development Steps

When continuing development, consider these high-impact areas:

1. **Enhance Legal Entity Recognition**: Add more specific legal entity types
2. **Improve Document Classification**: Fine-tune LEGAL-BERT for specific legal domains
3. **Semantic Document Search**: Implement similarity search using pgvector + LEGAL-BERT
4. **Agent Integration**: Connect legal NLP outputs to LangGraph agent workflows
5. **Batch Processing**: Optimize for processing multiple legal documents efficiently

### 📈 Success Metrics
- Legal document classification accuracy >90%
- Entity extraction recall >95% 
- Processing time <1 second per document
- Citation extraction accuracy >99% (already achieved)
- Test coverage >90% (currently achieved)

### 🔄 Continuous Integration
- All tests must pass before commits
- Black formatting enforced
- Real legal document validation
- Performance regression monitoring

This project represents a cutting-edge legal AI system with production-ready legal NLP capabilities. The hybrid approach provides the best of both worlds: spaCy's speed with LEGAL-BERT's semantic understanding.