# Development Workflow Guide for LeMCS

## Quick Start Development Setup

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (Python 3.12 compatible)
pip install -r requirements.txt

# Download spaCy model (required for legal NLP)
python -m spacy download en_core_web_sm

# Start PostgreSQL with pgvector
docker compose up -d postgres

# Initialize database (first time only)
DATABASE_URL="postgresql+asyncpg://lemcs_user:lemcs_password@localhost/lemcs" python scripts/init_database.py
```

### Running the Application
```bash
# Start development server
python main_simple.py
# Access at: http://localhost:8000
# API docs: http://localhost:8000/docs

# Start MCP server (for AI system integration)
DATABASE_URL="postgresql+asyncpg://lemcs_user:lemcs_password@localhost/lemcs" python mcp_server/server.py
```

## Testing Workflow

### Run All Tests
```bash
pytest tests/ -v
```

### Test Specific Components
```bash
# Legal NLP tests
pytest tests/test_legal_nlp.py -v
pytest tests/test_legal_bert.py -v 
pytest tests/test_hybrid_nlp.py -v

# Basic application tests
pytest tests/test_basic.py -v
```

### Code Quality
```bash
# Format code
black .

# Type checking (when mypy is added)
mypy .

# Linting (when flake8 is added)
flake8 .
```

## Development Patterns

### Adding New Legal NLP Features
1. **Always extend HybridLegalNLP** - don't create new top-level services
2. **Add tests first** - legal accuracy is critical
3. **Use existing legal document types** - lease_agreement, legal_complaint, etc.
4. **Test with real legal text** - not placeholder text

Example:
```python
# In nlp/hybrid_legal_nlp.py
def new_legal_feature(self, text: str) -> Dict[str, Any]:
    """Add new legal processing capability."""
    # Use both spaCy and BERT for comprehensive analysis
    spacy_result = self.spacy_service.some_method(text)
    bert_result = self.bert_service.some_method(text)
    
    return {"spacy": spacy_result, "bert": bert_result}

# In tests/test_hybrid_nlp.py  
def test_new_legal_feature(hybrid_nlp):
    result = hybrid_nlp.new_legal_feature("legal document text...")
    assert "spacy" in result
    assert "bert" in result
```

### Adding New API Endpoints
1. **Use FastAPI async patterns**
2. **Include legal domain validation**
3. **Add comprehensive error handling**
4. **Document with legal examples**

Example:
```python
# In api/routes/documents.py
@router.post("/analyze-legal-document")
async def analyze_legal_document(
    document: UploadFile = File(...),
    nlp_service: HybridLegalNLP = Depends(get_hybrid_nlp)
):
    try:
        # Extract text from document
        text = await extract_document_text(document)
        
        # Process with legal NLP
        analysis = nlp_service.comprehensive_analysis(text)
        
        return {"status": "success", "analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Legal analysis failed: {e}")
```

### Database Operations
1. **Use AsyncPG for all database operations**
2. **Store LEGAL-BERT embeddings in pgvector**
3. **Include legal document metadata**

Example:
```python
# Store document with legal analysis
async def store_legal_document(
    db: AsyncSession,
    text: str,
    analysis: Dict[str, Any],
    nlp_service: HybridLegalNLP
):
    # Get semantic embedding
    embedding = nlp_service.get_text_embedding(text)
    
    # Store in database with pgvector
    document = Document(
        content=text,
        document_type=analysis['document_analysis']['type'],
        embedding=embedding.numpy().tolist(),
        metadata=analysis
    )
    
    db.add(document)
    await db.commit()
```

## Legal NLP Development Guidelines

### Entity Processing
```python
# Always use hybrid approach for entity extraction
entities = nlp.extract_all_entities(legal_text)

# Access different entity sources
spacy_entities = entities['spacy_entities']
bert_entities = entities['bert_entities'] 
combined_entities = entities['combined_unique']  # Use this for production

# Check entity types
for entity in combined_entities:
    entity_text = entity.get('text')
    entity_label = entity.get('label')
    confidence = entity.get('confidence', 0.0)
```

### Document Classification
```python
# Get comprehensive document analysis
analysis = nlp.comprehensive_analysis(legal_text)

# Extract key information
doc_type = analysis['document_analysis']['type']
confidence = analysis['document_analysis']['confidence']
legal_domain = analysis['document_analysis']['legal_domain']
complexity = analysis['document_analysis']['complexity_score']

# Handle different document types
if doc_type == 'lease_agreement':
    # Process rental/property-specific entities
    pass
elif doc_type == 'legal_complaint':
    # Process litigation-specific entities
    pass
```

### Semantic Similarity
```python
# Calculate similarity between legal documents
similarity = nlp.calculate_similarity(doc1_text, doc2_text)

# Use for document clustering
if similarity > 0.8:
    # Documents are highly similar
    pass
elif similarity > 0.6:
    # Documents are moderately similar
    pass
```

## Common Development Tasks

### Adding a New Legal Document Type
1. Update classification logic in `legal_bert_service.py`
2. Add test cases in `test_legal_bert.py`
3. Update entity extraction patterns if needed
4. Add database migration for new document type
5. Update API documentation

### Improving Legal Entity Recognition
1. Add patterns to `legal_nlp_service.py` or `legal_bert_service.py`
2. Test with diverse legal documents
3. Validate accuracy against known entities
4. Update hybrid service to use new patterns

### Performance Optimization
1. **Profile with real legal documents** - not test data
2. **Cache model instances** - expensive to reload
3. **Use batch processing** for multiple documents
4. **Consider GPU acceleration** for LEGAL-BERT if available

### Debugging Legal NLP Issues
1. **Check model loading** - are spaCy and BERT models available?
2. **Validate input text** - is it actually legal content?
3. **Test with known documents** - use documents with expected results
4. **Compare model outputs** - spaCy vs BERT vs hybrid

Example debugging:
```python
# Check service status
status = nlp.get_service_status()
print(f"spaCy available: {status['spacy_available']}")
print(f"BERT available: {status['bert_available']}")

# Compare model outputs
comparison = nlp.compare_models(legal_text)
print(f"spaCy time: {comparison['performance']['spacy_time_seconds']}")
print(f"BERT time: {comparison['performance']['bert_time_seconds']}")
print(f"Recommendations: {comparison['recommendations']}")
```

## Git Workflow

### Commit Guidelines
- **Focus commits** - one feature per commit
- **Test before committing** - ensure all tests pass
- **Include legal context** - mention document types tested
- **Update documentation** - keep CLAUDE.md current

### Branch Strategy
- `main` - production-ready code
- Feature branches for new capabilities
- Always test legal NLP changes thoroughly

### Pre-commit Checklist
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Code formatted (`black .`)
- [ ] Legal NLP models load correctly
- [ ] Documentation updated if needed
- [ ] Real legal documents tested (when applicable)

## Deployment Considerations

### Production Environment
- Ensure LEGAL-BERT model cache directory is writable
- Configure proper GPU/CPU resources for transformers
- Set up monitoring for legal NLP processing times
- Implement proper error handling for model failures

### Security
- Never commit legal document content to repository
- Use environment variables for sensitive configuration
- Implement proper access controls for legal documents
- Audit logging for all legal document operations

Remember: Legal accuracy is paramount - when in doubt, test with real legal documents and use the hybrid approach for maximum reliability!