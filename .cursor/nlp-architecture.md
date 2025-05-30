# Legal NLP Architecture Guide for Cursor AI

## Overview
The LeMCS legal NLP system uses a hybrid approach combining spaCy's speed with LEGAL-BERT's semantic understanding, specifically designed for legal document processing.

## Core Components

### 1. HybridLegalNLP (Main Service)
**File**: `nlp/hybrid_legal_nlp.py`
**Purpose**: Primary interface combining spaCy + LEGAL-BERT
**Key Methods**:
```python
# Comprehensive analysis of legal documents
analysis = hybrid_nlp.comprehensive_analysis(legal_text)

# Extract entities using both models
entities = hybrid_nlp.extract_all_entities(legal_text)

# Calculate semantic similarity
similarity = hybrid_nlp.calculate_similarity(text1, text2)

# Compare model performance
comparison = hybrid_nlp.compare_models(legal_text)
```

### 2. LegalBERTService (Semantic Analysis)
**File**: `nlp/legal_bert_service.py`
**Model**: nlpaueb/legal-bert-base-uncased (trained on 12GB legal text)
**Capabilities**:
- Document classification (lease_agreement, legal_complaint, etc.)
- Semantic text embeddings for similarity
- Legal concept extraction
- High-confidence classification (>80% accuracy)

**Key Methods**:
```python
# Get LEGAL-BERT embeddings
embeddings = bert_service.encode_text(legal_text)

# Classify document type
classification = bert_service.classify_legal_text(legal_text)

# Extract legal concepts and entities
concepts = bert_service.extract_legal_concepts(legal_text)
```

### 3. LegalNLPService (Fast Processing)
**File**: `nlp/legal_nlp_service.py`  
**Model**: spaCy en_core_web_sm
**Capabilities**:
- Fast entity recognition (2x faster than BERT)
- Document structure analysis
- Legal pattern matching
- Real-time processing

**Key Methods**:
```python
# Fast entity extraction
entities = spacy_service.extract_entities(legal_text)

# Legal-specific entity extraction
legal_entities = spacy_service.extract_legal_entities(legal_text)

# Document structure analysis
structure = spacy_service.analyze_document_structure(legal_text)
```

## Performance Characteristics

### Speed Comparison (Measured)
- **spaCy**: ~0.027s for entity extraction
- **LEGAL-BERT**: ~0.052s for semantic analysis
- **Hybrid**: Optimal balance of speed and accuracy

### When to Use Each Component

#### Use spaCy Service Directly:
- Real-time processing requirements
- Simple entity extraction
- High-volume batch processing
- When speed > semantic accuracy

#### Use LEGAL-BERT Service Directly:
- Document classification tasks
- Semantic similarity calculations
- Legal concept extraction
- When accuracy > speed

#### Use Hybrid Service (Recommended):
- Comprehensive legal document analysis
- Production legal workflows
- When you need both speed and accuracy
- Default choice for most legal NLP tasks

## Legal Entity Types Supported

### Standard Entities (spaCy)
- PERSON: Names of individuals
- ORG: Organizations, companies
- GPE: Geopolitical entities (cities, states)
- DATE: Dates and time expressions
- MONEY: Monetary amounts

### Legal-Specific Entities (Custom)
- Case citations (e.g., "123 F.3d 456")
- Statutes (e.g., "42 U.S.C. § 1983")
- Legal concepts (plaintiff, defendant, breach, etc.)
- Contract terms and clauses
- Court names and jurisdictions

### Legal Document Types Classified
1. **lease_agreement**: Rental/property agreements
2. **legal_complaint**: Litigation documents
3. **legal_memorandum**: Analysis documents
4. **contract**: Commercial agreements
5. **unknown**: Unclassified documents

## Integration Patterns

### Document Processing Pipeline
```python
from nlp.hybrid_legal_nlp import HybridLegalNLP

# Initialize hybrid service
nlp = HybridLegalNLP()

# Process legal document
analysis = nlp.comprehensive_analysis(document_text)

# Extract structured data
doc_type = analysis['document_analysis']['type']
entities = analysis['entities']['combined_unique']
concepts = analysis['entities']['legal_concepts']
complexity = analysis['document_analysis']['complexity_score']
```

### Semantic Search Implementation
```python
# Get embeddings for similarity search
query_embedding = nlp.get_text_embedding(query_text)

# Store in pgvector database
await store_embedding(doc_id, query_embedding)

# Calculate similarity between documents
similarity_score = nlp.calculate_similarity(doc1, doc2)
```

### Multi-Document Analysis
```python
documents = [doc1, doc2, doc3]
analyses = []

for doc in documents:
    analysis = nlp.comprehensive_analysis(doc)
    analyses.append(analysis)

# Compare document types
doc_types = [a['document_analysis']['type'] for a in analyses]

# Find similar documents
similarities = []
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        sim = nlp.calculate_similarity(documents[i], documents[j])
        similarities.append((i, j, sim))
```

## Error Handling Patterns

### Model Loading Failures
```python
try:
    nlp = HybridLegalNLP()
except Exception as e:
    logger.error(f"Failed to load legal NLP models: {e}")
    # Fallback to basic text processing
    nlp = None
```

### Processing Errors
```python
try:
    analysis = nlp.comprehensive_analysis(text)
except Exception as e:
    logger.error(f"Legal NLP processing failed: {e}")
    # Return basic structure
    analysis = {"error": str(e), "document_analysis": {"type": "unknown"}}
```

## Testing Patterns

### Unit Testing
```python
def test_document_classification():
    nlp = HybridLegalNLP()
    lease_text = "This lease agreement between landlord and tenant..."
    analysis = nlp.comprehensive_analysis(lease_text)
    assert analysis['document_analysis']['type'] == 'lease_agreement'
```

### Performance Testing
```python
def test_processing_speed():
    nlp = HybridLegalNLP()
    text = "Legal document text..."
    
    start_time = time.time()
    analysis = nlp.comprehensive_analysis(text)
    processing_time = time.time() - start_time
    
    assert processing_time < 1.0  # Should process under 1 second
```

## Best Practices

1. **Always use HybridLegalNLP** for production legal document processing
2. **Cache model instances** - they're expensive to load
3. **Batch process** multiple documents when possible
4. **Validate inputs** - ensure text is legal document content
5. **Handle errors gracefully** - legal processing shouldn't crash systems
6. **Log performance metrics** - track processing times and accuracy
7. **Test with real legal documents** - use actual lease agreements, complaints, etc.

## Model Updates and Maintenance

### Updating Models
- spaCy models: `python -m spacy download en_core_web_sm`
- LEGAL-BERT: Automatic download from Hugging Face on first use
- Check for model updates quarterly

### Performance Monitoring
- Track entity extraction accuracy
- Monitor document classification confidence scores
- Measure processing times per document type
- Log memory usage for large documents

Remember: Legal accuracy is critical - when in doubt, use the hybrid approach for maximum reliability!