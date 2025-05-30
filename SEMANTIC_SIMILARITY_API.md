# Semantic Similarity API Documentation

The Semantic Similarity API provides endpoints for analyzing semantic relationships between legal citations using OpenAI embeddings and contextual analysis.

## Overview

This API enables:
- Extraction of semantic context around citations
- Generation of vector embeddings for citations
- Calculation of similarity scores between citations
- Discovery of semantically related citations
- Management of caching for performance optimization

## Base URL

All endpoints are prefixed with `/api/v1/semantic`

## Endpoints

### 1. Extract Citation Context

**POST** `/context`

Extracts semantic context information for a citation, including surrounding text, legal entities, concepts, and references.

#### Request Body
```json
{
    "citation_id": "uuid-string",
    "context_window": 500  // Number of characters around citation (100-2000)
}
```

#### Response
```json
{
    "citation_id": "uuid-string",
    "citation_text": "Miranda v. Arizona, 384 U.S. 436 (1966)",
    "surrounding_text": "The Supreme Court held in Miranda v. Arizona...",
    "extracted_entities": ["Supreme Court", "United States"],
    "legal_concepts": ["constitutional", "interrogation", "rights"],
    "case_names": ["Miranda v. Arizona"],
    "statutory_references": [],
    "position_in_document": 0.25
}
```

### 2. Generate Citation Embedding

**POST** `/embedding`

Generates a vector embedding for a citation using OpenAI's embedding model.

#### Request Body
```json
{
    "citation_id": "uuid-string",
    "include_surrounding_context": true
}
```

#### Response
```json
{
    "citation_id": "uuid-string",
    "embedding_dimension": 1536,
    "cache_hit": false,
    "embedding_text_preview": "Miranda v. Arizona, 384 U.S. 436 (1966) constitutional interrogation rights Supreme Court..."
}
```

### 3. Calculate Semantic Similarity

**POST** `/similarity`

Calculates semantic similarity between two citations using multiple metrics.

#### Request Body
```json
{
    "source_citation_id": "uuid-string",
    "target_citation_id": "uuid-string",
    "include_context": true
}
```

#### Response
```json
{
    "source_citation_id": "uuid-string",
    "target_citation_id": "uuid-string",
    "similarity_score": 0.85,
    "context_overlap": 0.72,
    "combined_confidence": 0.82,
    "match_reason": "shared_legal_concepts",
    "semantic_features": {
        "semantic_similarity": 0.85,
        "context_overlap": 0.72,
        "shared_concepts": ["constitutional", "rights"],
        "shared_entities": ["Supreme Court"],
        "shared_case_names": [],
        "position_distance": 0.15
    }
}
```

### 4. Find Semantic Matches

**POST** `/matches`

Finds citations that are semantically similar to a source citation.

#### Request Body
```json
{
    "source_citation_id": "uuid-string",
    "document_id": "uuid-string",  // Optional: limit to specific document
    "threshold": 0.7,              // Minimum similarity (0.0-1.0)
    "max_matches": 5               // Maximum results (1-20)
}
```

#### Response
```json
[
    {
        "source_citation_id": "uuid-string",
        "target_citation_id": "uuid-string",
        "target_citation_text": "Dickerson v. United States, 530 U.S. 428 (2000)",
        "similarity_score": 0.89,
        "context_overlap": 0.75,
        "combined_confidence": 0.85,
        "match_reason": "shared_case_references",
        "semantic_features": {
            "semantic_similarity": 0.89,
            "context_overlap": 0.75,
            "shared_concepts": ["Miranda", "constitutional", "interrogation"],
            "shared_entities": ["Supreme Court", "United States"],
            "shared_case_names": ["Miranda v. Arizona"],
            "position_distance": 0.3
        }
    }
]
```

### 5. Clean Cache

**POST** `/cache/cleanup`

Manually triggers cleanup of expired cache entries to free memory.

#### Response
```json
{
    "status": "success",
    "message": "Cache cleanup completed",
    "embedding_cache_cleared": true,
    "context_cache_cleared": true
}
```

### 6. Get Statistics

**GET** `/statistics`

Returns service statistics and configuration information.

#### Response
```json
{
    "embedding_cache_size": 42,
    "context_cache_size": 38,
    "cache_expiry_hours": 24,
    "last_cache_cleanup": "2025-01-30T10:15:30Z",
    "supported_legal_concepts": ["procedural", "constitutional", "contract", "tort", "criminal"],
    "embedding_dimension": 1536
}
```

## Match Reasons

The API identifies different reasons for semantic matches:

- `high_semantic_similarity`: Very high embedding similarity (>0.9)
- `strong_context_overlap`: Strong overlap in surrounding context (>0.8)
- `shared_case_references`: Citations reference the same cases
- `shared_legal_concepts`: Multiple shared legal concepts (>2)
- `proximity_in_document`: Citations appear close together in document
- `moderate_semantic_similarity`: Moderate embedding similarity (>0.7)
- `weak_similarity`: Low confidence match
- `error_fallback`: Error occurred during processing

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK`: Successful operation
- `400 Bad Request`: Invalid request parameters or missing document text
- `404 Not Found`: Citation or document not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error during processing

Error responses include a detail message:
```json
{
    "detail": "Citation {id} not found"
}
```

## Performance Considerations

1. **Caching**: Embeddings and contexts are cached for 24 hours to improve performance
2. **Batch Processing**: For multiple citations, consider using the find matches endpoint
3. **Context Window**: Larger context windows provide more accurate results but increase processing time
4. **Rate Limiting**: The underlying OpenAI service includes rate limiting protection

## Use Cases

1. **Citation Deduplication**: Find duplicate citations expressed differently
2. **Authority Analysis**: Identify citations to related legal authorities
3. **Document Linking**: Connect documents discussing similar legal concepts
4. **Research Enhancement**: Discover relevant cases based on semantic similarity
5. **Quality Assurance**: Verify citation consistency across documents