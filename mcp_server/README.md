# LeMCS MCP Server

Model Context Protocol (MCP) server for exposing LeMCS legal document processing capabilities to AI systems.

## Overview

The LeMCS MCP server provides standardized tools that allow Claude Desktop and other AI systems to interact with our legal document processing platform. This enables seamless integration of legal analysis capabilities into AI workflows.

## Features

### Document Management
- **upload_document**: Upload and process legal documents (DOCX/PDF)
- **get_document**: Retrieve document metadata and content
- **list_documents**: Browse available documents with filtering

### Citation Extraction
- **extract_citations**: Extract legal citations using eyecite with >99% accuracy
- **get_document_citations**: Retrieve all citations for a document
- **search_citations**: Search citations by text content
- **analyze_citation_authority**: Analyze precedential strength

### Workflow Management
- **get_workflows**: Monitor agent execution status
- **get_workflow_status**: Get detailed workflow information
- **get_citation_statistics**: Performance metrics and analytics

### Resources
- **document://{id}**: Access full document text content
- **citations://{id}**: Get formatted citation lists

## Installation

1. Install the MCP dependency:
```bash
pip install mcp>=1.2.0
```

2. Ensure database is running:
```bash
docker compose up -d postgres
```

## Usage

### Standalone Server
```bash
DATABASE_URL="postgresql+asyncpg://lemcs_user:lemcs_password@localhost/lemcs" python mcp/server.py
```

### Claude Desktop Integration

Add to your Claude Desktop configuration (`~/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "lemcs": {
      "command": "python",
      "args": ["/path/to/aidrafter/mcp/server.py"],
      "env": {
        "DATABASE_URL": "postgresql+asyncpg://lemcs_user:lemcs_password@localhost/lemcs"
      }
    }
  }
}
```

## Tool Examples

### Upload and Analyze a Document
```
User: "Upload the lease agreement from /path/to/lease.docx and extract all citations"
Claude: [Uses upload_document tool, then extract_citations tool automatically]
```

### Search for Specific Citations
```
User: "Find all citations mentioning 'landlord tenant' in the uploaded documents"
Claude: [Uses search_citations tool with query="landlord tenant"]
```

### Analyze Citation Authority
```
User: "What's the precedential strength of the citations in document abc-123?"
Claude: [Uses get_document_citations with include_authority=True]
```

## Architecture

The MCP server wraps our existing FastAPI endpoints and database operations into MCP tool definitions:

```
MCP Client (Claude Desktop)
    ↓ stdio transport
MCP Server (mcp/server.py)
    ↓ direct function calls
LeMCS Services (nlp/, agents/, db/)
    ↓ async operations
PostgreSQL Database
```

## Error Handling

- **Validation Errors**: Clear messages for invalid inputs
- **Database Errors**: Graceful handling of connection issues  
- **Processing Errors**: Detailed context for failed operations
- **Resource Not Found**: User-friendly error messages

## Security

- Input validation and sanitization
- Database connection security
- No authentication required for local stdio transport
- Future: OAuth2 integration for enterprise deployments

## Development

The server uses FastMCP framework with:
- Async/await throughout for performance
- Type hints for automatic schema generation
- Comprehensive error handling
- Resource management for database connections

## Monitoring

Use the workflow management tools to monitor:
- Citation extraction performance
- Processing times and error rates
- Agent execution status
- Database statistics

## Future Enhancements

- Full CRRACC consolidation workflow tools
- Vector similarity search capabilities  
- OpenAI integration for embeddings
- Enterprise authentication and rate limiting
- HTTP/SSE transport for remote access