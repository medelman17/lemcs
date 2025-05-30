# LeMCS MCP Server Design

## Overview

This document outlines the design for exposing LeMCS (Legal Memoranda Consolidation System) functionality through a Model Context Protocol (MCP) server. This will allow other AI systems to leverage our legal document processing capabilities as standardized tools.

## MCP Server Architecture

### Core Components

1. **FastMCP Server**: Main server using the official Python MCP SDK
2. **Tool Definitions**: Structured tools exposing LeMCS functionality
3. **Resource Handlers**: Access to document content and metadata
4. **Authentication**: Optional OAuth2 integration for enterprise use

### Transport Method
- **stdio**: Standard input/output for local desktop clients (Claude Desktop)
- **Future**: HTTP/SSE for remote integrations

## Tool Categories

### 1. Document Management Tools

#### `upload_document`
```python
@mcp.tool()
async def upload_document(
    file_path: str,
    document_type: str = "legal_memorandum",
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Upload a legal document and extract text content"""
```

#### `get_document`
```python
@mcp.tool()
async def get_document(document_id: str) -> Dict[str, Any]:
    """Retrieve document metadata and content"""
```

#### `list_documents`
```python
@mcp.tool()
async def list_documents(
    status: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """List available documents with optional filtering"""
```

### 2. Citation Extraction Tools

#### `extract_citations`
```python
@mcp.tool()
async def extract_citations(
    document_id: str,
    resolve_references: bool = True,
    analyze_authority: bool = True
) -> Dict[str, Any]:
    """Extract and analyze legal citations from a document"""
```

#### `get_document_citations`
```python
@mcp.tool()
async def get_document_citations(
    document_id: str,
    include_authority: bool = True
) -> List[Dict[str, Any]]:
    """Get all citations for a specific document"""
```

#### `search_citations`
```python
@mcp.tool()
async def search_citations(
    query: str,
    document_ids: Optional[List[str]] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search citations by text content"""
```

### 3. Legal Analysis Tools

#### `analyze_citation_authority`
```python
@mcp.tool()
async def analyze_citation_authority(citation_id: str) -> Dict[str, Any]:
    """Analyze the precedential authority of a specific citation"""
```

#### `get_citation_statistics`
```python
@mcp.tool()
async def get_citation_statistics(
    document_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get citation extraction performance statistics"""
```

### 4. Workflow Management Tools

#### `get_workflows`
```python
@mcp.tool()
async def get_workflows(
    agent_type: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Get recent agent workflow executions"""
```

#### `get_workflow_status`
```python
@mcp.tool()
async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """Get detailed status of a specific workflow"""
```

### 5. Future Consolidation Tools

#### `consolidate_memoranda`
```python
@mcp.tool()
async def consolidate_memoranda(
    document_ids: List[str],
    consolidation_method: str = "crracc",
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Consolidate multiple legal memoranda using CRRACC methodology"""
```

#### `get_consolidation_status`
```python
@mcp.tool()
async def get_consolidation_status(job_id: str) -> Dict[str, Any]:
    """Get status of a consolidation job"""
```

## Resource Definitions

### Document Content Resources
```python
@mcp.resource("document://{document_id}")
async def get_document_content(document_id: str) -> str:
    """Get the full text content of a document"""
```

### Citation Resources
```python
@mcp.resource("citations://{document_id}")
async def get_document_citations_resource(document_id: str) -> str:
    """Get all citations for a document as formatted text"""
```

## Error Handling Strategy

1. **Validation Errors**: Return clear messages for invalid inputs
2. **Database Errors**: Handle connection issues gracefully
3. **Processing Errors**: Provide detailed error context for failed operations
4. **Rate Limiting**: Future implementation for enterprise usage

## Authentication & Security

### Phase 1 (MVP)
- No authentication (local stdio transport)
- Input validation and sanitization

### Phase 2 (Enterprise)
- OAuth2 integration using `mcp.server.auth`
- Role-based access control
- Audit logging

## Configuration

### Environment Variables
```
LEMCS_DATABASE_URL=postgresql+asyncpg://...
LEMCS_LOG_LEVEL=INFO
LEMCS_MCP_SERVER_NAME=LeMCS Legal Processing Server
```

### Server Configuration
```python
# mcp_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="LeMCS Legal Processing Server",
    version="0.1.0",
    description="AI-powered legal document processing and citation analysis"
)
```

## Integration Benefits

1. **Standardized Interface**: Other AI applications can easily integrate
2. **Tool Discovery**: Automatic tool schema generation from type hints
3. **Error Handling**: Consistent error responses across all tools
4. **Monitoring**: Built-in progress tracking and logging
5. **Extensibility**: Easy to add new tools as LeMCS grows

## Implementation Plan

1. **Phase 1**: Core document and citation tools
2. **Phase 2**: Advanced analysis and search capabilities
3. **Phase 3**: Full consolidation workflow tools
4. **Phase 4**: Enterprise features (auth, monitoring, rate limiting)

## Usage Examples

### Claude Desktop Integration
```bash
# Add to Claude Desktop config
{
  "mcpServers": {
    "lemcs": {
      "command": "python",
      "args": ["/path/to/lemcs/mcp_server.py"],
      "env": {
        "LEMCS_DATABASE_URL": "postgresql+asyncpg://..."
      }
    }
  }
}
```

### Tool Usage in Claude
```
User: "Extract citations from document abc-123 and analyze their authority"
Claude: [Uses extract_citations tool with document_id="abc-123", analyze_authority=True]
```