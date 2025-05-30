"""
Test suite for citation graph data structure and functionality.
Tests graph building, network analysis, and visualization export features.
"""
import pytest
import asyncio
from typing import List
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
import uuid

from nlp.citation_graph import CitationGraph, CitationNode, CitationEdge, CitationGraphBuilder
from db.models import Citation, CitationRelationship


@pytest.fixture
def sample_citations_with_relationships():
    """Fixture providing sample citations with relationships for graph testing"""
    citations = []
    relationships = []
    
    # Document 1 citations
    full_citation_1 = Citation(
        id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        document_id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        citation_text="Brown v. Board of Education, 347 U.S. 483 (1954)",
        citation_type="FullCaseCitation",
        reporter="U.S.",
        volume="347",
        page="483",
        position_start=100,
        position_end=150,
        confidence_score=0.95,
        created_at=datetime.utcnow()
    )
    citations.append(full_citation_1)
    
    id_citation_1 = Citation(
        id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
        document_id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        citation_text="Id. at 493",
        citation_type="IdCitation",
        page="493",
        position_start=200,
        position_end=210,
        confidence_score=0.85,
        created_at=datetime.utcnow()
    )
    citations.append(id_citation_1)
    
    # Document 2 citations
    full_citation_2 = Citation(
        id=uuid.UUID("33333333-3333-3333-3333-333333333333"),
        document_id=uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        citation_text="Miranda v. Arizona, 384 U.S. 436 (1966)",
        citation_type="FullCaseCitation",
        reporter="U.S.",
        volume="384",
        page="436",
        position_start=300,
        position_end=350,
        confidence_score=0.92,
        created_at=datetime.utcnow()
    )
    citations.append(full_citation_2)
    
    supra_citation_2 = Citation(
        id=uuid.UUID("44444444-4444-4444-4444-444444444444"),
        document_id=uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        citation_text="Brown, supra, at 495",
        citation_type="SupraCitation",
        page="495",
        position_start=400,
        position_end=420,
        confidence_score=0.80,
        created_at=datetime.utcnow()
    )
    citations.append(supra_citation_2)
    
    # Citation relationships
    rel_1 = CitationRelationship(
        id=uuid.UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"),
        source_citation_id=uuid.UUID("22222222-2222-2222-2222-222222222222"),  # id_citation_1
        target_citation_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),  # full_citation_1
        relationship_type="id_reference",
        confidence_score=0.9,
        created_at=datetime.utcnow()
    )
    relationships.append(rel_1)
    
    rel_2 = CitationRelationship(
        id=uuid.UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"),
        source_citation_id=uuid.UUID("44444444-4444-4444-4444-444444444444"),  # supra_citation_2
        target_citation_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),  # full_citation_1
        relationship_type="supra_reference",
        confidence_score=0.85,
        created_at=datetime.utcnow()
    )
    relationships.append(rel_2)
    
    return citations, relationships


@pytest.fixture
def empty_citation_graph():
    """Fixture providing an empty citation graph"""
    return CitationGraph()


@pytest.fixture 
def populated_citation_graph(sample_citations_with_relationships):
    """Fixture providing a citation graph populated with sample data"""
    citations, relationships = sample_citations_with_relationships
    graph = CitationGraph()
    
    # Add nodes
    for citation in citations:
        graph.add_citation_node(citation)
    
    # Add edges
    for relationship in relationships:
        graph.add_citation_relationship(relationship)
    
    return graph


class TestCitationNode:
    """Test suite for CitationNode functionality"""
    
    def test_citation_node_creation(self, sample_citations_with_relationships):
        """Test citation node creation and properties"""
        citations, _ = sample_citations_with_relationships
        citation = citations[0]  # Brown v. Board
        
        node = CitationNode(
            citation_id=str(citation.id),
            document_id=str(citation.document_id),
            citation_text=citation.citation_text,
            citation_type=citation.citation_type,
            reporter=citation.reporter,
            volume=citation.volume,
            page=citation.page,
            confidence_score=citation.confidence_score
        )
        
        assert node.citation_id == str(citation.id)
        assert node.citation_type == "FullCaseCitation"
        assert node.is_full_citation()
        assert not node.is_reference_citation()
        assert node.reporter == "U.S."
        assert node.volume == "347"
    
    def test_citation_node_serialization(self, sample_citations_with_relationships):
        """Test citation node to dictionary conversion"""
        citations, _ = sample_citations_with_relationships
        citation = citations[1]  # Id citation
        
        node = CitationNode(
            citation_id=str(citation.id),
            document_id=str(citation.document_id),
            citation_text=citation.citation_text,
            citation_type=citation.citation_type
        )
        
        node_dict = node.to_dict()
        
        assert isinstance(node_dict, dict)
        assert node_dict["citation_id"] == str(citation.id)
        assert node_dict["citation_type"] == "IdCitation"
        assert "citation_text" in node_dict


class TestCitationEdge:
    """Test suite for CitationEdge functionality"""
    
    def test_citation_edge_creation(self, sample_citations_with_relationships):
        """Test citation edge creation and properties"""
        _, relationships = sample_citations_with_relationships
        relationship = relationships[0]  # id_reference
        
        edge = CitationEdge(
            source_id=str(relationship.source_citation_id),
            target_id=str(relationship.target_citation_id),
            relationship_type=relationship.relationship_type,
            confidence_score=relationship.confidence_score,
            created_at=relationship.created_at
        )
        
        assert edge.relationship_type == "id_reference"
        assert edge.confidence_score == 0.9
        assert edge.source_id == str(relationship.source_citation_id)
        assert edge.target_id == str(relationship.target_citation_id)
    
    def test_citation_edge_serialization(self, sample_citations_with_relationships):
        """Test citation edge to dictionary conversion"""
        _, relationships = sample_citations_with_relationships
        relationship = relationships[1]  # supra_reference
        
        edge = CitationEdge(
            source_id=str(relationship.source_citation_id),
            target_id=str(relationship.target_citation_id),
            relationship_type=relationship.relationship_type,
            confidence_score=relationship.confidence_score,
            created_at=relationship.created_at
        )
        
        edge_dict = edge.to_dict()
        
        assert isinstance(edge_dict, dict)
        assert edge_dict["relationship_type"] == "supra_reference"
        assert edge_dict["confidence_score"] == 0.85


class TestCitationGraph:
    """Test suite for CitationGraph functionality"""
    
    def test_empty_graph_creation(self, empty_citation_graph):
        """Test empty graph initialization"""
        graph = empty_citation_graph
        
        assert graph.metadata["node_count"] == 0
        assert graph.metadata["edge_count"] == 0
        assert graph.metadata["document_count"] == 0
        assert len(graph.metadata["relationship_types"]) == 0
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
    
    def test_add_citation_node(self, empty_citation_graph, sample_citations_with_relationships):
        """Test adding citation nodes to graph"""
        graph = empty_citation_graph
        citations, _ = sample_citations_with_relationships
        citation = citations[0]  # Brown v. Board
        
        node = graph.add_citation_node(citation)
        
        assert graph.metadata["node_count"] == 1
        assert graph.metadata["document_count"] == 1
        assert str(citation.id) in graph.nodes
        assert node.citation_type == "FullCaseCitation"
        assert str(citation.document_id) in graph.document_map
    
    def test_add_citation_relationship(self, empty_citation_graph, sample_citations_with_relationships):
        """Test adding citation relationships to graph"""
        graph = empty_citation_graph
        citations, relationships = sample_citations_with_relationships
        
        # Add nodes first
        for citation in citations[:2]:  # Brown and Id citations
            graph.add_citation_node(citation)
        
        # Add relationship
        relationship = relationships[0]  # id_reference
        edge = graph.add_citation_relationship(relationship)
        
        assert graph.metadata["edge_count"] == 1
        assert len(graph.metadata["relationship_types"]) == 1
        assert "id_reference" in graph.metadata["relationship_types"]
        assert edge.relationship_type == "id_reference"
    
    def test_get_document_citations(self, populated_citation_graph):
        """Test retrieving citations for a specific document"""
        graph = populated_citation_graph
        doc_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        
        doc_citations = graph.get_document_citations(doc_id)
        
        assert len(doc_citations) == 2  # Brown and Id citations
        citation_types = [node.citation_type for node in doc_citations]
        assert "FullCaseCitation" in citation_types
        assert "IdCitation" in citation_types
    
    def test_get_neighbors(self, populated_citation_graph):
        """Test getting neighboring citations"""
        graph = populated_citation_graph
        brown_citation_id = "11111111-1111-1111-1111-111111111111"
        
        # Get incoming neighbors (citations that reference Brown)
        incoming = graph.get_neighbors(brown_citation_id, direction="in")
        assert len(incoming) == 2  # Id and supra citations reference Brown
        
        # Get outgoing neighbors (citations that Brown references)
        outgoing = graph.get_neighbors(brown_citation_id, direction="out")
        assert len(outgoing) == 0  # Brown doesn't reference other citations
        
        # Get all neighbors
        all_neighbors = graph.get_neighbors(brown_citation_id, direction="both")
        assert len(all_neighbors) == 2
    
    def test_network_metrics(self, populated_citation_graph):
        """Test network analysis metrics calculation"""
        graph = populated_citation_graph
        
        metrics = graph.analyze_network_metrics()
        
        assert "node_count" in metrics
        assert "edge_count" in metrics
        assert "density" in metrics
        assert metrics["node_count"] == 4
        assert metrics["edge_count"] == 2
        assert "full_citation_count" in metrics
        assert "reference_citation_count" in metrics
        assert metrics["full_citation_count"] == 2
        assert metrics["reference_citation_count"] == 2
    
    def test_graph_statistics(self, populated_citation_graph):
        """Test comprehensive graph statistics"""
        graph = populated_citation_graph
        
        stats = graph.get_statistics()
        
        assert "metadata" in stats
        assert "network_metrics" in stats
        assert "document_distribution" in stats
        assert "citation_type_distribution" in stats
        assert "relationship_type_distribution" in stats
        
        # Check document distribution
        doc_dist = stats["document_distribution"]
        assert len(doc_dist) == 2  # Two documents
        
        # Check citation type distribution
        type_dist = stats["citation_type_distribution"]
        assert type_dist["FullCaseCitation"] == 2
        assert type_dist["IdCitation"] == 1
        assert type_dist["SupraCitation"] == 1
        
        # Check relationship type distribution
        rel_dist = stats["relationship_type_distribution"]
        assert rel_dist["id_reference"] == 1
        assert rel_dist["supra_reference"] == 1


class TestGraphExport:
    """Test suite for graph export functionality"""
    
    def test_d3_json_export(self, populated_citation_graph):
        """Test D3.js JSON format export"""
        graph = populated_citation_graph
        
        d3_data = graph.export_for_visualization("json")
        
        assert "nodes" in d3_data
        assert "links" in d3_data
        assert "metadata" in d3_data
        assert len(d3_data["nodes"]) == 4
        assert len(d3_data["links"]) == 2
        
        # Check node structure
        node = d3_data["nodes"][0]
        assert "id" in node
        assert "label" in node
        assert "type" in node
        assert "group" in node
        
        # Check link structure
        link = d3_data["links"][0]
        assert "source" in link
        assert "target" in link
        assert "relationship" in link
        assert "strength" in link
    
    def test_cytoscape_export(self, populated_citation_graph):
        """Test Cytoscape.js format export"""
        graph = populated_citation_graph
        
        cytoscape_data = graph.export_for_visualization("cytoscape")
        
        assert "elements" in cytoscape_data
        assert "metadata" in cytoscape_data
        
        elements = cytoscape_data["elements"]
        # Should have 4 nodes + 2 edges = 6 elements
        assert len(elements) == 6
        
        # Check that we have both nodes and edges
        nodes = [e for e in elements if "source" not in e["data"]]
        edges = [e for e in elements if "source" in e["data"]]
        assert len(nodes) == 4
        assert len(edges) == 2
    
    def test_dot_export(self, populated_citation_graph):
        """Test DOT format export"""
        graph = populated_citation_graph
        
        dot_data = graph.export_for_visualization("dot")
        
        assert isinstance(dot_data, str)
        assert "digraph CitationGraph" in dot_data
        assert "->" in dot_data  # Should have directed edges
        
    def test_graphml_export(self, populated_citation_graph):
        """Test GraphML format export"""
        graph = populated_citation_graph
        
        graphml_data = graph.export_for_visualization("graphml")
        
        assert isinstance(graphml_data, str)
        # Should be XML format or error message
        assert ("<?xml" in graphml_data) or ("<error>" in graphml_data)
    
    def test_invalid_export_format(self, populated_citation_graph):
        """Test handling of invalid export formats"""
        graph = populated_citation_graph
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            graph.export_for_visualization("invalid_format")


class TestCitationGraphBuilder:
    """Test suite for CitationGraphBuilder functionality"""
    
    @pytest.mark.asyncio
    async def test_graph_builder_initialization(self):
        """Test graph builder initialization"""
        builder = CitationGraphBuilder()
        assert builder is not None
    
    @pytest.mark.asyncio
    async def test_build_graph_from_documents_mock(self, sample_citations_with_relationships):
        """Test building graph from documents with mocked database"""
        citations, relationships = sample_citations_with_relationships
        
        # Mock database session
        mock_session = AsyncMock()
        
        # Mock citation query
        mock_citation_result = MagicMock()
        mock_citation_result.scalars.return_value.all.return_value = citations
        mock_session.execute.return_value = mock_citation_result
        
        # Mock relationship query  
        mock_rel_result = MagicMock()
        mock_rel_result.scalars.return_value.all.return_value = relationships
        
        # Configure mock to return different results for different queries
        def mock_execute(query):
            # Simple heuristic to distinguish query types
            query_str = str(query)
            if "citation_relationships" in query_str.lower():
                return mock_rel_result
            else:
                return mock_citation_result
        
        mock_session.execute.side_effect = mock_execute
        
        builder = CitationGraphBuilder()
        document_ids = ["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"]
        
        graph = await builder.build_graph_from_documents(document_ids, mock_session)
        
        assert graph.metadata["node_count"] == 4
        assert graph.metadata["edge_count"] == 2
        assert graph.metadata["document_count"] == 2


@pytest.mark.asyncio
async def test_integration_with_graph_service():
    """Integration test with graph service functionality"""
    # Test that we can create and use the graph service
    from nlp.citation_graph import citation_graph_builder
    
    assert citation_graph_builder is not None
    assert hasattr(citation_graph_builder, 'build_graph_from_documents')
    assert hasattr(citation_graph_builder, 'build_document_citation_graph')


if __name__ == "__main__":
    # Run a simple test
    asyncio.run(test_integration_with_graph_service()) 