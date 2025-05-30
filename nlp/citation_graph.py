"""
Citation Graph Data Structure for Legal Document Analysis.
Builds graph representations of citation relationships for network analysis and visualization.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import asyncio

import networkx as nx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from db.models import Citation, CitationRelationship, Document

logger = logging.getLogger(__name__)


@dataclass
class CitationNode:
    """Represents a citation node in the graph with full metadata"""
    citation_id: str
    document_id: str
    citation_text: str
    citation_type: str
    reporter: Optional[str] = None
    volume: Optional[str] = None
    page: Optional[str] = None
    position_start: Optional[int] = None
    position_end: Optional[int] = None
    confidence_score: Optional[float] = None
    court_level: Optional[str] = None
    jurisdiction: Optional[str] = None
    authority_score: Optional[float] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization"""
        return asdict(self)
    
    def is_reference_citation(self) -> bool:
        """Check if this is a reference citation (id, supra, short)"""
        return self.citation_type in ["IdCitation", "SupraCitation", "ShortCaseCitation"]
    
    def is_full_citation(self) -> bool:
        """Check if this is a full case citation"""
        return self.citation_type == "FullCaseCitation"


@dataclass
class CitationEdge:
    """Represents a citation relationship edge with metadata"""
    source_id: str
    target_id: str
    relationship_type: str
    confidence_score: float
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for serialization"""
        return asdict(self)


class CitationGraph:
    """
    Graph-based data structure for citation relationships using NetworkX.
    
    Supports:
    - Multi-document citation networks
    - Various relationship types (id, supra, short_form)
    - Network analysis (centrality, clustering, paths)
    - Visualization export (JSON, DOT, GraphML)
    - Performance optimization for large datasets
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for citation relationships
        self.nodes: Dict[str, CitationNode] = {}
        self.edges: Dict[Tuple[str, str], CitationEdge] = {}
        self.document_map: Dict[str, Set[str]] = defaultdict(set)  # doc_id -> citation_ids
        self.metadata = {
            "created_at": datetime.utcnow(),
            "node_count": 0,
            "edge_count": 0,
            "document_count": 0,
            "relationship_types": set()
        }
    
    def add_citation_node(self, citation: Citation) -> CitationNode:
        """Add a citation as a node in the graph"""
        node = CitationNode(
            citation_id=str(citation.id),
            document_id=str(citation.document_id),
            citation_text=citation.citation_text,
            citation_type=citation.citation_type,
            reporter=citation.reporter,
            volume=citation.volume,
            page=citation.page,
            position_start=citation.position_start,
            position_end=citation.position_end,
            confidence_score=citation.confidence_score,
            created_at=citation.created_at
        )
        
        # Add to graph structures
        self.graph.add_node(node.citation_id, **node.to_dict())
        self.nodes[node.citation_id] = node
        self.document_map[node.document_id].add(node.citation_id)
        
        # Update metadata
        self.metadata["node_count"] = len(self.nodes)
        self.metadata["document_count"] = len(self.document_map)
        
        return node
    
    def add_citation_relationship(self, relationship: CitationRelationship) -> CitationEdge:
        """Add a citation relationship as an edge in the graph"""
        edge = CitationEdge(
            source_id=str(relationship.source_citation_id),
            target_id=str(relationship.target_citation_id),
            relationship_type=relationship.relationship_type,
            confidence_score=relationship.confidence_score,
            created_at=relationship.created_at
        )
        
        # Add to graph structures
        self.graph.add_edge(
            edge.source_id, 
            edge.target_id,
            **edge.to_dict()
        )
        self.edges[(edge.source_id, edge.target_id)] = edge
        
        # Update metadata
        self.metadata["edge_count"] = len(self.edges)
        self.metadata["relationship_types"].add(edge.relationship_type)
        
        return edge
    
    def get_node(self, citation_id: str) -> Optional[CitationNode]:
        """Get a citation node by ID"""
        return self.nodes.get(citation_id)
    
    def get_edge(self, source_id: str, target_id: str) -> Optional[CitationEdge]:
        """Get a citation relationship edge"""
        return self.edges.get((source_id, target_id))
    
    def get_document_citations(self, document_id: str) -> List[CitationNode]:
        """Get all citation nodes for a specific document"""
        citation_ids = self.document_map.get(document_id, set())
        return [self.nodes[cid] for cid in citation_ids if cid in self.nodes]
    
    def get_neighbors(self, citation_id: str, direction: str = "both") -> List[CitationNode]:
        """
        Get neighboring citation nodes
        
        Args:
            citation_id: The citation to find neighbors for
            direction: "in" (predecessors), "out" (successors), or "both"
        """
        neighbors = []
        
        if direction in ["in", "both"]:
            # Citations that reference this one
            predecessors = list(self.graph.predecessors(citation_id))
            neighbors.extend([self.nodes[nid] for nid in predecessors if nid in self.nodes])
        
        if direction in ["out", "both"]:
            # Citations that this one references
            successors = list(self.graph.successors(citation_id))
            neighbors.extend([self.nodes[nid] for nid in successors if nid in self.nodes])
        
        return neighbors
    
    def get_citation_chains(self, citation_id: str, max_depth: int = 5) -> List[List[CitationNode]]:
        """Find citation chains (paths) starting from a given citation"""
        chains = []
        
        try:
            # Find all simple paths from this citation (limiting depth)
            for target_id in self.graph.nodes():
                if target_id != citation_id:
                    try:
                        paths = list(nx.all_simple_paths(
                            self.graph, 
                            citation_id, 
                            target_id, 
                            cutoff=max_depth
                        ))
                        
                        for path in paths:
                            chain = [self.nodes[node_id] for node_id in path if node_id in self.nodes]
                            if len(chain) > 1:  # Only include actual chains
                                chains.append(chain)
                    except nx.NetworkXNoPath:
                        continue
                        
        except Exception as e:
            logger.warning(f"Error finding citation chains: {e}")
        
        return chains
    
    def analyze_network_metrics(self) -> Dict[str, Any]:
        """Compute network analysis metrics for the citation graph"""
        if not self.graph.nodes():
            return {"error": "Empty graph"}
        
        try:
            metrics = {}
            
            # Basic graph properties
            metrics["node_count"] = self.graph.number_of_nodes()
            metrics["edge_count"] = self.graph.number_of_edges()
            metrics["density"] = nx.density(self.graph)
            metrics["is_connected"] = nx.is_weakly_connected(self.graph)
            
            # Centrality measures
            if self.graph.number_of_nodes() > 1:
                metrics["degree_centrality"] = nx.degree_centrality(self.graph)
                metrics["in_degree_centrality"] = nx.in_degree_centrality(self.graph)
                metrics["out_degree_centrality"] = nx.out_degree_centrality(self.graph)
                
                # Authority and hub scores (for citation networks)
                try:
                    hubs, authorities = nx.hits(self.graph, max_iter=100)
                    metrics["authority_scores"] = authorities
                    metrics["hub_scores"] = hubs
                except:
                    logger.warning("Could not compute HITS algorithm scores")
                
                # PageRank (another authority measure)
                try:
                    metrics["pagerank_scores"] = nx.pagerank(self.graph, max_iter=100)
                except:
                    logger.warning("Could not compute PageRank scores")
            
            # Community detection (for larger graphs)
            if self.graph.number_of_nodes() > 5:
                try:
                    # Convert to undirected for community detection
                    undirected = self.graph.to_undirected()
                    communities = nx.community.greedy_modularity_communities(undirected)
                    metrics["community_count"] = len(communities)
                    metrics["communities"] = [list(community) for community in communities]
                except:
                    logger.warning("Could not detect communities")
            
            # Citation-specific metrics
            full_citations = [nid for nid, node in self.nodes.items() if node.is_full_citation()]
            reference_citations = [nid for nid, node in self.nodes.items() if node.is_reference_citation()]
            
            metrics["full_citation_count"] = len(full_citations)
            metrics["reference_citation_count"] = len(reference_citations)
            metrics["relationship_types"] = list(self.metadata["relationship_types"])
            
            # Most cited cases (highest in-degree)
            in_degrees = dict(self.graph.in_degree())
            most_cited = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            metrics["most_cited_citations"] = [
                {
                    "citation_id": cid,
                    "citation_text": self.nodes[cid].citation_text,
                    "reference_count": count
                }
                for cid, count in most_cited if cid in self.nodes and count > 0
            ]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing network metrics: {e}")
            return {"error": str(e)}
    
    def export_for_visualization(self, format_type: str = "json") -> Dict[str, Any]:
        """
        Export graph in format suitable for visualization libraries
        
        Args:
            format_type: "json" (D3.js), "cytoscape", "graphml", "dot"
        """
        if format_type == "json":
            return self._export_d3_json()
        elif format_type == "cytoscape":
            return self._export_cytoscape()
        elif format_type == "graphml":
            return self._export_graphml()
        elif format_type == "dot":
            return self._export_dot()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_d3_json(self) -> Dict[str, Any]:
        """Export in D3.js force-directed graph format"""
        nodes = []
        links = []
        
        # Create nodes
        for citation_id, node in self.nodes.items():
            d3_node = {
                "id": citation_id,
                "label": node.citation_text[:100] + "..." if len(node.citation_text) > 100 else node.citation_text,
                "type": node.citation_type,
                "group": node.document_id,
                "size": 10 + (node.confidence_score * 20 if node.confidence_score else 10),
                "title": node.citation_text,
                **node.to_dict()
            }
            nodes.append(d3_node)
        
        # Create links
        for (source_id, target_id), edge in self.edges.items():
            d3_link = {
                "source": source_id,
                "target": target_id,
                "relationship": edge.relationship_type,
                "strength": edge.confidence_score,
                "title": f"{edge.relationship_type} (confidence: {edge.confidence_score:.2f})",
                **edge.to_dict()
            }
            links.append(d3_link)
        
        return {
            "nodes": nodes,
            "links": links,
            "metadata": {
                **self.metadata,
                "relationship_types": list(self.metadata["relationship_types"])
            }
        }
    
    def _export_cytoscape(self) -> Dict[str, Any]:
        """Export in Cytoscape.js format"""
        elements = []
        
        # Add nodes
        for citation_id, node in self.nodes.items():
            element = {
                "data": {
                    "id": citation_id,
                    "label": node.citation_text[:50] + "..." if len(node.citation_text) > 50 else node.citation_text,
                    **node.to_dict()
                },
                "classes": node.citation_type.lower()
            }
            elements.append(element)
        
        # Add edges
        for (source_id, target_id), edge in self.edges.items():
            element = {
                "data": {
                    "id": f"{source_id}-{target_id}",
                    "source": source_id,
                    "target": target_id,
                    **edge.to_dict()
                },
                "classes": edge.relationship_type.replace("_", "-")
            }
            elements.append(element)
        
        return {
            "elements": elements,
            "metadata": {
                **self.metadata,
                "relationship_types": list(self.metadata["relationship_types"])
            }
        }
    
    def _export_graphml(self) -> str:
        """Export as GraphML XML string"""
        try:
            return nx.generate_graphml(self.graph)
        except Exception as e:
            logger.error(f"Failed to export GraphML: {e}")
            return f"<error>Failed to export GraphML: {e}</error>"
    
    def _export_dot(self) -> str:
        """Export as DOT format for Graphviz"""
        try:
            return nx.nx_agraph.to_agraph(self.graph).to_string()
        except Exception as e:
            logger.error(f"Failed to export DOT: {e}")
            # Fallback to simple DOT format
            lines = ["digraph CitationGraph {"]
            
            # Add nodes
            for citation_id, node in self.nodes.items():
                label = node.citation_text.replace('"', '\\"')[:50]
                lines.append(f'  "{citation_id}" [label="{label}"];')
            
            # Add edges
            for (source_id, target_id), edge in self.edges.items():
                lines.append(f'  "{source_id}" -> "{target_id}" [label="{edge.relationship_type}"];')
            
            lines.append("}")
            return "\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        stats = {
            "metadata": {
                **self.metadata,
                "relationship_types": list(self.metadata["relationship_types"])
            },
            "network_metrics": self.analyze_network_metrics(),
            "document_distribution": {
                doc_id: len(citation_ids) 
                for doc_id, citation_ids in self.document_map.items()
            },
            "citation_type_distribution": {},
            "relationship_type_distribution": {}
        }
        
        # Citation type distribution
        type_counts = defaultdict(int)
        for node in self.nodes.values():
            type_counts[node.citation_type] += 1
        stats["citation_type_distribution"] = dict(type_counts)
        
        # Relationship type distribution
        rel_counts = defaultdict(int)
        for edge in self.edges.values():
            rel_counts[edge.relationship_type] += 1
        stats["relationship_type_distribution"] = dict(rel_counts)
        
        return stats


class CitationGraphBuilder:
    """Service for building citation graphs from database queries"""
    
    async def build_graph_from_documents(
        self, 
        document_ids: List[str], 
        db_session: AsyncSession,
        include_embeddings: bool = False
    ) -> CitationGraph:
        """
        Build a citation graph from a list of documents
        
        Args:
            document_ids: List of document UUIDs to include
            db_session: Database session
            include_embeddings: Whether to include citation embeddings
        """
        graph = CitationGraph()
        
        try:
            # Query all citations for the documents
            citation_query = select(Citation).where(Citation.document_id.in_(document_ids))
            citation_result = await db_session.execute(citation_query)
            citations = citation_result.scalars().all()
            
            # Add citation nodes
            citation_ids = []
            for citation in citations:
                graph.add_citation_node(citation)
                citation_ids.append(str(citation.id))
            
            # Query all relationships for these citations
            relationship_query = select(CitationRelationship).where(
                and_(
                    CitationRelationship.source_citation_id.in_(citation_ids),
                    CitationRelationship.target_citation_id.in_(citation_ids)
                )
            )
            relationship_result = await db_session.execute(relationship_query)
            relationships = relationship_result.scalars().all()
            
            # Add relationship edges
            for relationship in relationships:
                graph.add_citation_relationship(relationship)
            
            logger.info(f"Built citation graph with {len(citations)} nodes and {len(relationships)} edges")
            return graph
            
        except Exception as e:
            logger.error(f"Failed to build citation graph: {e}")
            raise
    
    async def build_document_citation_graph(
        self, 
        document_id: str, 
        db_session: AsyncSession,
        include_cross_document: bool = False
    ) -> CitationGraph:
        """
        Build a citation graph for a single document
        
        Args:
            document_id: Document UUID
            db_session: Database session
            include_cross_document: Whether to include citations from other documents
        """
        if include_cross_document:
            # Find all related documents through citation relationships
            related_docs = await self._find_related_documents(document_id, db_session)
            return await self.build_graph_from_documents(related_docs, db_session)
        else:
            return await self.build_graph_from_documents([document_id], db_session)
    
    async def _find_related_documents(self, document_id: str, db_session: AsyncSession) -> List[str]:
        """Find documents related through citation relationships"""
        try:
            # Start with the primary document
            related_docs = {document_id}
            
            # Find citations in this document
            citation_query = select(Citation.id).where(Citation.document_id == document_id)
            citation_result = await db_session.execute(citation_query)
            citation_ids = [str(cid[0]) for cid in citation_result.fetchall()]
            
            if not citation_ids:
                return [document_id]
            
            # Find relationships involving these citations
            relationship_query = select(
                CitationRelationship.source_citation_id,
                CitationRelationship.target_citation_id
            ).where(
                or_(
                    CitationRelationship.source_citation_id.in_(citation_ids),
                    CitationRelationship.target_citation_id.in_(citation_ids)
                )
            )
            relationship_result = await db_session.execute(relationship_query)
            relationships = relationship_result.fetchall()
            
            # Find all citation IDs involved
            all_citation_ids = set(citation_ids)
            for source_id, target_id in relationships:
                all_citation_ids.add(str(source_id))
                all_citation_ids.add(str(target_id))
            
            # Find documents for all these citations
            doc_query = select(Citation.document_id.distinct()).where(
                Citation.id.in_(list(all_citation_ids))
            )
            doc_result = await db_session.execute(doc_query)
            for doc_id_tuple in doc_result.fetchall():
                related_docs.add(str(doc_id_tuple[0]))
            
            return list(related_docs)
            
        except Exception as e:
            logger.error(f"Error finding related documents: {e}")
            return [document_id]


# Global service instance
citation_graph_builder = CitationGraphBuilder() 