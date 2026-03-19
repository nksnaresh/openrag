from __future__ import annotations

import threading
from typing import Any

import networkx as nx

from openrag.models.graph import EdgeType, GraphEdge, GraphNode, NodeType, SubGraph
from openrag.storage.base import BaseGraphDBAdapter


class NetworkXAdapter(BaseGraphDBAdapter):
    """In-memory directed graph adapter backed by NetworkX.

    Each namespace gets its own `networkx.DiGraph`. All operations are
    synchronous under the hood but wrapped in async for interface compliance.
    """

    def __init__(self, config: object = None, **kwargs: Any) -> None:
        # namespace → nx.DiGraph
        self._graphs: dict[str, nx.DiGraph] = {}
        self._lock = threading.Lock()

    def _graph(self, namespace: str) -> nx.DiGraph:
        """Return (or create) the DiGraph for the given namespace."""
        if namespace not in self._graphs:
            self._graphs[namespace] = nx.DiGraph()
        return self._graphs[namespace]

    async def initialize(self) -> None:
        """No-op for in-memory adapter."""

    async def upsert_node(self, namespace: str, node: GraphNode) -> None:
        """Insert or update a graph node (edges are preserved on update)."""
        with self._lock:
            g = self._graph(namespace)
            g.add_node(
                node.node_id,
                node_type=node.node_type.value,
                label=node.label,
                tenant_id=node.tenant_id,
                namespace=namespace,
                **node.properties,
            )

    async def upsert_edge(self, namespace: str, edge: GraphEdge) -> None:
        """Insert or update a directed edge between two nodes."""
        with self._lock:
            g = self._graph(namespace)
            g.add_edge(
                edge.source_id,
                edge.target_id,
                edge_id=edge.edge_id,
                edge_type=edge.edge_type.value,
                weight=edge.weight,
                **edge.properties,
            )

    async def get_node(self, node_id: str) -> GraphNode | None:
        """Retrieve a single node by ID (searches all namespaces)."""
        with self._lock:
            for g in self._graphs.values():
                if g.has_node(node_id):
                    attrs = dict(g.nodes[node_id])
                    return self._node_from_attrs(node_id, attrs)
        return None

    async def traverse(
        self,
        start_node_id: str,
        depth: int = 2,
        edge_types: list[str] | None = None,
    ) -> SubGraph:
        """BFS from start_node_id up to `depth` hops, optionally filtered by edge type."""
        with self._lock:
            target_graph: nx.DiGraph | None = None
            for g in self._graphs.values():
                if g.has_node(start_node_id):
                    target_graph = g
                    break
            if target_graph is None:
                return SubGraph()

            # BFS limited to depth
            visited_nodes: set[str] = set()
            visited_edges: list[GraphEdge] = []
            queue = [(start_node_id, 0)]
            visited_nodes.add(start_node_id)

            while queue:
                current, current_depth = queue.pop(0)
                if current_depth >= depth:
                    continue
                for neighbor in target_graph.successors(current):
                    edge_data = target_graph.edges[current, neighbor]
                    etype = edge_data.get("edge_type", "")
                    if edge_types and etype not in edge_types:
                        continue
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        queue.append((neighbor, current_depth + 1))
                    visited_edges.append(
                        GraphEdge(
                            edge_id=edge_data.get("edge_id", f"{current}->{neighbor}"),
                            source_id=current,
                            target_id=neighbor,
                            edge_type=EdgeType(etype) if etype else EdgeType.CONTAINS,
                            weight=float(edge_data.get("weight", 1.0)),
                            properties={
                                k: v for k, v in edge_data.items()
                                if k not in ("edge_id", "edge_type", "weight")
                            },
                        )
                    )

            nodes = [
                self._node_from_attrs(nid, dict(target_graph.nodes[nid]))
                for nid in visited_nodes
                if target_graph.has_node(nid)
            ]
            return SubGraph(nodes=nodes, edges=visited_edges)

    async def find_nodes(
        self,
        namespace: str,
        node_type: str | None = None,
        label_contains: str | None = None,
        properties: dict[str, Any] | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        """Search for nodes in a namespace matching the given criteria."""
        with self._lock:
            g = self._graphs.get(namespace)
            if g is None:
                return []

            results: list[GraphNode] = []
            for node_id, attrs in g.nodes(data=True):
                if node_type and attrs.get("node_type") != node_type:
                    continue
                if label_contains and label_contains.lower() not in str(
                    attrs.get("label", "")
                ).lower():
                    continue
                if properties and not all(attrs.get(k) == v for k, v in properties.items()):
                    continue
                results.append(self._node_from_attrs(node_id, dict(attrs)))
                if len(results) >= limit:
                    break
            return results

    async def search_context(
        self, query: str, namespace: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Simple keyword-based search for entity context in the graph."""
        keywords = set(query.lower().split())
        # Greedy search through nodes in this namespace
        nodes = await self.find_nodes(namespace, limit=limit * 10)
        
        results = []
        for node in nodes:
            node_text = (node.label + " " + " ".join(str(v) for v in node.properties.values())).lower()
            overlap = len(keywords.intersection(set(node_text.split())))
            if overlap > 0:
                results.append({
                    "id": node.node_id,
                    "content": f"Entity [{node.label}]: {node.properties.get('description', 'No description available')}",
                    "score": float(overlap),
                    "namespace": namespace,
                    "mode": "graph"
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and all its incident edges (searches all namespaces)."""
        with self._lock:
            for g in self._graphs.values():
                if g.has_node(node_id):
                    g.remove_node(node_id)
                    return

    async def close(self) -> None: pass

    async def clear_namespace(self, namespace: str) -> None:
        """Remove the entire graph for a namespace (useful for tests)."""
        with self._lock:
            self._graphs.pop(namespace, None)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _node_from_attrs(node_id: str, attrs: dict[str, Any]) -> GraphNode:
        """Reconstruct a GraphNode from stored NetworkX attributes."""
        raw_type = attrs.pop("node_type", NodeType.ENTITY.value)
        try:
            node_type = NodeType(raw_type)
        except ValueError:
            node_type = NodeType.ENTITY

        return GraphNode(
            node_id=node_id,
            node_type=node_type,
            label=str(attrs.pop("label", node_id)),
            tenant_id=str(attrs.pop("tenant_id", "default")),
            namespace=str(attrs.pop("namespace", "default")),
            properties=attrs,
        )
