"""Spatial structure graphs for Edinburgh 2011 Intermediate Zones.

The three graphs share node_index / IntZone. They are not merged:

- graph.geo: undirected rook adjacency (shared boundary)
- graph.road: directed road network-distance graph
- graph.mobility: directed observed origin-destination graph

    PYTHONPATH=src python -m graph.geo
    PYTHONPATH=src python -m graph.road
    PYTHONPATH=src python -m graph.mobility
"""
