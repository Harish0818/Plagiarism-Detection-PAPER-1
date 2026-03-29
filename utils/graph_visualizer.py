import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from datetime import datetime
import json
import math
from collections import defaultdict
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AdvancedGraphVisualizer:
    def __init__(self):
        # Color schemes
        self.color_palettes = {
            'default': {
                'paper': '#1f77b4',
                'highly_cited': '#ff7f0e',
                'influential': '#2ca02c',
                'problematic': '#d62728',
                'recent': '#9467bd',
                'review': '#8c564b',
                'thesis': '#e377c2'
            },
            'cool': {
                'paper': '#2E86AB',
                'highly_cited': '#A23B72',
                'influential': '#F18F01',
                'problematic': '#C73E1D',
                'recent': '#6A8EAE',
                'review': '#57A773',
                'thesis': '#3D348B'
            },
            'warm': {
                'paper': '#FF6B35',
                'highly_cited': '#004E89',
                'influential': '#FF9F1C',
                'problematic': '#EF476F',
                'recent': '#06D6A0',
                'review': '#118AB2',
                'thesis': '#073B4C'
            }
        }
        
        self.current_palette = 'default'
        
        # Layout algorithms
        self.layouts = {
            'spring': nx.spring_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'spectral': nx.spectral_layout,
            'circular': nx.circular_layout,
            'shell': nx.shell_layout,
            'force_atlas': None  # Would use fa2 if available
        }
    
    def create_citation_graph(self, edges: List[Tuple], papers: Optional[List[Dict]] = None) -> nx.DiGraph:
        """Create an enhanced citation graph with metadata"""
        G = nx.DiGraph()
        
        # Process edges with weights and timestamps
        edge_metadata = defaultdict(list)
        
        for edge in edges:
            if len(edge) >= 2:
                source, target = edge[0], edge[1]
                edge_metadata[(source, target)].append({
                    'count': edge_metadata[(source, target)][-1]['count'] + 1 
                    if edge_metadata[(source, target)] else 1,
                    'context': edge[2] if len(edge) > 2 else '',
                    'timestamp': edge[3] if len(edge) > 3 else datetime.now()
                })
        
        # Add edges with aggregated metadata
        for (source, target), metadata_list in edge_metadata.items():
            total_count = sum(m['count'] for m in metadata_list)
            contexts = [m['context'] for m in metadata_list if m['context']]
            
            G.add_edge(source, target, 
                      weight=total_count,
                      contexts=contexts[:5],  # Keep top contexts
                      first_cited=min(m['timestamp'] for m in metadata_list),
                      last_cited=max(m['timestamp'] for m in metadata_list))
        
        # Add node metadata if provided
        if papers:
            for paper in papers:
                if isinstance(paper, dict) and 'title' in paper:
                    node_id = paper.get('id', paper['title'][:50])
                    G.add_node(node_id, **paper)
        
        # Calculate graph metrics
        self._calculate_graph_metrics(G)
        
        return G
    
    def _calculate_graph_metrics(self, G: nx.DiGraph):
        """Calculate and store graph metrics"""
        if len(G.nodes()) == 0:
            return
        
        # Centrality measures
        try:
            pr = nx.pagerank(G, alpha=0.85)
            nx.set_node_attributes(G, pr, 'pagerank')
        except:
            pass
        
        try:
            betweenness = nx.betweenness_centrality(G)
            nx.set_node_attributes(G, betweenness, 'betweenness')
        except:
            pass
        
        try:
            closeness = nx.closeness_centrality(G)
            nx.set_node_attributes(G, closeness, 'closeness')
        except:
            pass
        
        # Degree measures
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        
        nx.set_node_attributes(G, in_degree, 'in_degree')
        nx.set_node_attributes(G, out_degree, 'out_degree')
        nx.set_node_attributes(G, {n: in_degree[n] + out_degree[n] 
                                  for n in G.nodes()}, 'total_degree')
    
    def visualize_graph(self, edges: List[Tuple], papers: Optional[List[Dict]] = None,
                       layout: str = 'spring', figsize: Tuple[int, int] = (14, 10),
                       title: str = "Citation Network Analysis") -> plt.Figure:
        """Create advanced matplotlib visualization"""
        G = self.create_citation_graph(edges, papers)
        
        if len(G.nodes()) == 0:
            return self._create_empty_plot(title)
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        
        # Create grid for main plot and metrics
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[2, 1, 1])
        
        # Main graph plot
        ax_main = fig.add_subplot(gs[0, :])
        
        # Side plots for metrics
        ax_metrics1 = fig.add_subplot(gs[1, 0])
        ax_metrics2 = fig.add_subplot(gs[1, 1])
        ax_metrics3 = fig.add_subplot(gs[1, 2])
        
        # Get layout positions
        pos = self._get_layout(G, layout)
        
        # Calculate node properties
        node_sizes, node_colors, node_labels = self._calculate_node_properties(G)
        
        # Draw edges with varying widths and transparency
        edge_widths = [2 + np.log1p(G[u][v]['weight']) for u, v in G.edges()]
        edge_alphas = [0.3 + 0.7 * (G[u][v]['weight'] / max(edge_widths)) 
                      for u, v in G.edges()]
        
        # Draw edges
        for (u, v), width, alpha in zip(G.edges(), edge_widths, edge_alphas):
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=width,
                alpha=alpha,
                edge_color='#888888',
                arrows=True,
                arrowsize=20,
                arrowstyle='-|>',
                connectionstyle='arc3,rad=0.1',
                ax=ax_main
            )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.9,
            edgecolors='white',
            linewidths=1.5,
            ax=ax_main
        )
        
        # Draw labels with smart positioning
        self._draw_smart_labels(G, pos, ax_main)
        
        # Add title
        ax_main.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Remove axis
        ax_main.axis('off')
        
        # Create metrics visualizations
        self._create_metrics_visualizations(G, [ax_metrics1, ax_metrics2, ax_metrics3])
        
        # Add color legend
        self._add_color_legend(ax_main)
        
        # Add statistics text
        stats_text = self._get_graph_statistics(G)
        fig.text(0.02, 0.98, stats_text, transform=fig.transFigure,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        return fig
    
    def visualize_interactive_graph(self, edges: List[Tuple], papers: Optional[List[Dict]] = None,
                                  layout: str = 'spring', height: int = 700) -> go.Figure:
        """Create a dedicated interactive Plotly Network Graph without subplots."""
        G = self.create_citation_graph(edges, papers)
        
        if len(G.nodes()) == 0:
            return self._create_empty_plotly()
        
        # 1. Generate node positions using NetworkX layout
        pos = self._get_layout(G, layout)
        
        # 2. Create Edge Traces (Lines)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None]) # None prevents connecting separate lines
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # 3. Create Node Traces (Markers)
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        
        # Calculate colors and sizes based on node properties
        sizes, colors, _ = self._calculate_node_properties(G)

        for i, node in enumerate(G.nodes()):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"<b>{str(node)[:50]}</b>") # Tooltip content
            node_color.append(colors[i])
            # Scale down for Plotly markers (e.g., divide by 15 for better clarity)
            node_size.append(sizes[i] / 15) 

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[str(n)[:15] for n in G.nodes()], # Label on the graph
            textposition="top center",
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line_width=2)
        )

        # 4. Final Figure Assembly
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        # FIX: Correctly nested title properties
                        title=dict(
                            text='Citation Network Analysis',
                            font=dict(size=16)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        # Remove axes for a clean "Proper View" graph
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=height
                    ))
        return fig
    
    def visualize_3d_graph(self, edges: List[Tuple], papers: Optional[List[Dict]] = None) -> go.Figure:
        """Create 3D visualization of citation network"""
        G = self.create_citation_graph(edges, papers)
        
        if len(G.nodes()) == 0:
            return self._create_empty_plotly()
        
        # Create 3D layout
        pos = nx.spring_layout(G, dim=3, seed=42)
        
        # Prepare 3D data
        node_x, node_y, node_z = [], [], []
        node_text, node_size, node_color = [], [], []
        
        # Calculate node properties
        node_sizes_2d, node_colors_2d, _ = self._calculate_node_properties(G)
        
        for i, node in enumerate(G.nodes()):
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            
            # Node info
            info = f"<b>{str(node)[:50]}</b><br>"
            if 'in_degree' in G.nodes[node]:
                info += f"Citations: {G.nodes[node]['in_degree']}<br>"
            if 'pagerank' in G.nodes[node]:
                info += f"Influence: {G.nodes[node]['pagerank']:.3f}"
            
            node_text.append(info)
            node_size.append(node_sizes_2d[i] * 0.5)  # Scale down for 3D
            node_color.append(node_colors_2d[i])
        
        # Create 3D node trace
        node_trace_3d = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            text=[str(n)[:20] for n in G.nodes()],
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                opacity=0.9,
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        # Create edge traces
        edge_x, edge_y, edge_z = [], [], []
        
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        edge_trace_3d = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            showlegend=False
        )
        
        # Create 3D figure
        fig = go.Figure(data=[edge_trace_3d, node_trace_3d])
        
        fig.update_layout(
            title="3D Citation Network",
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                bgcolor='white'
            ),
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_community_graph(self, edges: List[Tuple], papers: Optional[List[Dict]] = None) -> go.Figure:
        """Visualize citation network with community detection"""
        G = self.create_citation_graph(edges, papers)
        
        if len(G.nodes()) < 3:
            return self._create_empty_plotly()
        
        # Detect communities (simplified)
        try:
            # Use Louvain method for community detection
            import community as community_louvain
            partition = community_louvain.best_partition(G.to_undirected())
        except:
            # Fallback: assign random communities
            partition = {node: i % 5 for i, node in enumerate(G.nodes())}
        
        # Add community info to nodes
        for node, comm in partition.items():
            G.nodes[node]['community'] = comm
        
        # Get layout
        pos = nx.spring_layout(G, seed=42)
        
        # Create traces for each community
        traces = []
        
        # Get unique communities
        communities = set(partition.values())
        colors = px.colors.qualitative.Set3
        
        for i, comm in enumerate(communities):
            # Get nodes in this community
            comm_nodes = [node for node in G.nodes() if partition[node] == comm]
            
            if not comm_nodes:
                continue
            
            # Get positions for these nodes
            x_vals = [pos[node][0] for node in comm_nodes]
            y_vals = [pos[node][1] for node in comm_nodes]
            
            # Get node properties
            sizes = []
            texts = []
            for node in comm_nodes:
                # Size based on degree
                size = 10 + G.nodes[node].get('in_degree', 0) * 2
                sizes.append(size)
                
                # Text
                text = f"<b>{str(node)[:40]}</b><br>"
                text += f"Community: {comm}<br>"
                if 'in_degree' in G.nodes[node]:
                    text += f"Citations: {G.nodes[node]['in_degree']}"
                texts.append(text)
            
            # Create trace for this community
            trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers+text',
                text=[str(n)[:15] for n in comm_nodes],
                hovertext=texts,
                hoverinfo='text',
                marker=dict(
                    size=sizes,
                    color=colors[i % len(colors)],
                    line=dict(width=2, color='white')
                ),
                name=f"Community {comm + 1}",
                showlegend=True
            )
            
            traces.append(trace)
        
        # Create edges trace
        edge_x, edge_y = [], []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='gray'),
            hoverinfo='none',
            showlegend=False
        )
        
        # Combine all traces
        data = [edge_trace] + traces
        
        # Create figure
        fig = go.Figure(data=data)
        
        fig.update_layout(
            title="Citation Network with Community Detection",
            showlegend=True,
            hovermode='closest',
            height=700,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _get_layout(self, G: nx.DiGraph, layout_name: str) -> Dict:
        """Get node positions using specified layout algorithm"""
        if layout_name in self.layouts and self.layouts[layout_name]:
            try:
                return self.layouts[layout_name](G, seed=42)
            except:
                pass
        
        # Fallback to spring layout
        return nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    
    def _calculate_node_properties(self, G: nx.DiGraph) -> Tuple[List, List, Dict]:
        """Calculate node sizes, colors, and labels"""
        node_sizes = []
        node_colors = []
        node_labels = {}
        
        # Get metrics
        in_degrees = nx.get_node_attributes(G, 'in_degree')
        pageranks = nx.get_node_attributes(G, 'pagerank')
        
        max_in_degree = max(in_degrees.values()) if in_degrees else 1
        max_pagerank = max(pageranks.values()) if pageranks else 1
        
        for node in G.nodes():
            # Size based on in-degree
            size = 300 + 2000 * (in_degrees.get(node, 0) / max_in_degree)
            node_sizes.append(size)
            
            # Color based on PageRank and other factors
            pagerank = pageranks.get(node, 0)
            in_degree = in_degrees.get(node, 0)
            
            # Determine node type
            if pagerank > 0.8 * max_pagerank:
                color = self.color_palettes[self.current_palette]['influential']
            elif in_degree > 0.5 * max_in_degree:
                color = self.color_palettes[self.current_palette]['highly_cited']
            elif 'problematic' in str(node).lower():
                color = self.color_palettes[self.current_palette]['problematic']
            elif G.nodes[node].get('type') == 'review':
                color = self.color_palettes[self.current_palette]['review']
            elif G.nodes[node].get('type') == 'thesis':
                color = self.color_palettes[self.current_palette]['thesis']
            elif G.nodes[node].get('year', 0) > 2018:
                color = self.color_palettes[self.current_palette]['recent']
            else:
                color = self.color_palettes[self.current_palette]['paper']
            
            node_colors.append(color)
            
            # Create label (truncate if too long)
            label = str(node)
            if len(label) > 30:
                label = label[:27] + '...'
            node_labels[node] = label
        
        return node_sizes, node_colors, node_labels
    
    def _draw_smart_labels(self, G: nx.DiGraph, pos: Dict, ax):
        """Draw labels with smart positioning to avoid overlap"""
        labels = {}
        for node in G.nodes():
            label = str(node)
            if len(label) > 30:
                label = label[:27] + '...'
            labels[node] = label
        
        # Draw important labels first (high degree nodes)
        important_nodes = sorted(G.nodes(), 
                                key=lambda n: G.nodes[n].get('in_degree', 0), 
                                reverse=True)[:10]
        
        for node in important_nodes:
            x, y = pos[node]
            ax.text(x, y + 0.03, labels[node],
                   fontsize=9, fontweight='bold',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    def _create_metrics_visualizations(self, G: nx.DiGraph, axes: List):
        """Create metrics visualizations in subplots"""
        # Plot 1: Degree distribution
        in_degrees = [d for n, d in G.in_degree()]
        if in_degrees:
            axes[0].hist(in_degrees, bins=20, edgecolor='black', alpha=0.7)
            axes[0].set_title('Citation Distribution', fontsize=10)
            axes[0].set_xlabel('Number of Citations')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Centrality scatter
        if hasattr(G, 'pagerank') and hasattr(G, 'betweenness'):
            pageranks = [G.nodes[n].get('pagerank', 0) for n in G.nodes()]
            betweenness = [G.nodes[n].get('betweenness', 0) for n in G.nodes()]
            
            axes[1].scatter(pageranks, betweenness, alpha=0.6)
            axes[1].set_title('Centrality Comparison', fontsize=10)
            axes[1].set_xlabel('PageRank')
            axes[1].set_ylabel('Betweenness')
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Top papers
        top_nodes = sorted(G.nodes(), 
                          key=lambda n: G.nodes[n].get('in_degree', 0), 
                          reverse=True)[:5]
        
        if top_nodes:
            labels = [str(n)[:20] + '...' if len(str(n)) > 20 else str(n) 
                     for n in top_nodes]
            values = [G.nodes[n].get('in_degree', 0) for n in top_nodes]
            
            axes[2].barh(labels, values, alpha=0.7)
            axes[2].set_title('Top Cited Papers', fontsize=10)
            axes[2].set_xlabel('Citations')
            axes[2].tick_params(axis='y', labelsize=8)
    
    def _add_color_legend(self, ax):
        """Add color legend to plot"""
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=self.color_palettes[self.current_palette]['paper'], 
                  markersize=10, label='Regular Paper'),
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=self.color_palettes[self.current_palette]['highly_cited'], 
                  markersize=10, label='Highly Cited'),
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=self.color_palettes[self.current_palette]['influential'], 
                  markersize=10, label='Influential'),
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=self.color_palettes[self.current_palette]['recent'], 
                  markersize=10, label='Recent'),
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=self.color_palettes[self.current_palette]['problematic'], 
                  markersize=10, label='Problematic')
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    def _get_graph_statistics(self, G: nx.DiGraph) -> str:
        """Get graph statistics as formatted string"""
        stats = [
            f"Graph Statistics:",
            f"Nodes: {G.number_of_nodes()}",
            f"Edges: {G.number_of_edges()}",
            f"Density: {nx.density(G):.4f}",
            f"Avg. Degree: {np.mean([d for n, d in G.degree()]):.1f}"
        ]
        
        if hasattr(G, 'pagerank'):
            max_pr_node = max(G.nodes(), key=lambda n: G.nodes[n].get('pagerank', 0))
            max_pr = G.nodes[max_pr_node].get('pagerank', 0)
            stats.append(f"Max PageRank: {max_pr:.3f}")
        
        return '\n'.join(stats)
    
    def _add_centrality_heatmap(self, G: nx.DiGraph, fig: go.Figure, row: int, col: int):
        """Add centrality heatmap to subplot"""
        # Create centrality matrix (simplified)
        nodes = list(G.nodes())
        centrality_matrix = []
        
        for i, node1 in enumerate(nodes[:10]):  # Limit to first 10 nodes
            row_data = []
            for j, node2 in enumerate(nodes[:10]):
                if i == j:
                    # Self similarity
                    row_data.append(G.nodes[node1].get('pagerank', 0))
                elif G.has_edge(node1, node2) or G.has_edge(node2, node1):
                    # Connected
                    row_data.append(0.5)
                else:
                    # Not connected
                    row_data.append(0.0)
            centrality_matrix.append(row_data)
        
        if centrality_matrix:
            heatmap = go.Heatmap(
                z=centrality_matrix,
                colorscale='Viridis',
                showscale=True
            )
            fig.add_trace(heatmap, row=row, col=col)
            fig.update_xaxes(title_text="Node", row=row, col=col)
            fig.update_yaxes(title_text="Node", row=row, col=col)
    
    def _add_temporal_analysis(self, G: nx.DiGraph, fig: go.Figure, row: int, col: int):
        """Add temporal analysis to subplot"""
        # Extract years from nodes
        years = []
        citations = []
        
        for node in G.nodes():
            year = G.nodes[node].get('year')
            if year and isinstance(year, (int, float)) and 1900 <= year <= 2025:
                years.append(year)
                citations.append(G.nodes[node].get('in_degree', 0))
        
        if years:
            # Create scatter plot
            scatter = go.Scatter(
                x=years,
                y=citations,
                mode='markers',
                marker=dict(size=8, opacity=0.6),
                text=[str(n)[:30] for n in G.nodes() if 'year' in G.nodes[n]]
            )
            
            fig.add_trace(scatter, row=row, col=col)
            fig.update_xaxes(title_text="Year", row=row, col=col)
            fig.update_yaxes(title_text="Citations", row=row, col=col)
    
    def _create_empty_plot(self, title: str = "") -> plt.Figure:
        """Create empty plot with message"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5, 0.5, 
            'No citation network data available', 
            ha='center', va='center', 
            fontsize=16, color='gray'
        )
        if title:
            ax.set_title(title, fontsize=14)
        plt.axis("off")
        return fig
    
    def _create_empty_plotly(self) -> go.Figure:
        """Create empty Plotly figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text="No citation network data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400
        )
        return fig
    
    def export_graph_data(self, G: nx.DiGraph, format: str = 'json') -> str:
        """Export graph data in various formats"""
        if format == 'json':
            data = nx.node_link_data(G)
            return json.dumps(data, indent=2, default=str)
        elif format == 'gexf':
            return '\n'.join(nx.generate_gexf(G))
        elif format == 'graphml':
            return '\n'.join(nx.generate_graphml(G))
        else:
            raise ValueError(f"Unsupported format: {format}")

# Module-level functions
def visualize_graph(edges: List[Tuple]) -> plt.Figure:
    visualizer = AdvancedGraphVisualizer()
    return visualizer.visualize_graph(edges)

def visualize_interactive_graph(edges: List[Tuple]) -> go.Figure:
    visualizer = AdvancedGraphVisualizer()
    return visualizer.visualize_interactive_graph(edges)

def visualize_3d_graph(edges: List[Tuple]) -> go.Figure:
    visualizer = AdvancedGraphVisualizer()
    return visualizer.visualize_3d_graph(edges)

def visualize_community_graph(edges: List[Tuple]) -> go.Figure:
    visualizer = AdvancedGraphVisualizer()
    return visualizer.create_community_graph(edges)