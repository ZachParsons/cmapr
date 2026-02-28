"""
HTML visualization generation.

This module generates standalone HTML files with D3.js force-directed
graph visualizations.
"""

import json
from pathlib import Path
from concept_mapper.graph.model import ConceptGraph
from concept_mapper.export.d3 import export_d3_json, to_d3_dict


def generate_html(
    graph: ConceptGraph,
    output_dir: Path,
    title: str = "Concept Network",
    width: int = 1200,
    height: int = 800,
    include_evidence: bool = False,
) -> Path:
    """
    Generate standalone HTML visualization of the graph.

    Creates an HTML file with embedded D3.js force-directed graph visualization.
    The visualization is interactive: nodes can be dragged, and hovering shows
    tooltips with node/edge information.

    Args:
        graph: ConceptGraph to visualize
        output_dir: Output directory
        title: Page title (default: "Concept Network")
        width: Visualization width in pixels (default: 1200)
        height: Visualization height in pixels (default: 800)
        include_evidence: Include evidence sentences in tooltips (default: False)

    Returns:
        Path to generated HTML file

    Example:
        >>> from concept_mapper.graph import ConceptGraph
        >>> from pathlib import Path
        >>> graph = ConceptGraph()
        >>> graph.add_node("consciousness")
        >>> graph.add_node("being")
        >>> graph.add_edge("consciousness", "being", weight=0.85)
        >>> html_path = generate_html(graph, Path("output/"))
        >>> print(f"Open {html_path} in browser")

    Features:
        - Force-directed layout with simulation
        - Interactive node dragging
        - Tooltips on hover
        - Color-coded communities
        - Node size by centrality/frequency
        - Edge width by weight
        - Zoom and pan support
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export graph data to JSON file (for reference)
    data_path = output_dir / "graph_data.json"
    export_d3_json(graph, data_path, include_evidence=include_evidence)

    # Get graph data as dict for inlining
    graph_data = to_d3_dict(graph, include_evidence=include_evidence)
    graph_data_json = json.dumps(graph_data, ensure_ascii=False)

    # Generate HTML file with inlined data
    html_path = output_dir / "index.html"

    html_content = _generate_html_template(
        title=title,
        width=width,
        height=height,
        graph_data_json=graph_data_json,
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_path


def _generate_html_template(
    title: str,
    width: int,
    height: int,
    graph_data_json: str,
) -> str:
    """Generate HTML template with D3.js visualization and inlined data."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
        }}

        #graph {{
            display: block;
            width: 100vw;
            height: 100vh;
            background: white;
        }}

        .node {{
            cursor: pointer;
            stroke: #fff;
            stroke-width: 1.5px;
        }}

        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}

        .node-label {{
            font-size: 10px;
            pointer-events: none;
            text-anchor: middle;
            fill: #333;
        }}

        .link-label {{
            font-size: 9px;
            pointer-events: none;
            text-anchor: middle;
            fill: #888;
        }}

        .tooltip {{
            position: fixed;
            padding: 8px 12px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            max-width: 300px;
        }}

        #overlay {{
            position: fixed;
            bottom: 16px;
            left: 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            align-items: flex-start;
        }}

        .controls {{
            display: flex;
            gap: 8px;
        }}

        button {{
            padding: 6px 14px;
            background: rgba(0,0,0,0.6);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
        }}

        button:hover {{
            background: rgba(0,0,0,0.8);
        }}

        .info {{
            padding: 6px 10px;
            background: rgba(0,0,0,0.5);
            border-radius: 4px;
            font-size: 12px;
            color: #eee;
        }}

        h1 {{
            position: fixed;
            top: 12px;
            left: 16px;
            margin: 0;
            font-size: 16px;
            color: #333;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <svg id="graph"></svg>

    <div id="overlay">
        <div class="info" id="info">Loading graph...</div>
        <div class="controls">
            <button onclick="resetZoom()">Reset Zoom</button>
            <button onclick="restartSimulation()">Restart Layout</button>
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <script>
        let width = window.innerWidth;
        let height = window.innerHeight;

        window.addEventListener("resize", () => {{
            width = window.innerWidth;
            height = window.innerHeight;
            svg.attr("width", width).attr("height", height);
            if (window.simulation) {{
                window.simulation.force("center", d3.forceCenter(width / 2, height / 2));
                window.simulation.alpha(0.3).restart();
            }}
        }});

        // Color scale for communities
        const color = d3.scaleOrdinal(d3.schemeCategory10);

        // Create SVG
        const svg = d3.select("#graph");

        // Arrowhead marker
        svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 10)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#999")
            .attr("fill-opacity", 0.6);

        const g = svg.append("g");

        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});

        svg.call(zoom);

        // Tooltip
        const tooltip = d3.select("#tooltip");

        // Inlined graph data (avoids CORS issues with file:// URLs)
        const data = {graph_data_json};

        // Initialize visualization
        (function() {{
            // Update info
            d3.select("#info").html(
                `Nodes: ${{data.nodes.length}} | Links: ${{data.links.length}}`
            );

            // Create force simulation
            const simulation = d3.forceSimulation(data.nodes)
                .force("link", d3.forceLink(data.links)
                    .id(d => d.id)
                    .distance(d => 100 / (d.weight || 1)))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(d => d.size * 5));

            // Node radius lookup (matches circle r attr below)
            const nodeRadius = d => Math.sqrt(d.size || 1) * 3 + 5;

            // Create links
            const link = g.append("g")
                .selectAll("line")
                .data(data.links)
                .join("line")
                .attr("class", "link")
                .attr("marker-end", "url(#arrowhead)")
                .attr("stroke-width", d => Math.sqrt(d.weight || 1) * 2)
                .on("mouseover", (event, d) => {{
                    let html = `<strong>${{d.source.id}} → ${{d.target.id}}</strong><br>`;
                    html += `Weight: ${{d.weight.toFixed(3)}}<br>`;
                    if (d.label) html += `Type: ${{d.label}}<br>`;
                    if (d.evidence) {{
                        html += `<br><em>${{d.evidence[0]}}</em>`;
                    }}
                    tooltip.html(html)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px")
                        .style("opacity", 1);
                }})
                .on("mouseout", () => {{
                    tooltip.style("opacity", 0);
                }});

            // Create nodes
            const node = g.append("g")
                .selectAll("circle")
                .data(data.nodes)
                .join("circle")
                .attr("class", "node")
                .attr("r", d => Math.sqrt(d.size || 1) * 3 + 5)
                .attr("fill", d => color(d.group || 0))
                .call(drag(simulation))
                .on("mouseover", (event, d) => {{
                    let html = `<strong>${{d.label}}</strong><br>`;
                    if (d.frequency) html += `Frequency: ${{d.frequency}}<br>`;
                    if (d.pos) html += `POS: ${{d.pos}}<br>`;
                    if (d.definition) html += `<br>${{d.definition}}`;
                    tooltip.html(html)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px")
                        .style("opacity", 1);
                }})
                .on("mouseout", () => {{
                    tooltip.style("opacity", 0);
                }});

            // Create edge labels
            const linkLabel = g.append("g")
                .selectAll("text")
                .data(data.links)
                .join("text")
                .attr("class", "link-label")
                .text(d => d.verb || d.label || "relates to");

            // Create node labels
            const label = g.append("g")
                .selectAll("text")
                .data(data.nodes)
                .join("text")
                .attr("class", "node-label")
                .text(d => d.label)
                .attr("dy", d => Math.sqrt(d.size || 1) * 3 + 20);

            // Update positions on simulation tick
            simulation.on("tick", () => {{
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => {{
                        const dx = d.target.x - d.source.x;
                        const dy = d.target.y - d.source.y;
                        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                        const r = nodeRadius(d.target) + 8; // +8 for arrowhead
                        return d.target.x - (dx / dist) * r;
                    }})
                    .attr("y2", d => {{
                        const dx = d.target.x - d.source.x;
                        const dy = d.target.y - d.source.y;
                        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                        const r = nodeRadius(d.target) + 8;
                        return d.target.y - (dy / dist) * r;
                    }});

                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);

                linkLabel
                    .attr("x", d => (d.source.x + d.target.x) / 2)
                    .attr("y", d => (d.source.y + d.target.y) / 2);

                label
                    .attr("x", d => d.x)
                    .attr("y", d => d.y);
            }});

            // Store simulation for controls
            window.simulation = simulation;
            window.label = label;
        }})();

        // Drag behavior
        function drag(simulation) {{
            function dragstarted(event) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }}

            function dragged(event) {{
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }}

            function dragended(event) {{
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }}

            return d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        }}

        // Control functions
        function resetZoom() {{
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity);
        }}

        function restartSimulation() {{
            window.simulation.alpha(1).restart();
        }}
    </script>
</body>
</html>"""
