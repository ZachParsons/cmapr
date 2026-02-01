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
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
        }}

        #container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }}

        h1 {{
            margin: 0 0 20px 0;
            color: #333;
        }}

        #graph {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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

        .tooltip {{
            position: absolute;
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

        .controls {{
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }}

        button {{
            padding: 8px 16px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}

        button:hover {{
            background: #45a049;
        }}

        .info {{
            margin-top: 20px;
            padding: 12px;
            background: white;
            border-radius: 4px;
            border: 1px solid #ddd;
            font-size: 14px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div id="container">
        <h1>{title}</h1>
        <svg id="graph" width="{width}" height="{height}"></svg>
        <div class="controls">
            <button onclick="resetZoom()">Reset Zoom</button>
            <button onclick="toggleLabels()">Toggle Labels</button>
            <button onclick="restartSimulation()">Restart Layout</button>
        </div>
        <div class="info" id="info">
            Loading graph...
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <script>
        const width = {width};
        const height = {height};
        let showLabels = true;

        // Color scale for communities
        const color = d3.scaleOrdinal(d3.schemeCategory10);

        // Create SVG
        const svg = d3.select("#graph");
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

            // Create links
            const link = g.append("g")
                .selectAll("line")
                .data(data.links)
                .join("line")
                .attr("class", "link")
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

            // Create labels
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
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);

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

        function toggleLabels() {{
            showLabels = !showLabels;
            window.label.style("opacity", showLabels ? 1 : 0);
        }}

        function restartSimulation() {{
            window.simulation.alpha(1).restart();
        }}
    </script>
</body>
</html>"""
