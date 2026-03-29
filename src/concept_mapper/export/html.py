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
        >>> html_path = generate_html(graph, Path("data/output/"))
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
            background: #ffffff;
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

        .node-label {{
            pointer-events: none;
            text-anchor: middle;
            fill: #222;
            font-weight: 500;
        }}

        .link-label {{
            font-size: 9px;
            pointer-events: none;
            text-anchor: middle;
            fill: #555;
        }}

        .tooltip {{
            position: fixed;
            padding: 8px 12px;
            background: rgba(20, 20, 20, 0.88);
            color: #f0f0f0;
            border-radius: 5px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.15s;
            max-width: 340px;
            line-height: 1.5;
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
            background: rgba(0,0,0,0.55);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }}

        button:hover {{
            background: rgba(0,0,0,0.75);
        }}

        .info {{
            padding: 5px 10px;
            background: rgba(0,0,0,0.45);
            border-radius: 4px;
            font-size: 11px;
            color: #ddd;
        }}

        h1 {{
            position: fixed;
            top: 12px;
            left: 16px;
            margin: 0;
            font-size: 15px;
            color: #333;
            pointer-events: none;
        }}

        #legend {{
            position: fixed;
            top: 12px;
            right: 16px;
            background: rgba(255,255,255,0.92);
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 11px;
            color: #333;
            line-height: 1.8;
        }}

        .legend-row {{
            display: flex;
            align-items: center;
            gap: 7px;
        }}

        .legend-swatch {{
            width: 28px;
            height: 3px;
            flex-shrink: 0;
        }}

        .legend-swatch.dashed {{
            background: repeating-linear-gradient(
                to right,
                #bbb 0px, #bbb 5px,
                transparent 5px, transparent 9px
            );
            height: 2px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <svg id="graph"></svg>

    <div id="legend">
        <div style="font-weight:600;margin-bottom:4px;">Edge types</div>
        <div class="legend-row"><div class="legend-swatch" style="background:#4e79a7"></div>definition</div>
        <div class="legend-row"><div class="legend-swatch" style="background:#59a14f"></div>kind-of</div>
        <div class="legend-row"><div class="legend-swatch" style="background:#f28e2b"></div>production</div>
        <div class="legend-row"><div class="legend-swatch" style="background:#e15759"></div>dependence</div>
        <div class="legend-row"><div class="legend-swatch" style="background:#edc948"></div>property</div>
        <div class="legend-row"><div class="legend-swatch" style="background:#76b7b2"></div>relation</div>
        <div class="legend-row"><div class="legend-swatch" style="background:#b07aa1"></div>component</div>
        <div class="legend-row"><div class="legend-swatch dashed"></div>co-occurrence</div>
    </div>

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

        // Edge type color palette
        const EDGE_COLORS = {{
            "definition":   "#4e79a7",
            "kind-of":      "#59a14f",
            "production":   "#f28e2b",
            "dependence":   "#e15759",
            "property":     "#edc948",
            "relation":     "#76b7b2",
            "component":    "#b07aa1",
            "cooccurrence": "#bbbbbb",
        }};
        const EDGE_COLOR_DEFAULT = "#aaaaaa";

        // Community color scale
        const communityColor = d3.scaleOrdinal(d3.schemeTableau10);

        // Node radius: 6 baseline + bonus from rarity score (0–5 range)
        const nodeRadius = d => Math.max(6, 6 + (d.score || 0) * 2.5);

        // Edge color helper
        const edgeColor = d => EDGE_COLORS[d.type] || EDGE_COLOR_DEFAULT;

        // Directed types (component and cooccurrence are undirected)
        const DIRECTED_TYPES = new Set(["definition", "kind-of", "production", "dependence", "property", "relation"]);

        // Create SVG
        const svg = d3.select("#graph");
        const defs = svg.append("defs");

        // Per-type arrowhead markers
        Object.entries(EDGE_COLORS).forEach(([type, color]) => {{
            if (!DIRECTED_TYPES.has(type)) return;
            defs.append("marker")
                .attr("id", `arrow-${{type}}`)
                .attr("viewBox", "0 -5 10 10")
                .attr("refX", 10)
                .attr("refY", 0)
                .attr("markerWidth", 6)
                .attr("markerHeight", 6)
                .attr("orient", "auto")
                .append("path")
                .attr("d", "M0,-5L10,0L0,5")
                .attr("fill", color)
                .attr("fill-opacity", 0.85);
        }});

        const g = svg.append("g");

        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.05, 12])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});

        svg.call(zoom);

        // Tooltip
        const tooltip = d3.select("#tooltip");

        // Inlined graph data
        const data = {graph_data_json};

        // Initialize visualization
        (function() {{
            d3.select("#info").html(
                `Nodes: ${{data.nodes.length}} | Links: ${{data.links.length}}`
            );

            // Force simulation — stronger repulsion, distance by weight
            const simulation = d3.forceSimulation(data.nodes)
                .force("link", d3.forceLink(data.links)
                    .id(d => d.id)
                    .distance(d => 180 / Math.sqrt(d.weight || 1))
                    .strength(0.6))
                .force("charge", d3.forceManyBody()
                    .strength(d => {{
                        // Reduce attraction for very high-degree hubs
                        const deg = data.links.filter(
                            l => l.source === d || l.source.id === d.id ||
                                 l.target === d || l.target.id === d.id
                        ).length;
                        return deg > 10 ? -600 : -400;
                    }}))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide()
                    .radius(d => nodeRadius(d) + (d.label || "").length * 3.8 + 4));

            // Links
            const link = g.append("g")
                .attr("class", "links")
                .selectAll("line")
                .data(data.links)
                .join("line")
                .attr("stroke", edgeColor)
                .attr("stroke-opacity", d => d.type === "cooccurrence" ? 0.35 : 0.65)
                .attr("stroke-width", d => d.type === "cooccurrence" ? 1 : Math.min(4, 1 + Math.sqrt(d.weight || 1)))
                .attr("stroke-dasharray", d => d.type === "cooccurrence" ? "6,4" : null)
                .attr("marker-end", d => DIRECTED_TYPES.has(d.type) ? `url(#arrow-${{d.type}})` : null)
                .on("mouseover", (event, d) => {{
                    const src = d.source.id || d.source;
                    const tgt = d.target.id || d.target;
                    const arrow = DIRECTED_TYPES.has(d.type) ? "→" : "↔";
                    let html = `<strong>${{src}} ${{arrow}} ${{tgt}}</strong><br>`;
                    html += `<span style="color:#bbb">${{d.verb || d.type || "relates to"}}</span>`;
                    if (d.weight && d.weight > 1) html += ` (×${{d.weight}})`;
                    if (d.evidence && d.evidence.length) {{
                        html += `<br><br><em style="color:#ccc">${{d.evidence[0]}}</em>`;
                    }}
                    tooltip.html(html)
                        .style("left", (event.clientX + 12) + "px")
                        .style("top", (event.clientY - 12) + "px")
                        .style("opacity", 1);
                }})
                .on("mouseout", () => tooltip.style("opacity", 0));

            // Nodes
            const node = g.append("g")
                .attr("class", "nodes")
                .selectAll("circle")
                .data(data.nodes)
                .join("circle")
                .attr("class", "node")
                .attr("r", nodeRadius)
                .attr("fill", d => communityColor(d.group || 0))
                .call(drag(simulation))
                .on("mouseover", (event, d) => {{
                    let html = `<strong style="font-size:13px">${{d.label}}</strong>`;
                    if (d.score) html += ` <span style="color:#bbb">(score: ${{d.score.toFixed(2)}})</span>`;
                    if (d.frequency) html += `<br>Frequency: ${{d.frequency}}`;
                    if (d.pos) html += `<br>POS: ${{d.pos}}`;
                    if (d.definition) html += `<br><br>${{d.definition}}`;
                    tooltip.html(html)
                        .style("left", (event.clientX + 12) + "px")
                        .style("top", (event.clientY - 12) + "px")
                        .style("opacity", 1);
                }})
                .on("mouseout", () => tooltip.style("opacity", 0))
                .on("dblclick", (event, d) => {{
                    d.fx = null;
                    d.fy = null;
                    simulation.alphaTarget(0.1).restart();
                }});

            // Edge labels — always visible unless density exceeds 3 edges per 100×100 px,
            // in which case fall back to hover-only (verb shown in tooltip).
            const LABEL_DENSITY_THRESHOLD = 3; // edges per 10 000 px²
            const edgeDensity = data.links.length / (width * height) * 10000;
            const showLinkLabels = edgeDensity <= LABEL_DENSITY_THRESHOLD;

            const linkLabel = showLinkLabels
                ? g.append("g")
                    .attr("class", "link-labels")
                    .selectAll("text")
                    .data(data.links.filter(d => d.type !== "cooccurrence"))
                    .join("text")
                    .attr("class", "link-label")
                    .attr("fill", d => edgeColor(d))
                    .text(d => d.verb || d.label || "")
                : null;

            // Node labels — size proportional to score/radius
            const label = g.append("g")
                .attr("class", "node-labels")
                .selectAll("text")
                .data(data.nodes)
                .join("text")
                .attr("class", "node-label")
                .text(d => d.label)
                .attr("font-size", d => Math.max(10, nodeRadius(d) * 1.1))
                .attr("dy", d => nodeRadius(d) + 13);

            // Simulation tick
            simulation.on("tick", () => {{
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => {{
                        if (!DIRECTED_TYPES.has(d.type)) return d.target.x;
                        const dx = d.target.x - d.source.x;
                        const dy = d.target.y - d.source.y;
                        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                        const r = nodeRadius(d.target) + 9;
                        return d.target.x - (dx / dist) * r;
                    }})
                    .attr("y2", d => {{
                        if (!DIRECTED_TYPES.has(d.type)) return d.target.y;
                        const dx = d.target.x - d.source.x;
                        const dy = d.target.y - d.source.y;
                        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                        const r = nodeRadius(d.target) + 9;
                        return d.target.y - (dy / dist) * r;
                    }});

                node.attr("cx", d => d.x).attr("cy", d => d.y);

                if (linkLabel) linkLabel
                    .attr("x", d => (d.source.x + d.target.x) / 2)
                    .attr("y", d => (d.source.y + d.target.y) / 2 - 4);

                label.attr("x", d => d.x).attr("y", d => d.y);
            }});

            window.simulation = simulation;
        }})();

        // Drag behavior
        function drag(simulation) {{
            return d3.drag()
                .on("start", (event) => {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                }})
                .on("drag", (event) => {{
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                }})
                .on("end", (event) => {{
                    if (!event.active) simulation.alphaTarget(0);
                    // fx/fy intentionally kept — node stays pinned where dropped.
                    // Double-click the node to release it back to the simulation.
                }});
        }}

        function resetZoom() {{
            svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
        }}

        function restartSimulation() {{
            window.simulation.alpha(1).restart();
        }}
    </script>
</body>
</html>"""
