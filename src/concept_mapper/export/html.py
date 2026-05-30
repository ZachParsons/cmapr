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
    include_evidence: bool = True,
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
    output_path = Path(output_dir)
    if output_path.suffix.lower() == ".html":
        output_dir = output_path.parent
        html_filename = output_path.name
    else:
        output_dir = output_path
        html_filename = "index.html"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export graph data to JSON file (for reference)
    data_path = output_dir / "graph_data.json"
    export_d3_json(graph, data_path, include_evidence=include_evidence)

    # Get graph data as dict for inlining
    graph_data = to_d3_dict(graph, include_evidence=include_evidence)
    graph_data_json = json.dumps(graph_data, ensure_ascii=False)

    # Generate HTML file with inlined data
    html_path = output_dir / html_filename

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
            pointer-events: bounding-box;
            cursor: help;
            text-anchor: middle;
            fill: #555;
            paint-order: stroke;
            stroke: rgba(255,255,255,0.85);
            stroke-width: 3px;
            stroke-linejoin: round;
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

        .legend-row input[type=checkbox] {{
            width: 13px;
            height: 13px;
            cursor: pointer;
            flex-shrink: 0;
            accent-color: #555;
        }}

        #detail-panel {{
            position: fixed;
            top: 0;
            right: 0;
            width: 280px;
            height: 100%;
            background: rgba(255,255,255,0.97);
            border-left: 1px solid #ddd;
            padding: 16px 16px 32px;
            overflow-y: auto;
            transform: translateX(100%);
            transition: transform 0.2s ease;
            z-index: 200;
            box-sizing: border-box;
            font-size: 12px;
        }}

        #detail-panel.open {{
            transform: translateX(0);
        }}

        #detail-panel h2 {{
            margin: 0 24px 10px 0;
            font-size: 15px;
            color: #222;
            word-break: break-all;
        }}

        #detail-panel .close-btn {{
            position: absolute;
            top: 10px;
            right: 12px;
            background: none;
            border: none;
            font-size: 16px;
            cursor: pointer;
            color: #888;
            padding: 2px 6px;
        }}

        #detail-panel .close-btn:hover {{
            color: #333;
            background: #eee;
            border-radius: 3px;
        }}

        .panel-meta {{
            color: #555;
            margin: 2px 0;
        }}

        .panel-edge {{
            display: flex;
            align-items: baseline;
            gap: 5px;
            margin: 5px 0;
            line-height: 1.4;
        }}

        .panel-edge-type {{
            font-size: 10px;
            color: #888;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <svg id="graph"></svg>

    <div id="legend">
        <div style="font-weight:600;margin-bottom:6px;">Edge types</div>
        <!-- rows injected by JS so checkboxes share closure with D3 selections -->
    </div>

    <div id="detail-panel">
        <button class="close-btn" onclick="closeDetailPanel()">✕</button>
        <div id="detail-panel-content"></div>
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
            "opposition":   "#d4a0a0",
            "property":     "#edc948",
            "relation":     "#76b7b2",
            "component":    "#b07aa1",
            "recurrence":   "#7a8aa0",
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
        const DIRECTED_TYPES = new Set(["definition", "kind-of", "production", "dependence", "property", "relation", "recurrence"]);

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

        // Close detail panel when clicking on empty canvas
        svg.on("click", () => closeDetailPanel());

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

            // Cluster force — when nodes have a `chapter` attribute (from
            // `cmapr cluster`), pull each node toward its chapter's centroid.
            // Centroids sit on a circle around the canvas centre.
            const clusterKeys = Array.from(new Set(
                data.nodes.map(d => d.chapter).filter(c => c != null)
            ));
            if (clusterKeys.length >= 2) {{
                const cx = width / 2, cy = height / 2;
                const radius = Math.min(width, height) * 0.35;
                const centroids = {{}};
                clusterKeys.forEach((key, i) => {{
                    const angle = (2 * Math.PI * i) / clusterKeys.length;
                    centroids[key] = {{
                        x: cx + radius * Math.cos(angle),
                        y: cy + radius * Math.sin(angle),
                    }};
                }});
                simulation.force("cluster", alpha => {{
                    data.nodes.forEach(d => {{
                        const centroid = centroids[d.chapter];
                        if (!centroid) return;
                        d.vx -= (d.x - centroid.x) * alpha * 0.05;
                        d.vy -= (d.y - centroid.y) * alpha * 0.05;
                    }});
                }});
            }}

            // Shared tooltip handler for both edge lines and edge labels
            const showLinkTooltip = (event, d) => {{
                const src = d.source.id || d.source;
                const tgt = d.target.id || d.target;
                const arrow = DIRECTED_TYPES.has(d.type) ? "→" : "↔";
                let html = `<strong>${{src}} ${{arrow}} ${{tgt}}</strong><br>`;
                html += `<span style="color:#bbb">${{d.verb || d.type || "relates to"}}</span>`;
                if (d.weight && d.weight > 1) html += ` (×${{d.weight}})`;
                if (d.relation_types && d.relation_types.length > 1) {{
                    html += `<br><span style="color:#999;font-size:11px">also: `
                        + d.relation_types.slice(1).map(t => {{
                            const w = d.weight_by_type ? d.weight_by_type[t] : null;
                            return w != null ? `${{t}} (×${{w}})` : t;
                        }}).join(", ")
                        + `</span>`;
                }}
                if (d.evidence_by_type && d.relation_types) {{
                    d.relation_types.forEach(t => {{
                        const ev = d.evidence_by_type[t] || [];
                        if (ev.length) {{
                            html += `<br><hr style="border-color:#444;margin:4px 0">`
                                + `<div style="color:#888;font-size:11px;margin-bottom:2px">${{t}}</div>`
                                + ev.map(s => `<em style="color:#ccc">${{s}}</em>`).join("<br>");
                        }}
                    }});
                }} else if (d.evidence && d.evidence.length) {{
                    html += d.evidence
                        .map(s => `<br><hr style="border-color:#444;margin:4px 0"><em style="color:#ccc">${{s}}</em>`)
                        .join("");
                }}
                tooltip.html(html)
                    .style("left", (event.clientX + 12) + "px")
                    .style("top", (event.clientY - 12) + "px")
                    .style("opacity", 1);
            }};
            const hideLinkTooltip = () => tooltip.style("opacity", 0);

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
                .on("mouseover", showLinkTooltip)
                .on("mouseout", hideLinkTooltip);

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
                    if (d.score) {{
                        html += `<br>Significance: <strong>${{d.score.toFixed(2)}}</strong>`
                            + ` <span style="color:#888;font-size:11px">(rarity-based — higher = more distinctive in this corpus)</span>`;
                    }}
                    if (d.frequency) html += `<br>Frequency: ${{d.frequency}}`;
                    if (d.pos) html += `<br>Part of speech: ${{d.pos}}`;
                    if (d.definition) {{
                        html += `<br><br>${{d.definition}}`;
                    }} else {{
                        // Derive an in-corpus characterisation by searching outgoing
                        // edges sourced at this node, preferring the most definitional
                        // relation types first.
                        const DEF_PRIORITY = ["definition", "kind-of", "property", "relation"];
                        const TYPE_LABELS = {{
                            "definition": "Definition",
                            "kind-of":    "Kind of",
                            "property":   "Described as",
                            "relation":   "From the text",
                        }};
                        let chosen = null;
                        for (const t of DEF_PRIORITY) {{
                            chosen = data.links.find(l => {{
                                const src = l.source.id !== undefined ? l.source.id : l.source;
                                return src === d.id && l.type === t && l.evidence && l.evidence.length;
                            }});
                            if (chosen) break;
                        }}
                        if (chosen) {{
                            html += `<br><br><div style="color:#888;font-size:11px;margin-bottom:2px">${{TYPE_LABELS[chosen.type]}}</div>`
                                + `<em style="color:#ccc">${{chosen.evidence[0]}}</em>`;
                        }}
                    }}
                    tooltip.html(html)
                        .style("left", (event.clientX + 12) + "px")
                        .style("top", (event.clientY - 12) + "px")
                        .style("opacity", 1);
                }})
                .on("mouseout", () => tooltip.style("opacity", 0))
                .on("click", (event, d) => {{
                    event.stopPropagation();
                    showDetailPanel(d);
                }})
                .on("dblclick", (event, d) => {{
                    event.stopPropagation();
                    if (focusedNodeId === d.id) {{
                        focusedNodeId = null;
                    }} else {{
                        focusedNodeId = d.id;
                    }}
                    d.fx = null;
                    d.fy = null;
                    simulation.alphaTarget(0.1).restart();
                    applyVisibility();
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
                    .text(d => {{
                        if (d.relation_types && d.relation_types.length > 1) {{
                            return d.relation_types
                                .map(t => (d.verb_by_type && d.verb_by_type[t]) || t)
                                .join(" / ");
                        }}
                        return d.verb || d.label || "";
                    }})
                    .on("mouseover", showLinkTooltip)
                    .on("mouseout", hideLinkTooltip)
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

            // ----------------------------------------------------------------
            // 11.1 — Edge type toggle
            // ----------------------------------------------------------------
            const hiddenTypes = new Set();
            let focusedNodeId = null;

            function applyVisibility() {{
                function linkVisible(l) {{
                    if (hiddenTypes.has(l.type)) return false;
                    if (focusedNodeId === null) return true;
                    const src = l.source.id !== undefined ? l.source.id : l.source;
                    const tgt = l.target.id !== undefined ? l.target.id : l.target;
                    return src === focusedNodeId || tgt === focusedNodeId;
                }}
                link.style("display", d => linkVisible(d) ? null : "none");
                if (linkLabel) linkLabel.style("display", d => linkVisible(d) ? null : "none");

                const visibleIds = new Set();
                data.links.forEach(l => {{
                    if (linkVisible(l)) {{
                        visibleIds.add(l.source.id !== undefined ? l.source.id : l.source);
                        visibleIds.add(l.target.id !== undefined ? l.target.id : l.target);
                    }}
                }});
                if (focusedNodeId !== null) visibleIds.add(focusedNodeId);
                node.style("display",  d => visibleIds.has(d.id) ? null : "none");
                label.style("display", d => visibleIds.has(d.id) ? null : "none");
            }}

            // Build legend rows with checkboxes (inside IIFE for closure)
            const LEGEND_TYPES = [
                {{ type: "definition",   color: "#4e79a7", dashed: false }},
                {{ type: "kind-of",      color: "#59a14f", dashed: false }},
                {{ type: "production",   color: "#f28e2b", dashed: false }},
                {{ type: "dependence",   color: "#e15759", dashed: false }},
                {{ type: "opposition",   color: "#d4a0a0", dashed: false }},
                {{ type: "property",     color: "#edc948", dashed: false }},
                {{ type: "relation",     color: "#76b7b2", dashed: false }},
                {{ type: "component",    color: "#b07aa1", dashed: false }},
                {{ type: "recurrence",   color: "#7a8aa0", dashed: true  }},
                {{ type: "cooccurrence", color: "#bbbbbb", dashed: true  }},
            ];
            const legendEl = document.getElementById("legend");
            LEGEND_TYPES.forEach(({{ type, color, dashed }}) => {{
                // Only show types that appear in the data
                const hasType = data.links.some(l => l.type === type);
                if (!hasType) return;

                const row = document.createElement("div");
                row.className = "legend-row";

                const cb = document.createElement("input");
                cb.type = "checkbox";
                cb.checked = true;
                cb.title = `Show/hide ${{type}} edges`;
                cb.addEventListener("change", () => {{
                    if (cb.checked) hiddenTypes.delete(type);
                    else hiddenTypes.add(type);
                    applyVisibility();
                }});

                const swatch = document.createElement("div");
                swatch.className = "legend-swatch" + (dashed ? " dashed" : "");
                if (!dashed) swatch.style.background = color;

                const lbl = document.createElement("span");
                lbl.textContent = type;

                row.appendChild(cb);
                row.appendChild(swatch);
                row.appendChild(lbl);
                legendEl.appendChild(row);
            }});

            // ----------------------------------------------------------------
            // 11.2 — Node detail panel
            // ----------------------------------------------------------------
            function showDetailPanel(d) {{
                const nodeId = d.id;
                const edges = data.links.filter(l => {{
                    const src = l.source.id !== undefined ? l.source.id : l.source;
                    const tgt = l.target.id !== undefined ? l.target.id : l.target;
                    return src === nodeId || tgt === nodeId;
                }});

                let html = `<h2>${{d.label}}</h2>`;
                if (d.score    != null) html += `<div class="panel-meta">Rarity score: ${{d.score.toFixed(2)}}</div>`;
                if (d.frequency != null) html += `<div class="panel-meta">Frequency: ${{d.frequency}}</div>`;
                if (d.pos)              html += `<div class="panel-meta">POS: ${{d.pos}}</div>`;
                if (d.definition)       html += `<div class="panel-meta" style="margin-top:6px">${{d.definition}}</div>`;

                html += `<hr style="margin:10px 0"><div style="font-weight:600;margin-bottom:6px">Connections (${{edges.length}})</div>`;

                edges.forEach(l => {{
                    const src = l.source.id !== undefined ? l.source.id : l.source;
                    const tgt = l.target.id !== undefined ? l.target.id : l.target;
                    const other = src === nodeId ? tgt : src;
                    const isSource = src === nodeId;
                    const directed = DIRECTED_TYPES.has(l.type);
                    const arrow = directed ? (isSource ? "→" : "←") : "↔";
                    const color = EDGE_COLORS[l.type] || EDGE_COLOR_DEFAULT;
                    let typeText = l.verb || l.type;
                    if (l.relation_types && l.relation_types.length > 1) {{
                        typeText = l.relation_types.join(" / ");
                    }}
                    html += `<div class="panel-edge">
                        <span style="color:${{color}};font-size:14px">■</span>
                        <span>${{arrow}} <strong>${{other}}</strong></span>
                        <span class="panel-edge-type">${{typeText}}</span>
                    </div>`;
                }});

                document.getElementById("detail-panel-content").innerHTML = html;
                document.getElementById("detail-panel").classList.add("open");
            }}

            window.showDetailPanel = showDetailPanel;
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
                    // Double-click the node to expand its neighbourhood (also releases the pin).
                }});
        }}

        function resetZoom() {{
            svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
        }}

        function restartSimulation() {{
            window.simulation.alpha(1).restart();
        }}

        function closeDetailPanel() {{
            document.getElementById("detail-panel").classList.remove("open");
        }}
    </script>
</body>
</html>"""
