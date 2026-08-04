from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def build_kg_graph_payload(
    stock_graph_edges_path: Path | None = None,
    report_llm_relations_path: Path | None = None,
    kg_stock_edges_llm_path: Path | None = None,
) -> dict[str, Any]:
    """Build a complete static KG browser payload from alphagraph KG artifacts."""

    stock_edges_path = Path(stock_graph_edges_path) if stock_graph_edges_path else None
    kg_edges_path = Path(kg_stock_edges_llm_path) if kg_stock_edges_llm_path else None
    relations_path = Path(report_llm_relations_path) if report_llm_relations_path else None
    edge_path = stock_edges_path if stock_edges_path and stock_edges_path.exists() else kg_edges_path
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stock_graph_edges_path": str(stock_edges_path) if stock_edges_path else "",
        "kg_stock_edges_llm_path": str(kg_edges_path) if kg_edges_path else "",
        "edge_source_path": str(edge_path) if edge_path else "",
        "report_llm_relations_path": str(relations_path) if relations_path else "",
        "months": [],
        "edge_types": [],
        "nodes": [],
        "links": [],
        "edge_rows": [],
        "relation_rows": [],
        "reports": [],
        "summary": {
            "node_count": 0,
            "edge_count": 0,
            "relation_count": 0,
            "report_count": 0,
            "edge_type_counts": {},
        },
    }

    relation_rows = _load_relation_rows(relations_path)
    payload["relation_rows"] = relation_rows
    report_ids = sorted(
        {
            str(row.get("report_id", ""))
            for row in relation_rows
            if str(row.get("report_id", "")).strip()
        }
    )
    payload["reports"] = [{"report_id": report_id} for report_id in report_ids]
    payload["summary"]["relation_count"] = len(relation_rows)
    payload["summary"]["report_count"] = len(report_ids)

    if edge_path is None or not edge_path.exists():
        return payload

    try:
        edges_df = _read_frame(edge_path)
    except Exception:
        return payload

    required_columns = {"src_code", "dst_code", "edge_type"}
    if edges_df.empty or not required_columns.issubset(edges_df.columns):
        return payload

    normalized = edges_df.copy()
    for column in ("month", "src_code", "dst_code", "edge_type", "source_id", "report_id"):
        if column not in normalized.columns:
            normalized[column] = ""
        normalized[column] = normalized[column].fillna("").astype(str)
    if "edge_weight" not in normalized.columns:
        normalized["edge_weight"] = 1.0
    normalized["edge_weight"] = pd.to_numeric(normalized["edge_weight"], errors="coerce").fillna(0.0)
    normalized = normalized.loc[
        normalized["src_code"].str.len().gt(0)
        & normalized["dst_code"].str.len().gt(0)
        & normalized["edge_type"].str.len().gt(0)
        & normalized["src_code"].ne(normalized["dst_code"])
    ].copy()
    if normalized.empty:
        return payload

    normalized = normalized.sort_values(
        ["month", "edge_type", "src_code", "dst_code", "source_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    edge_rows: list[dict[str, Any]] = []
    degree: dict[str, int] = {}
    for index, row in enumerate(normalized.itertuples(index=False)):
        src_code = str(row.src_code)
        dst_code = str(row.dst_code)
        degree[src_code] = degree.get(src_code, 0) + 1
        degree[dst_code] = degree.get(dst_code, 0) + 1
        edge = {
            "id": f"kg-edge-{index}",
            "month": str(row.month),
            "source": src_code,
            "target": dst_code,
            "src_code": src_code,
            "dst_code": dst_code,
            "edge_type": str(row.edge_type),
            "edge_weight": float(row.edge_weight or 0.0),
            "source_id": str(row.source_id),
            "report_id": str(row.report_id),
        }
        edge_rows.append(edge)

    edge_type_counts = {
        str(edge_type): int(count)
        for edge_type, count in normalized["edge_type"].value_counts().sort_index().items()
    }
    payload["months"] = sorted(normalized["month"].dropna().astype(str).unique().tolist())
    payload["edge_types"] = sorted(normalized["edge_type"].dropna().astype(str).unique().tolist())
    payload["nodes"] = [
        {"id": code, "label": code, "degree": int(count), "edge_count": int(count)}
        for code, count in sorted(degree.items())
    ]
    payload["links"] = edge_rows
    payload["edge_rows"] = edge_rows
    payload["summary"] = {
        "node_count": len(payload["nodes"]),
        "edge_count": len(edge_rows),
        "relation_count": len(relation_rows),
        "report_count": len(report_ids),
        "edge_type_counts": edge_type_counts,
    }
    return payload


def write_kg_html_report(
    output_path: Path,
    stock_graph_edges_path: Path | None = None,
    kg_stock_edges_llm_path: Path | None = None,
    report_llm_relations_path: Path | None = None,
) -> dict[str, Any]:
    payload = build_kg_graph_payload(
        stock_graph_edges_path=stock_graph_edges_path,
        kg_stock_edges_llm_path=kg_stock_edges_llm_path,
        report_llm_relations_path=report_llm_relations_path,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_kg_html(payload), encoding="utf-8")
    return payload


def render_kg_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False)
    return _HTML_TEMPLATE.replace("__KG_GRAPH_DATA__", payload_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate alphagraph KG HTML browser")
    parser.add_argument("--stock-graph-edges-path", type=Path, default=None)
    parser.add_argument("--kg-stock-edges-llm-path", type=Path, default=None)
    parser.add_argument("--report-llm-relations-path", type=Path, default=None)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()

    write_kg_html_report(
        output_path=args.output_html,
        stock_graph_edges_path=args.stock_graph_edges_path,
        kg_stock_edges_llm_path=args.kg_stock_edges_llm_path,
        report_llm_relations_path=args.report_llm_relations_path,
    )
    print(f"KG HTML report written to: {args.output_html}")


def _load_relation_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        relations_df = _read_frame(path)
    except Exception:
        return []
    if relations_df.empty:
        return []
    normalized = relations_df.copy()
    rows: list[dict[str, Any]] = []
    for index, row in normalized.reset_index(drop=True).iterrows():
        item = {str(key): _json_safe(value) for key, value in row.to_dict().items()}
        item.setdefault("id", f"kg-relation-{index}")
        rows.append(item)
    return rows


def _read_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


_HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>alphagraph KG 浏览器</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/antd@5/dist/reset.css" />
  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
  <style>
    :root {
      --ag-blue: #1677ff;
      --ag-text: #172033;
      --ag-muted: #667085;
      --ag-border: #d9dfe9;
      --ag-bg: #f5f7fb;
      --ag-card: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ag-text);
      background: var(--ag-bg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    .shell { max-width: 1480px; margin: 0 auto; padding: 24px; }
    .hero, .card {
      background: var(--ag-card);
      border: 1px solid var(--ag-border);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    .hero { padding: 20px 22px; margin-bottom: 16px; }
    .hero h1 { margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }
    .muted { color: var(--ag-muted); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    .grid { display: grid; gap: 16px; }
    .stats { grid-template-columns: repeat(5, minmax(0, 1fr)); margin-bottom: 16px; }
    .stat { padding: 16px; }
    .stat-label { color: var(--ag-muted); font-size: 13px; }
    .stat-value { margin-top: 6px; font-size: 24px; font-weight: 650; }
    .card { padding: 16px; margin-bottom: 16px; }
    .toolbar {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }
    .controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    select, input, button {
      height: 32px;
      border-radius: 6px;
      border: 1px solid var(--ag-border);
      background: #fff;
      color: var(--ag-text);
      padding: 0 10px;
      font-size: 14px;
    }
    select { min-width: 156px; }
    input { width: 260px; }
    button { cursor: pointer; }
    button.primary { border-color: var(--ag-blue); background: var(--ag-blue); color: #fff; }
    .browser-grid { display: grid; grid-template-columns: minmax(0, 2fr) minmax(280px, 0.8fr); gap: 16px; }
    #kg-graph {
      min-height: 520px;
      border: 1px solid var(--ag-border);
      border-radius: 8px;
      background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
      overflow: hidden;
    }
    .kg-empty {
      height: 100%;
      min-height: 360px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--ag-muted);
      text-align: center;
      padding: 24px;
    }
    .type-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 7px 0;
      border-bottom: 1px solid #edf0f5;
      font-size: 13px;
    }
    .tag {
      display: inline-block;
      max-width: 220px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      border: 1px solid var(--ag-border);
      border-radius: 6px;
      padding: 2px 7px;
      background: #fff;
    }
    .table-wrap { overflow: auto; border: 1px solid var(--ag-border); border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; min-width: 980px; background: #fff; }
    th, td { padding: 9px 10px; border-bottom: 1px solid #edf0f5; text-align: left; font-size: 13px; }
    th { position: sticky; top: 0; background: #f8fafc; z-index: 1; font-weight: 650; }
    @media (max-width: 960px) {
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .browser-grid { grid-template-columns: 1fr; }
      input { width: min(100%, 260px); }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>alphagraph KG 浏览器</h1>
      <div class="muted">独立静态 KG HTML；图视图最多绘制当前筛选前 500 条边，表格和 payload 保留完整 KG。</div>
      <div class="muted mono" id="kg-paths" style="margin-top: 8px;"></div>
    </section>

    <section class="grid stats">
      <div class="card stat"><div class="stat-label">KG节点</div><div class="stat-value" id="stat-nodes">0</div></div>
      <div class="card stat"><div class="stat-label">KG边</div><div class="stat-value" id="stat-edges">0</div></div>
      <div class="card stat"><div class="stat-label">LLM关系</div><div class="stat-value" id="stat-relations">0</div></div>
      <div class="card stat"><div class="stat-label">筛选节点</div><div class="stat-value" id="stat-filter-nodes">0</div></div>
      <div class="card stat"><div class="stat-label">筛选边</div><div class="stat-value" id="stat-filter-edges">0</div></div>
    </section>

    <section class="card">
      <div class="toolbar">
        <div class="controls">
          <select id="kg-month-select" aria-label="月份筛选"></select>
          <select id="kg-edge-type-select" aria-label="边类型筛选"></select>
          <input id="kg-search-input" placeholder="搜索股票/报告/source_id" />
        </div>
        <div class="controls">
          <button id="kg-zoom-in">放大</button>
          <button id="kg-zoom-out">缩小</button>
          <button id="kg-reset">重置</button>
          <button id="kg-fit" class="primary">适配视图</button>
        </div>
      </div>
      <div class="browser-grid">
        <div>
          <div id="kg-graph-summary" class="muted"></div>
          <div id="kg-graph" style="margin-top: 8px;"></div>
        </div>
        <aside>
          <h3 style="margin-top:0;">边类型分布</h3>
          <div id="kg-type-list"></div>
        </aside>
      </div>
    </section>

    <section class="card">
      <div class="toolbar">
        <h2 style="margin:0;font-size:18px;">完整边表</h2>
        <div class="muted" id="kg-edge-table-summary"></div>
      </div>
      <div class="table-wrap">
        <table id="kg-edge-table">
          <thead>
            <tr>
              <th>月份</th><th>源股票</th><th>目标股票</th><th>边类型</th><th>权重</th><th>report_id</th><th>source_id</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    const kgGraphData = __KG_GRAPH_DATA__;
    let kgGraphState = null;

    const edgePalette = [
      '#1677ff', '#13c2c2', '#52c41a', '#faad14', '#f5222d',
      '#722ed1', '#eb2f96', '#2f54eb', '#fa8c16', '#08979c'
    ];
    const edgeTypeColor = (edgeType) => {
      const value = String(edgeType || 'unknown');
      let hash = 0;
      for (let i = 0; i < value.length; i += 1) hash = ((hash << 5) - hash) + value.charCodeAt(i);
      return edgePalette[Math.abs(hash) % edgePalette.length];
    };
    const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));

    function summarizeKgEdges(edges) {
      const nodes = new Set();
      const typeCounts = {};
      (edges || []).forEach((edge) => {
        if (edge.src_code) nodes.add(edge.src_code);
        if (edge.dst_code) nodes.add(edge.dst_code);
        const edgeType = String(edge.edge_type || 'unknown');
        typeCounts[edgeType] = (typeCounts[edgeType] || 0) + 1;
      });
      return { nodeCount: nodes.size, edgeCount: (edges || []).length, typeCounts };
    }

    function filteredEdges() {
      const month = document.getElementById('kg-month-select').value;
      const edgeType = document.getElementById('kg-edge-type-select').value;
      const query = document.getElementById('kg-search-input').value.trim().toLowerCase();
      return (kgGraphData.edge_rows || []).filter((row) => {
        if (month !== 'all' && String(row.month || '') !== month) return false;
        if (edgeType !== 'all' && String(row.edge_type || '') !== edgeType) return false;
        if (!query) return true;
        return [row.src_code, row.dst_code, row.source_id, row.report_id, row.edge_type]
          .some((value) => String(value || '').toLowerCase().includes(query));
      });
    }

    function fillSelect(select, values, allLabel) {
      select.innerHTML = `<option value="all">${allLabel}</option>` + (values || [])
        .map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`)
        .join('');
    }

    function renderTypeList(typeCounts) {
      const container = document.getElementById('kg-type-list');
      const entries = Object.entries(typeCounts || {}).sort((a, b) => b[1] - a[1]);
      if (!entries.length) {
        container.innerHTML = '<div class="muted">暂无 KG 边</div>';
        return;
      }
      container.innerHTML = entries.map(([edgeType, count]) => `
        <div class="type-row">
          <span class="tag" style="border-color:${edgeTypeColor(edgeType)};color:${edgeTypeColor(edgeType)}">${escapeHtml(edgeType)}</span>
          <strong>${count}</strong>
        </div>
      `).join('');
    }

    function renderEdgeTable(edges) {
      const body = document.querySelector('#kg-edge-table tbody');
      const summary = document.getElementById('kg-edge-table-summary');
      summary.textContent = `当前表格 ${edges.length} 条边`;
      if (!edges.length) {
        body.innerHTML = '<tr><td colspan="7" class="muted">暂无 KG 边</td></tr>';
        return;
      }
      body.innerHTML = edges.map((edge) => `
        <tr>
          <td>${escapeHtml(edge.month)}</td>
          <td class="mono">${escapeHtml(edge.src_code)}</td>
          <td class="mono">${escapeHtml(edge.dst_code)}</td>
          <td><span class="tag" style="border-color:${edgeTypeColor(edge.edge_type)};color:${edgeTypeColor(edge.edge_type)}">${escapeHtml(edge.edge_type)}</span></td>
          <td>${(Number(edge.edge_weight) || 0).toFixed(4)}</td>
          <td class="mono">${escapeHtml(edge.report_id)}</td>
          <td class="mono">${escapeHtml(edge.source_id)}</td>
        </tr>
      `).join('');
    }

    function fitKgGraph(duration = 250) {
      if (!kgGraphState || !kgGraphState.nodes.length) return;
      const { svg, zoom, nodes, width, height } = kgGraphState;
      const xs = nodes.map((d) => Number(d.x)).filter(Number.isFinite);
      const ys = nodes.map((d) => Number(d.y)).filter(Number.isFinite);
      if (!xs.length || !ys.length) return;
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const graphWidth = Math.max(maxX - minX, 1);
      const graphHeight = Math.max(maxY - minY, 1);
      const scale = Math.max(0.18, Math.min(3, 0.86 / Math.max(graphWidth / width, graphHeight / height)));
      const tx = width / 2 - scale * (minX + maxX) / 2;
      const ty = height / 2 - scale * (minY + maxY) / 2;
      svg.transition().duration(duration).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
    }

    function renderGraph(edges) {
      const container = document.getElementById('kg-graph');
      const summary = document.getElementById('kg-graph-summary');
      const graphEdges = (edges || []).slice(0, 500);
      const stats = summarizeKgEdges(edges || []);
      summary.textContent = `筛选后节点: ${stats.nodeCount}，边: ${stats.edgeCount}，当前绘图: ${graphEdges.length} 条边。`;
      container.innerHTML = '';
      kgGraphState = null;
      if (!window.d3) {
        container.innerHTML = '<div class="kg-empty">D3 未加载，无法渲染 KG 图。</div>';
        return;
      }
      if (!graphEdges.length) {
        container.innerHTML = '<div class="kg-empty">暂无 KG 边</div>';
        return;
      }
      const degree = {};
      graphEdges.forEach((edge) => {
        degree[edge.src_code] = (degree[edge.src_code] || 0) + 1;
        degree[edge.dst_code] = (degree[edge.dst_code] || 0) + 1;
      });
      const nodes = Object.keys(degree).sort().map((code) => ({ id: code, degree: degree[code] }));
      const links = graphEdges.map((edge) => ({
        source: edge.src_code,
        target: edge.dst_code,
        edge_type: edge.edge_type,
        edge_weight: Number(edge.edge_weight) || 0,
        report_id: edge.report_id,
        source_id: edge.source_id,
        month: edge.month,
      }));
      const width = Math.max(container.clientWidth, 320);
      const height = Math.max(container.clientHeight, 520);
      const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
      const viewport = svg.append('g');
      const zoom = d3.zoom().scaleExtent([0.15, 5]).on('zoom', (event) => viewport.attr('transform', event.transform));
      svg.call(zoom);
      const maxDegree = d3.max(nodes, (d) => Number(d.degree) || 0) || 1;
      const radiusScale = d3.scaleSqrt().domain([1, maxDegree]).range([6, 18]);
      const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id((d) => d.id).distance(90).strength(0.3))
        .force('charge', d3.forceManyBody().strength(-220))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius((d) => radiusScale(Number(d.degree) || 1) + 5));
      const link = viewport.append('g').attr('stroke-opacity', 0.52).selectAll('line').data(links).join('line')
        .attr('stroke-width', (d) => 0.8 + Math.min(Math.max(Number(d.edge_weight) || 0, 0), 2.4))
        .attr('stroke', (d) => edgeTypeColor(d.edge_type));
      link.append('title').text((d) => `${d.source.id || d.source} -> ${d.target.id || d.target}\n${d.edge_type}\nweight=${(Number(d.edge_weight) || 0).toFixed(4)}\nreport=${d.report_id || '-'}`);
      const node = viewport.append('g').selectAll('circle').data(nodes).join('circle')
        .attr('r', (d) => radiusScale(Number(d.degree) || 1))
        .attr('fill', '#fff')
        .attr('stroke', '#1677ff')
        .attr('stroke-width', 1.8)
        .call(d3.drag()
          .on('start', (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on('end', (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));
      node.append('title').text((d) => `code=${d.id}\ndegree=${d.degree}`);
      const labels = viewport.append('g').selectAll('text').data(nodes).join('text')
        .text((d) => d.id).attr('font-size', 10).attr('fill', '#172033').attr('dx', 7).attr('dy', 3).style('pointer-events', 'none');
      simulation.on('tick', () => {
        link.attr('x1', (d) => d.source.x).attr('y1', (d) => d.source.y).attr('x2', (d) => d.target.x).attr('y2', (d) => d.target.y);
        node.attr('cx', (d) => d.x).attr('cy', (d) => d.y);
        labels.attr('x', (d) => d.x).attr('y', (d) => d.y);
      });
      kgGraphState = { svg, zoom, nodes, width, height };
      setTimeout(() => fitKgGraph(0), 550);
    }

    function render() {
      const edges = filteredEdges();
      const stats = summarizeKgEdges(edges);
      document.getElementById('stat-filter-nodes').textContent = stats.nodeCount;
      document.getElementById('stat-filter-edges').textContent = stats.edgeCount;
      renderTypeList(stats.typeCounts);
      renderGraph(edges);
      renderEdgeTable(edges);
    }

    function init() {
      const summary = kgGraphData.summary || {};
      document.getElementById('stat-nodes').textContent = summary.node_count || 0;
      document.getElementById('stat-edges').textContent = summary.edge_count || 0;
      document.getElementById('stat-relations').textContent = summary.relation_count || 0;
      document.getElementById('kg-paths').textContent = `完整股票图边: ${kgGraphData.stock_graph_edges_path || '-'} | LLM子图边: ${kgGraphData.kg_stock_edges_llm_path || '-'} | LLM关系: ${kgGraphData.report_llm_relations_path || '-'}`;
      fillSelect(document.getElementById('kg-month-select'), kgGraphData.months || [], '全部月份');
      fillSelect(document.getElementById('kg-edge-type-select'), kgGraphData.edge_types || [], '全部边类型');
      ['kg-month-select', 'kg-edge-type-select', 'kg-search-input'].forEach((id) => {
        document.getElementById(id).addEventListener(id === 'kg-search-input' ? 'input' : 'change', render);
      });
      document.getElementById('kg-zoom-in').addEventListener('click', () => kgGraphState?.svg.transition().duration(180).call(kgGraphState.zoom.scaleBy, 1.25));
      document.getElementById('kg-zoom-out').addEventListener('click', () => kgGraphState?.svg.transition().duration(180).call(kgGraphState.zoom.scaleBy, 0.8));
      document.getElementById('kg-reset').addEventListener('click', () => kgGraphState?.svg.transition().duration(220).call(kgGraphState.zoom.transform, d3.zoomIdentity));
      document.getElementById('kg-fit').addEventListener('click', () => fitKgGraph(220));
      window.addEventListener('resize', render);
      render();
    }
    init();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
