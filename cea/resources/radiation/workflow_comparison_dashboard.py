"""Generate an interactive comparison dashboard for roof-workflow test runs."""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
from typing import Any

import pandas as pd

DEFAULT_TESTS_ROOT = r"C:\Users\Andre\cea-scenarios\tests"
DEFAULT_OUTPUT_HTML = r"C:\Users\Andre\cea-scenarios\tests\ucea_custom_roof_dashboard.html"
DEFAULT_PRIMARY_KPI = "delta_total_E_PV_gen_kWh"
DEFAULT_EQUIVALENCE_REL = 0.01

WORKFLOW_0 = "WF0"
WORKFLOW_1 = "WF1"
CASE_TABLE_COLUMNS = [
    "case_id",
    "test_group",
    "variant",
    "pv_panel",
    "delta_total_E_PV_gen_kWh",
    "delta_total_area_PV_m2",
    "delta_total_specific_yield_kWh_m2",
    "equivalence_threshold",
    "verdict",
    "quality_issues",
]


def _normalise_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _discover_comparison_roots(tests_root: str) -> list[str]:
    pattern = os.path.join(os.path.abspath(tests_root), "**", "outputs", "data", "roof-workflow-comparison")
    paths = [path for path in glob.glob(pattern, recursive=True) if os.path.isdir(path)]
    unique = {_normalise_path(path): os.path.abspath(path) for path in paths}
    return [unique[key] for key in sorted(unique)]


def _case_context(tests_root: str, comparison_root: str) -> dict[str, str]:
    relative_root = os.path.relpath(comparison_root, tests_root)
    parts = relative_root.split(os.sep)
    try:
        index = parts.index("outputs")
        context_parts = parts[:index]
    except ValueError:
        context_parts = parts
    if not context_parts:
        context_parts = [os.path.basename(tests_root.rstrip("\\/")) or "tests"]
    test_group = context_parts[0]
    variant = " / ".join(context_parts[1:]) if len(context_parts) > 1 else "base"
    case_id = " > ".join(context_parts)
    return {"case_id": case_id, "test_group": test_group, "variant": variant}


def _safe_read_csv(path: str) -> tuple[pd.DataFrame | None, str | None]:
    try:
        return pd.read_csv(path), None
    except Exception as exc:  # pragma: no cover
        return None, f"failed_to_read:{os.path.basename(path)} ({exc})"


def _preferred_metrics_file(comparison_root: str) -> tuple[str | None, list[str]]:
    files = sorted(glob.glob(os.path.join(comparison_root, "scenario_metrics*.csv")))
    if not files:
        return None, []
    default_file = os.path.join(comparison_root, "scenario_metrics.csv")
    if default_file in files:
        return default_file, files
    files_sorted = sorted(files, key=lambda path: (len(os.path.basename(path)), os.path.basename(path)))
    return files_sorted[0], files


def _extract_numeric_suffix(value: str) -> float | None:
    matches = re.findall(r"-?\d+(?:[.,]\d+)?", value)
    if not matches:
        return None
    token = matches[-1].replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _merge_issues(*issue_groups: list[str]) -> str:
    merged: list[str] = []
    for group in issue_groups:
        merged.extend(group)
    return ";".join(list(dict.fromkeys([item for item in merged if item])))


def _to_number(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        parsed = float(value)
        if pd.isna(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _load_panel_rows(comparison_root: str) -> tuple[list[dict[str, Any]], list[str]]:
    path = os.path.join(comparison_root, "panel_placement_summary.csv")
    if not os.path.exists(path):
        return [], ["missing_panel_placement_summary"]
    panel_df, panel_error = _safe_read_csv(path)
    if panel_error:
        return [], [panel_error]
    if panel_df is None or panel_df.empty:
        return [], ["empty_panel_placement_summary"]
    return panel_df.to_dict(orient="records"), []


def _load_geometry_preview(comparison_root: str) -> tuple[str | None, str | None, list[str]]:
    image_paths = sorted(glob.glob(os.path.join(comparison_root, "*_workflow_geometry_comparison_3d.png")))
    if not image_paths:
        return None, None, ["missing_geometry_comparison_image"]
    image_path = image_paths[0]
    try:
        with open(image_path, "rb") as stream:
            encoded = base64.b64encode(stream.read()).decode("ascii")
    except Exception as exc:  # pragma: no cover
        return None, image_path, [f"failed_to_read_image:{os.path.basename(image_path)} ({exc})"]
    return f"data:image/png;base64,{encoded}", image_path, []


def _fallback_rows_from_metrics(
    metadata: dict[str, Any],
    scenario_metrics_df: pd.DataFrame,
    panel_rows: list[dict[str, Any]],
    image_data_uri: str | None,
    image_path: str | None,
    issues: list[str],
) -> list[dict[str, Any]]:
    if scenario_metrics_df.empty:
        row = dict(metadata)
        row.update(
            {
                "pv_panel": "unknown",
                "missing_status": "missing",
                "source_mode": "missing",
                "quality_issues": _merge_issues(issues, ["missing_scenario_deltas_and_metrics"]),
                "panel_placement_rows": panel_rows,
                "geometry_image_data_uri": image_data_uri,
                "geometry_image_path": image_path or "",
            }
        )
        return [row]

    rows: list[dict[str, Any]] = []
    grouped = scenario_metrics_df.groupby("pv_panel", dropna=False)
    for pv_panel, panel_df in grouped:
        row = dict(metadata)
        panel_key = str(pv_panel) if pd.notna(pv_panel) else "unknown"
        workflow_map = {
            workflow_id: data.iloc[0]
            for workflow_id, data in panel_df.groupby("workflow_id", dropna=False)
            if not data.empty
        }
        wf0 = workflow_map.get(WORKFLOW_0)
        wf1 = workflow_map.get(WORKFLOW_1)
        missing_workflows = [workflow for workflow in [WORKFLOW_0, WORKFLOW_1] if workflow not in workflow_map]
        workflow_issues = [f"missing_workflow_{workflow}" for workflow in missing_workflows]
        for metric in ["E_PV_gen_kWh", "area_PV_m2", "specific_yield_kWh_m2", "radiation_kWh_m2"]:
            wf0_value = _to_number(wf0.get(metric) if wf0 is not None else None)
            wf1_value = _to_number(wf1.get(metric) if wf1 is not None else None)
            row[f"value_wf0_{metric}"] = wf0_value
            row[f"value_wf1_{metric}"] = wf1_value
            row[f"delta_total_{metric}"] = None if wf0_value is None or wf1_value is None else wf1_value - wf0_value

        row.update(
            {
                "pv_panel": panel_key,
                "missing_status": "ok" if not missing_workflows else "missing",
                "source_mode": "metrics_fallback",
                "quality_issues": _merge_issues(issues + ["missing_scenario_deltas_used_metrics_fallback"], workflow_issues),
                "panel_placement_rows": [entry for entry in panel_rows if str(entry.get("pv_panel", "")) == panel_key],
                "geometry_image_data_uri": image_data_uri,
                "geometry_image_path": image_path or "",
            }
        )
        rows.append(row)
    return rows


def collect_cases(tests_root: str) -> pd.DataFrame:
    """Collect and normalise case rows from roof-workflow comparison folders."""

    tests_root_abs = os.path.abspath(tests_root)
    rows: list[dict[str, Any]] = []
    for comparison_root in _discover_comparison_roots(tests_root_abs):
        metadata = _case_context(tests_root_abs, comparison_root)
        metadata["comparison_root"] = comparison_root
        metadata["comparison_root_rel"] = os.path.relpath(comparison_root, tests_root_abs)
        metadata["sweep_value"] = _extract_numeric_suffix(metadata["variant"])

        panel_rows, panel_issues = _load_panel_rows(comparison_root)
        image_data_uri, image_path, image_issues = _load_geometry_preview(comparison_root)

        preferred_metrics_path, all_metrics_files = _preferred_metrics_file(comparison_root)
        metrics_issues: list[str] = []
        scenario_metrics_df = pd.DataFrame()
        if preferred_metrics_path is None:
            metrics_issues.append("missing_scenario_metrics")
        else:
            scenario_metrics_df, metrics_error = _safe_read_csv(preferred_metrics_path)
            if metrics_error:
                metrics_issues.append(metrics_error)
                scenario_metrics_df = pd.DataFrame()
            elif scenario_metrics_df is None:
                scenario_metrics_df = pd.DataFrame()

        base_issues = panel_issues + image_issues + metrics_issues
        deltas_path = os.path.join(comparison_root, "scenario_deltas.csv")
        if not os.path.exists(deltas_path):
            rows.extend(
                _fallback_rows_from_metrics(
                    metadata=metadata,
                    scenario_metrics_df=scenario_metrics_df,
                    panel_rows=panel_rows,
                    image_data_uri=image_data_uri,
                    image_path=image_path,
                    issues=base_issues + ["missing_scenario_deltas"],
                )
            )
            continue

        deltas_df, deltas_error = _safe_read_csv(deltas_path)
        if deltas_error or deltas_df is None or deltas_df.empty:
            fallback_issue = deltas_error or "empty_scenario_deltas"
            rows.extend(
                _fallback_rows_from_metrics(
                    metadata=metadata,
                    scenario_metrics_df=scenario_metrics_df,
                    panel_rows=panel_rows,
                    image_data_uri=image_data_uri,
                    image_path=image_path,
                    issues=base_issues + [fallback_issue],
                )
            )
            continue

        for _, delta_row in deltas_df.iterrows():
            row = dict(metadata)
            row.update(delta_row.to_dict())
            panel_name = str(row.get("pv_panel") or "unknown")
            row.update(
                {
                    "pv_panel": panel_name,
                    "source_mode": "scenario_deltas",
                    "quality_issues": _merge_issues(base_issues),
                    "panel_placement_rows": [entry for entry in panel_rows if str(entry.get("pv_panel", "")) == panel_name],
                    "geometry_image_data_uri": image_data_uri,
                    "geometry_image_path": image_path or "",
                    "scenario_metrics_source": preferred_metrics_path or "",
                    "scenario_metrics_files": all_metrics_files,
                }
            )
            rows.append(row)

    cases_df = pd.DataFrame(rows)
    if cases_df.empty:
        return cases_df
    defaults = {
        "delta_total_E_PV_gen_kWh": None,
        "delta_total_area_PV_m2": None,
        "delta_total_specific_yield_kWh_m2": None,
        "delta_total_radiation_kWh_m2": None,
        "delta_total_shading_cv_aw": None,
        "delta_total_shading_cv_aw_detrended": None,
        "value_wf0_E_PV_gen_kWh": None,
        "value_wf1_E_PV_gen_kWh": None,
        "missing_status": None,
        "quality_issues": "",
    }
    for column, default in defaults.items():
        if column not in cases_df.columns:
            cases_df[column] = default
    return cases_df


def classify_cases(df: pd.DataFrame, primary_kpi: str, equivalence_rel: float) -> pd.DataFrame:
    """Classify cases with a relative tolerance around WF0."""

    classified = df.copy()
    if classified.empty:
        classified["primary_kpi"] = primary_kpi
        classified["equivalence_rel"] = equivalence_rel
        classified["equivalence_threshold"] = None
        classified["verdict"] = []
        classified["abs_primary_delta"] = []
        return classified

    if primary_kpi not in classified.columns:
        classified[primary_kpi] = None
        classified["quality_issues"] = classified["quality_issues"].fillna("").astype(str).str.strip(";")
        classified["quality_issues"] = classified["quality_issues"].apply(
            lambda current: ";".join([entry for entry in [current, f"missing_primary_kpi:{primary_kpi}"] if entry])
        )

    baseline_column = primary_kpi.replace("delta_total_", "value_wf0_")
    if baseline_column not in classified.columns:
        classified[baseline_column] = None
        classified["quality_issues"] = classified["quality_issues"].fillna("").astype(str).str.strip(";")
        classified["quality_issues"] = classified["quality_issues"].apply(
            lambda current: ";".join([entry for entry in [current, f"missing_baseline_kpi:{baseline_column}"] if entry])
        )

    primary_delta = pd.to_numeric(classified[primary_kpi], errors="coerce")
    primary_baseline = pd.to_numeric(classified[baseline_column], errors="coerce")
    threshold = (primary_baseline.abs() * equivalence_rel).fillna(equivalence_rel)

    verdicts: list[str] = []
    for delta, current_threshold in zip(primary_delta.tolist(), threshold.tolist(), strict=False):
        if pd.isna(delta):
            verdicts.append("Unknown")
        elif abs(delta) <= current_threshold:
            verdicts.append("Equivalent")
        elif delta > 0:
            verdicts.append("Improved")
        else:
            verdicts.append("Regression")

    classified["primary_kpi"] = primary_kpi
    classified["equivalence_rel"] = equivalence_rel
    classified["equivalence_threshold"] = threshold
    classified["verdict"] = verdicts
    classified["abs_primary_delta"] = primary_delta.abs()
    return classified.sort_values(by=["abs_primary_delta", "case_id"], ascending=[False, True]).reset_index(drop=True)


def _records_for_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, float) and pd.isna(value):
                clean[key] = None
            else:
                clean[key] = value
        records.append(clean)
    return records


def _panel_rows_long(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "panel_placement_rows" not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        panel_rows = record.get("panel_placement_rows") or []
        if not isinstance(panel_rows, list):
            continue
        for panel_row in panel_rows:
            if not isinstance(panel_row, dict):
                continue
            merged = {
                "case_id": record.get("case_id"),
                "test_group": record.get("test_group"),
                "variant": record.get("variant"),
                "pv_panel": record.get("pv_panel"),
            }
            merged.update(panel_row)
            rows.append(merged)
    return rows


def render_dashboard(df_cases: pd.DataFrame, output_html: str) -> None:
    """Render interactive HTML and normalised CSV exports."""

    output_html = os.path.abspath(output_html)
    output_dir = os.path.dirname(output_html)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if df_cases.empty:
        with open(output_html, "w", encoding="utf-8") as stream:
            stream.write("<html><body><h1>No cases found</h1></body></html>\n")
        return

    primary_kpi = str(df_cases.get("primary_kpi", pd.Series([DEFAULT_PRIMARY_KPI])).iloc[0] or DEFAULT_PRIMARY_KPI)
    equivalence_rel = float(df_cases.get("equivalence_rel", pd.Series([DEFAULT_EQUIVALENCE_REL])).iloc[0])
    records = _records_for_json(df_cases)
    panel_rows = _panel_rows_long(df_cases)
    output_stem = os.path.splitext(output_html)[0]
    cases_csv_path = f"{output_stem}_cases.csv"
    panel_csv_path = f"{output_stem}_panel_rows.csv"
    pd.DataFrame(records).to_csv(cases_csv_path, index=False)
    pd.DataFrame(panel_rows).to_csv(panel_csv_path, index=False)

    payload = {
        "cases": records,
        "panel_rows": panel_rows,
        "primary_kpi": primary_kpi,
        "equivalence_rel": equivalence_rel,
        "columns": CASE_TABLE_COLUMNS,
    }
    payload_json = json.dumps(payload, ensure_ascii=True, default=str)

    html_output = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UCEA Custom-Roof Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { margin: 0; font-family: "Trebuchet MS", "Segoe UI", sans-serif; background: #f3f6ef; color: #1f2c22; }
    .page { max-width: 1400px; margin: 0 auto; padding: 20px; display: grid; gap: 14px; }
    .hero { background: linear-gradient(120deg, #0f766e, #14532d); color: #fff; border-radius: 12px; padding: 16px; }
    .controls, .kpis, .charts, .table-wrap, .details { background: #fff; border: 1px solid #d7dfd4; border-radius: 10px; padding: 12px; }
    .controls { display: grid; gap: 8px; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); align-items: end; }
    label { display: grid; gap: 5px; font-size: 0.9rem; color: #42614f; }
    input, select, button { border: 1px solid #d7dfd4; border-radius: 7px; padding: 8px; font: inherit; }
    button { cursor: pointer; background: #14532d; color: #fff; border: 0; font-weight: 700; }
    .kpis { display: grid; gap: 8px; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }
    .kpi { border: 1px solid #d7dfd4; border-radius: 8px; padding: 8px; background: #f6fbf5; }
    .kpi .l { font-size: 0.8rem; color: #42614f; } .kpi .v { font-size: 1.2rem; font-weight: 700; }
    .charts { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
    .chart { min-height: 320px; border: 1px solid #d7dfd4; border-radius: 8px; }
    .table-scroll { max-height: 380px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th, td { border-bottom: 1px solid #e4ebe1; padding: 6px; text-align: left; }
    th { position: sticky; top: 0; background: #f1f5ef; cursor: pointer; }
    .tag { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.74rem; font-weight: 700; }
    .tag.Improved { background: #dcfce7; color: #166534; } .tag.Regression { background: #fee2e2; color: #991b1b; }
    .tag.Equivalent { background: #fef3c7; color: #b45309; } .tag.Unknown { background: #e5e7eb; color: #374151; }
    .detail-grid { display: grid; gap: 8px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .card { border: 1px solid #d7dfd4; border-radius: 8px; padding: 8px; background: #f8fcf7; }
    .card img { width: 100%; border: 1px solid #d7dfd4; border-radius: 6px; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h2 style="margin:0;">UCEA Custom-Roof Comparison Dashboard</h2>
      <div style="margin-top:6px;">Primary KPI: <strong>__PRIMARY_KPI__</strong> | Equivalence tolerance: <strong>__EQUIV_REL__</strong></div>
      <div style="margin-top:4px;">Normalised tables: <code>__CASES_CSV__</code> and <code>__PANEL_CSV__</code></div>
    </section>
    <section class="controls">
      <label>Search case<input id="searchInput" type="text" placeholder="Type case, variant, issue"></label>
      <label>Verdict<select id="verdictFilter"></select></label>
      <label>Group<select id="groupFilter"></select></label>
      <label>Deep-dive group<select id="deepDiveGroup"></select></label>
      <button id="downloadCasesBtn" type="button">Download Filtered Cases CSV</button>
      <button id="downloadPanelBtn" type="button">Download Filtered Panel CSV</button>
    </section>
    <section class="kpis" id="kpiContainer"></section>
    <section class="charts">
      <div class="chart" id="rankingChart"></div>
      <div class="chart" id="correlationChart"></div>
      <div class="chart" id="deepDiveChart"></div>
    </section>
    <section class="table-wrap">
      <h3 style="margin-top:0;">Case Comparison Table</h3>
      <div class="table-scroll"><table id="caseTable"><thead></thead><tbody></tbody></table></div>
    </section>
    <section class="details">
      <h3 style="margin-top:0;">Per-Case Details</h3>
      <div class="detail-grid" id="caseDetails"></div>
    </section>
  </div>
<script>
const DASHBOARD_DATA = __PAYLOAD_JSON__;
const allCases = DASHBOARD_DATA.cases;
const allPanelRows = DASHBOARD_DATA.panel_rows;
const columns = DASHBOARD_DATA.columns;
const primaryKpi = DASHBOARD_DATA.primary_kpi;
let filteredCases = [...allCases];
let sortState = { column: "abs_primary_delta", ascending: false };
function toNumber(v){ if(v===null||v===undefined||v===""){return null;} const n=Number(v); return Number.isFinite(n)?n:null; }
function fmt(v,d=2){ const n=toNumber(v); return n===null?"n/a":n.toLocaleString(undefined,{maximumFractionDigits:d}); }
function uniq(rows,key){ return [...new Set(rows.map(r=>r[key]||"Unknown"))].sort((a,b)=>String(a).localeCompare(String(b))); }
function options(selectId, values){ document.getElementById(selectId).innerHTML = values.map(v=>`<option value="${v}">${v}</option>`).join(""); }
function populateFilters(){ options("verdictFilter",["All",...uniq(allCases,"verdict")]); options("groupFilter",["All",...uniq(allCases,"test_group")]); options("deepDiveGroup",["All",...uniq(allCases,"test_group")]); }
function applyFilters(){
  const q=document.getElementById("searchInput").value.trim().toLowerCase();
  const vf=document.getElementById("verdictFilter").value;
  const gf=document.getElementById("groupFilter").value;
  filteredCases=allCases.filter(r=>{
    if(vf!=="All" && String(r.verdict||"Unknown")!==vf){return false;}
    if(gf!=="All" && String(r.test_group||"Unknown")!==gf){return false;}
    if(!q){return true;}
    const blob=[r.case_id,r.test_group,r.variant,r.pv_panel,r.quality_issues].join(" ").toLowerCase();
    return blob.includes(q);
  });
  sortRows(); renderKpis(); renderRanking(); renderCorrelation(); renderDeepDive(); renderTable(); renderDetails();
}
function sortRows(){ const c=sortState.column; filteredCases.sort((a,b)=>{ const an=toNumber(a[c]), bn=toNumber(b[c]); if(an!==null&&bn!==null){ return sortState.ascending?an-bn:bn-an;} const as=String(a[c]??""), bs=String(b[c]??""); return sortState.ascending?as.localeCompare(bs):bs.localeCompare(as); }); }
function colour(v){ if(v==="Improved"){return "#166534";} if(v==="Regression"){return "#991b1b";} if(v==="Equivalent"){return "#b45309";} return "#4b5563"; }
function renderKpis(){
  const total=filteredCases.length, improved=filteredCases.filter(r=>r.verdict==="Improved").length, equivalent=filteredCases.filter(r=>r.verdict==="Equivalent").length, regression=filteredCases.filter(r=>r.verdict==="Regression").length, unknown=filteredCases.filter(r=>r.verdict==="Unknown").length;
  const sumE=filteredCases.reduce((a,r)=>a+(toNumber(r.delta_total_E_PV_gen_kWh)||0),0), sumA=filteredCases.reduce((a,r)=>a+(toNumber(r.delta_total_area_PV_m2)||0),0);
  const cards=[["Cases",total],["Improved",improved],["Equivalent",equivalent],["Regression",regression],["Unknown",unknown],["Total ΔE (kWh)",fmt(sumE,2)],["Total ΔArea (m²)",fmt(sumA,2)]];
  document.getElementById("kpiContainer").innerHTML=cards.map(c=>`<div class="kpi"><div class="l">${c[0]}</div><div class="v">${c[1]}</div></div>`).join("");
}
function renderRanking(){ const ordered=[...filteredCases].sort((a,b)=>(toNumber(b[primaryKpi])||0)-(toNumber(a[primaryKpi])||0)); Plotly.react("rankingChart",[{type:"bar",x:ordered.map(r=>r.case_id),y:ordered.map(r=>toNumber(r[primaryKpi])||0),marker:{color:ordered.map(r=>colour(r.verdict))}}],{title:"Ranking by "+primaryKpi,margin:{t:45,r:20,b:120,l:45},xaxis:{tickangle:-35},yaxis:{title:"Delta"}},{responsive:true}); }
function renderCorrelation(){ Plotly.react("correlationChart",[{type:"scatter",mode:"markers+text",x:filteredCases.map(r=>toNumber(r.delta_total_area_PV_m2)),y:filteredCases.map(r=>toNumber(r.delta_total_E_PV_gen_kWh)),text:filteredCases.map(r=>r.case_id),textposition:"top center",marker:{size:11,color:filteredCases.map(r=>colour(r.verdict)),opacity:0.85}}],{title:"Correlation: ΔArea vs ΔEnergy",margin:{t:45,r:20,b:45,l:55},xaxis:{title:"delta_total_area_PV_m2"},yaxis:{title:"delta_total_E_PV_gen_kWh"}},{responsive:true}); }
function renderDeepDive(){ const group=document.getElementById("deepDiveGroup").value; const selected=filteredCases.filter(r=>group==="All"||String(r.test_group||"Unknown")===group); selected.sort((a,b)=>{ const an=toNumber(a.sweep_value), bn=toNumber(b.sweep_value); if(an!==null&&bn!==null){return an-bn;} return String(a.variant).localeCompare(String(b.variant)); }); Plotly.react("deepDiveChart",[{type:"bar",x:selected.map(r=>r.variant),y:selected.map(r=>toNumber(r.delta_total_E_PV_gen_kWh)||0),marker:{color:selected.map(r=>colour(r.verdict))}}],{title:"Group Deep-Dive: "+group,margin:{t:45,r:20,b:100,l:55},xaxis:{tickangle:-25},yaxis:{title:"delta_total_E_PV_gen_kWh"}},{responsive:true}); }
function renderTable(){
  const table=document.getElementById("caseTable"), thead=table.querySelector("thead"), tbody=table.querySelector("tbody");
  thead.innerHTML=`<tr>${columns.map(c=>`<th data-column="${c}">${c}</th>`).join("")}</tr>`;
  tbody.innerHTML=filteredCases.map(r=>`<tr><td>${r.case_id||""}</td><td>${r.test_group||""}</td><td>${r.variant||""}</td><td>${r.pv_panel||""}</td><td>${fmt(r.delta_total_E_PV_gen_kWh,2)}</td><td>${fmt(r.delta_total_area_PV_m2,2)}</td><td>${fmt(r.delta_total_specific_yield_kWh_m2,3)}</td><td><span class="tag ${r.verdict||"Unknown"}">${r.verdict||"Unknown"}</span></td><td>${r.quality_issues||""}</td></tr>`).join("");
  thead.querySelectorAll("th").forEach(th=>th.addEventListener("click",()=>{ const c=th.getAttribute("data-column"); if(sortState.column===c){sortState.ascending=!sortState.ascending;} else {sortState.column=c;sortState.ascending=true;} sortRows(); renderTable(); renderDetails(); }));
}
function renderDetails(){
  document.getElementById("caseDetails").innerHTML=filteredCases.map(r=>{ const panelRows=Array.isArray(r.panel_placement_rows)?r.panel_placement_rows:[]; const panelTable=panelRows.map(p=>`<tr><td>${p.workflow_id||""}</td><td>${p.direction||""}</td><td>${fmt(p.tilt_deg,2)}</td><td>${fmt(p.panel_area_m2,2)}</td><td>${fmt(p.share_of_generation,3)}</td></tr>`).join(""); const image=r.geometry_image_data_uri?`<img src="${r.geometry_image_data_uri}" alt="Geometry comparison for ${r.case_id}">`:""; return `<article class="card"><h4 style="margin:0 0 5px 0;">${r.case_id}</h4><div style="font-size:.82rem;color:#42614f;margin-bottom:6px;">${r.variant} | <span class="tag ${r.verdict||"Unknown"}">${r.verdict||"Unknown"}</span></div><div style="font-size:.84rem;">ΔE: <strong>${fmt(r.delta_total_E_PV_gen_kWh,2)}</strong> kWh | ΔArea: <strong>${fmt(r.delta_total_area_PV_m2,2)}</strong> m² | ΔYield: <strong>${fmt(r.delta_total_specific_yield_kWh_m2,3)}</strong></div>${image}<table style="width:100%;margin-top:8px;font-size:.78rem;border-collapse:collapse;"><thead><tr><th>Workflow</th><th>Direction</th><th>Tilt</th><th>Panel Area</th><th>Gen Share</th></tr></thead><tbody>${panelTable}</tbody></table><div style="font-size:.78rem;color:#42614f;margin-top:6px;">Issues: ${r.quality_issues||"none"}</div></article>`; }).join("");
}
function csvEscape(v){ const t=String(v??""); if(t.includes(",")||t.includes("\"")||t.includes("\n")){ return "\"" + t.replace(/"/g,"\"\"") + "\""; } return t; }
function toCsv(rows){ if(!rows.length){return "";} const cols=Object.keys(rows[0]); return [cols.join(","), ...rows.map(r=>cols.map(c=>csvEscape(r[c])).join(","))].join("\n"); }
function download(name, text){ const blob=new Blob([text],{type:"text/csv;charset=utf-8;"}); const url=URL.createObjectURL(blob); const link=document.createElement("a"); link.href=url; link.download=name; link.click(); URL.revokeObjectURL(url); }
function exportCases(){ const rows=filteredCases.map(r=>{ const slim={}; columns.forEach(c=>slim[c]=r[c]); slim.primary_kpi=r[primaryKpi]; return slim; }); download("filtered_cases.csv",toCsv(rows)); }
function exportPanel(){ const ids=new Set(filteredCases.map(r=>r.case_id)); download("filtered_panel_rows.csv",toCsv(allPanelRows.filter(r=>ids.has(r.case_id)))); }
document.getElementById("searchInput").addEventListener("input",applyFilters);
document.getElementById("verdictFilter").addEventListener("change",applyFilters);
document.getElementById("groupFilter").addEventListener("change",applyFilters);
document.getElementById("deepDiveGroup").addEventListener("change",renderDeepDive);
document.getElementById("downloadCasesBtn").addEventListener("click",exportCases);
document.getElementById("downloadPanelBtn").addEventListener("click",exportPanel);
populateFilters();
applyFilters();
</script>
</body>
</html>
"""
    html_output = html_output.replace("__PRIMARY_KPI__", primary_kpi)
    html_output = html_output.replace("__EQUIV_REL__", f"{equivalence_rel:.2%}")
    html_output = html_output.replace("__CASES_CSV__", cases_csv_path)
    html_output = html_output.replace("__PANEL_CSV__", panel_csv_path)
    html_output = html_output.replace("__PAYLOAD_JSON__", payload_json)
    with open(output_html, "w", encoding="utf-8") as stream:
        stream.write(html_output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an interactive dashboard from roof-workflow comparison runs.")
    parser.add_argument("--tests-root", default=DEFAULT_TESTS_ROOT, help="Root folder containing test scenarios.")
    parser.add_argument("--output-html", default=DEFAULT_OUTPUT_HTML, help="Path for the generated HTML dashboard.")
    parser.add_argument(
        "--primary-kpi",
        default=DEFAULT_PRIMARY_KPI,
        help="Primary delta KPI column used for ranking and verdict classification.",
    )
    parser.add_argument(
        "--equivalence-rel",
        default=DEFAULT_EQUIVALENCE_REL,
        type=float,
        help="Relative tolerance for Equivalent verdicts (for example 0.01 = ±1%).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    tests_root = os.path.abspath(args.tests_root)
    output_html = os.path.abspath(args.output_html)
    primary_kpi = args.primary_kpi
    equivalence_rel = float(args.equivalence_rel)
    cases = collect_cases(tests_root)
    classified = classify_cases(cases, primary_kpi=primary_kpi, equivalence_rel=equivalence_rel)
    render_dashboard(classified, output_html)
    print(f"[dashboard] Cases collected: {len(classified)}")
    print(f"[dashboard] Output HTML: {output_html}")
    print(f"[dashboard] Normalised CSV exports: {os.path.splitext(output_html)[0]}_cases.csv and {os.path.splitext(output_html)[0]}_panel_rows.csv")


if __name__ == "__main__":
    main()
