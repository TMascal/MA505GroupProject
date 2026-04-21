# Timothy Mascal
# April 2026
# Created with the Aid of Claude Code
#
# Causal analysis using DoWhy for identification and estimation.
# FCI (via causal-learn) is used for discovery to account for possible
# unobserved confounders — it produces a PAG rather than a CPDAG, so
# bidirected edges (i ↔ j) are preserved as latent common causes.
#
# --- Assumptions encoded as background knowledge ---
#
# Exogeneity:
#   - is_military  : flight type is determined before the flight; nothing
#                    in the graph causes it.
#   - weather      : only subregion can cause local weather patterns;
#                    no in-graph event causes weather.
#   - subregion    : geographic region is fixed; no in-graph event causes it.
#
# Causal ordering:
#   - pilot_error  → cfit  (required): controlled flight into terrain is a
#                    downstream consequence of pilot error, never the cause.
#   - subregion    → pilot_error, shot_down, cfit, weather  (not the reverse).
#   - mechanical   → high_lethality  (not the reverse).
#
# Outcome:
#   - high_lethality : binary outcome (FatalityRate > 0.5); nothing in the
#                      graph is caused by the outcome.
#
# Data filters applied before discovery:
#   - Rows with unknown or undetermined cause are excluded.
#   - Rows where Aboard == 1 and Fatalities == 1 (solo accidents) are excluded
#     to avoid trivially perfect lethality scores distorting the graph.
#   - Rows where Aboard == 0 (produces Inf FatalityRate) are excluded.

import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import graphviz
from dowhy import CausalModel
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def prepare_data(exclude_solo: bool = True, max_samples: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load and encode data for causal discovery.

    Subregion is integer-coded (0–N) so FCI treats it as a single node.
    FlightType is integer-coded as 3 classes: military=0, commercial=1, civilian=2.
    Causes are binary presence indicators.
    fatality_rate is the continuous normalized fatality rate (fatalities / aboard).
    KCI handles mixed continuous and categorical variables without distributional
    assumptions, so no binning or binarization of the outcome is needed.
    """
    df = pd.read_csv("data/labeled_accidents.csv")

    rate = pd.to_numeric(df["FatalityRate"], errors="coerce")
    df = df[rate.notna() & np.isfinite(rate)].copy()
    df = df.dropna(subset=["Cause"])
    df = df[~df["Cause"].str.lower().str.strip().isin(["unknown", "undetermined"])].copy()
    if exclude_solo:
        df = df[~((df["Aboard"] == 1) & (df["Fatalities"] == 1))].copy()
    df = df[df["Subregion"] != "unknown"].copy()

    # Cause binary indicators
    all_causes: set[str] = set()
    for cell in df["Cause"]:
        for c in str(cell).split(","):
            all_causes.add(c.strip().lower())
    all_causes.discard("unknown")
    all_causes.discard("undetermined")

    for cause in sorted(all_causes):
        df[cause] = df["Cause"].apply(
            lambda raw, _c=cause: int(_c in [t.strip().lower() for t in str(raw).split(",")])
        )
    feature_cols = sorted(all_causes)

    # Flight type — binary is_military
    df["is_military"] = (df["FlightType"].str.lower().str.strip() == "military").astype(int)
    feature_cols = ["is_military"] + feature_cols

    # Subregion — integer coded
    codes = {s: i for i, s in enumerate(sorted(df["Subregion"].unique()))}
    df["subregion"] = df["Subregion"].map(codes)
    feature_cols = ["subregion"] + feature_cols

    # Outcome — binary for FCI structure learning, continuous for regression
    rate = pd.to_numeric(df["FatalityRate"], errors="coerce")
    df["high_lethality"] = (rate > 0.5).astype(float)
    df.loc[rate.isna(), "high_lethality"] = np.nan
    df["fatality_rate"] = rate
    feature_cols = feature_cols + ["high_lethality"]

    data_df = df[feature_cols + ["fatality_rate"]].dropna().reset_index(drop=True)
    if max_samples is not None and len(data_df) > max_samples:
        data_df = data_df.sample(max_samples, random_state=42).reset_index(drop=True)
    return data_df, feature_cols


def build_background_knowledge() -> BackgroundKnowledge:
    """Encode domain assumptions as causal-learn background knowledge.

    These constraints are passed to FCI to orient edges that data alone
    cannot resolve, and to rule out causal directions that violate
    domain logic.
    """
    bk = BackgroundKnowledge()
    # Weather is exogenous except for subregion
    bk.add_forbidden_by_pattern("(?!subregion$).*", "weather")
    bk.add_forbidden_by_pattern("weather", "subregion")
    # Subregion is a root — nothing causes it
    bk.add_forbidden_by_pattern("cfit",        "subregion")
    bk.add_forbidden_by_pattern("shot_down",   "subregion")
    bk.add_forbidden_by_pattern("pilot_error", "subregion")
    # is_military is exogenous — nothing in the graph causes it
    bk.add_forbidden_by_pattern(".*", "is_military")
    # pilot_error → cfit is required; cfit → pilot_error is impossible
    bk.add_required_by_pattern("pilot_error", "cfit")
    bk.add_forbidden_by_pattern("cfit", "pilot_error")
    # high_lethality is the outcome — it cannot cause anything
    bk.add_forbidden_by_pattern("high_lethality", ".*")
    # mechanical → high_lethality (not the reverse)
    bk.add_forbidden_by_pattern("high_lethality", "mechanical_trouble")
    return bk


def pag_to_dowhy_graph(nodes: list[str], adj: np.ndarray) -> nx.DiGraph:
    """Convert a causal-learn PAG adjacency matrix to a networkx DiGraph for DoWhy.

    PAG adjacency matrix encoding — adj[i][j] is the mark AT node i:
      -1  tail  (—)
       1  arrowhead  (>)
       2  circle  (o)

    Conversion rules:
      tail — arrowhead  (adj[i][j]==-1, adj[j][i]==1)  →  directed edge i → j
      arrowhead — arrowhead (both==1)                   →  bidirected i ↔ j:
            represented as a latent node L_i_j with edges L→i and L→j
      All other mark combinations (circles, undirected) are skipped with a warning,
      as they represent genuine uncertainty that cannot be collapsed to a DAG.
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)

    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            mi = adj[i][j]  # mark at node i
            mj = adj[j][i]  # mark at node j
            if mi == 0 and mj == 0:
                continue  # no edge
            elif mi == -1 and mj == 1:
                dag.add_edge(nodes[i], nodes[j])
            elif mi == 1 and mj == -1:
                dag.add_edge(nodes[j], nodes[i])
            elif mi == 1 and mj == 1:
                # Bidirected edge → latent common cause
                latent = f"U_{nodes[i]}_{nodes[j]}"
                dag.add_node(latent, observed=False)
                dag.add_edge(latent, nodes[i])
                dag.add_edge(latent, nodes[j])
            else:
                print(f"  [warn] Skipping uncertain edge between '{nodes[i]}' and "
                      f"'{nodes[j]}' (marks: {mi}, {mj})")

    return dag


def render_graph(dag: nx.DiGraph, filename: str = "output/dowhy/pag") -> graphviz.Digraph:
    """Render the DoWhy causal graph to a PNG file.

    Observed nodes are drawn as plain ellipses.
    Latent (hidden confounder) nodes are drawn as dashed ellipses in grey,
    labelled with the two variables they confound.
    Edges from latent nodes are drawn in grey to visually distinguish them.

    Args:
        dag:      The networkx DiGraph produced by pag_to_dowhy_graph.
        filename: Output path without extension — PNG is saved at filename.png.

    Returns:
        The graphviz Digraph object.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    dot = graphviz.Digraph(name="PAG", graph_attr={"rankdir": "LR"})

    for node, attrs in dag.nodes(data=True):
        if attrs.get("observed") is False:
            dot.node(node, style="dashed", color="grey", fontcolor="grey", shape="ellipse")
        else:
            dot.node(node, shape="ellipse")

    for src, dst in dag.edges():
        is_latent_src = dag.nodes[src].get("observed") is False
        dot.edge(src, dst, color="grey" if is_latent_src else "black")

    dot.render(filename, format="png", cleanup=True)
    print(f"  Graph saved to {filename}.png")
    return dot


def identify_and_estimate(dag: nx.DiGraph, data_df: pd.DataFrame,
                           treatment: str, outcome: str) -> float | None:
    """Run DoWhy identification and linear regression estimation for one treatment.

    Returns the estimated total causal effect, or None if estimation failed.
    """
    model = CausalModel(
        data=data_df,
        treatment=treatment,
        outcome=outcome,
        graph=dag,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified)

    try:
        estimate = model.estimate_effect(
            identified,
            method_name="backdoor.linear_regression",
        )
        print(f"  Estimated total effect of '{treatment}' on '{outcome}': {estimate.value:.4f}")
        return estimate.value
    except Exception as e:
        print(f"  Estimation failed: {e}")
        return None




if __name__ == "__main__":
    # FCI discovers structure on binary high_lethality; regression uses continuous fatality_rate.
    FCI_OUTCOME  = "high_lethality"
    REG_OUTCOME  = "fatality_rate"
    TREATMENTS = ["cfit", "collision", "fire", "fuel", "mechanical",
                  "pilot_error", "sabotage", "shot_down", "weather"]

    print("Loading and encoding data...")
    data_df, feature_cols = prepare_data(exclude_solo=True, max_samples=5000)
    print(f"  {len(data_df)} rows, {len(feature_cols)} nodes: {feature_cols}\n")

    PAG_CACHE = "output/pag_kci.pkl"
    bk = build_background_knowledge()

    if os.path.exists(PAG_CACHE):
        print(f"Loading cached PAG from {PAG_CACHE}...")
        with open(PAG_CACHE, "rb") as f:
            pag, cached_cols = pickle.load(f)
        if cached_cols != feature_cols:
            print("  [warn] Cached node list differs from current data — rerunning FCI.")
            pag = None
    else:
        pag = None

    if pag is None:
        print("Running FCI with chi-square to learn PAG (accounts for latent confounders)...")
        pag, _ = fci(
            data_df[feature_cols].to_numpy(dtype=float),
            independence_test_method="chisq",
            alpha=0.05,
            background_knowledge=bk,
            show_progress=True,
            node_names=feature_cols,
        )
        with open(PAG_CACHE, "wb") as f:
            pickle.dump((pag, feature_cols), f)
        print(f"  PAG saved to {PAG_CACHE}.\n")

    print("  FCI complete.\n")

    print("Converting PAG to DoWhy graph...")
    dag = pag_to_dowhy_graph(feature_cols, pag.graph)

    # Rename the binary discovery node to the continuous regression outcome so
    # the graph topology is preserved but DoWhy regresses against fatality_rate.
    nx.relabel_nodes(dag, {FCI_OUTCOME: REG_OUTCOME}, copy=False)

    directed  = [(u, v) for u, v in dag.edges() if not dag.nodes[u].get("observed") is False]
    latent    = [n for n, d in dag.nodes(data=True) if d.get("observed") is False]
    print(f"  Directed edges : {directed}")
    print(f"  Latent nodes   : {latent}\n")
    render_graph(dag, filename="output/dowhy/pag")

    # Drop the binary column — regression uses fatality_rate (continuous)
    reg_data = data_df.drop(columns=[FCI_OUTCOME])

    total_effects = {}
    for treatment in TREATMENTS:
        print(f"{'='*60}")
        print(f"Treatment: {treatment} → {REG_OUTCOME}")
        print(f"{'='*60}")
        total_effects[treatment] = identify_and_estimate(dag, reg_data, treatment, REG_OUTCOME)
        print()

    print(f"{'='*60}")
    print("TOTAL CAUSAL IMPACT SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Cause':<15} {'Effect on FatalityRate':>22}")
    print(f"  {'-'*38}")
    for treatment, effect in sorted(total_effects.items(),
                                    key=lambda x: x[1] if x[1] is not None else 0,
                                    reverse=True):
        val = f"{effect:>+.4f}" if effect is not None else "   n/a"
        print(f"  {treatment:<15} {val:>22}")