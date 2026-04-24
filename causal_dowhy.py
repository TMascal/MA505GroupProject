# Timothy Mascal
# April 2026
# Created with the Aid of Claude Code
#
# Causal analysis using DoWhy for identification and estimation.
# FCI (via causal-learn) is used for discovery to account for possible
# unobserved confounders — it produces a PAG.
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
#   - pilot_error  → mechanical (required): improper operation can damage
#                    equipment; mechanical failure does not cause pilot error.
#   - subregion    → pilot_error, shot_down, cfit, weather  (not the reverse).
#   - mechanical   → high_lethality  (not the reverse).
#   - sabotage     → cfit: disabling aircraft can cause cfit.
#   - sabotage     → fire: explosives/arson cause fire.
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
import statsmodels.api as sm
from dowhy import CausalModel
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def prepare_data(exclude_solo: bool = True, max_samples: int | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load and encode data for causal discovery.

    Subregion is integer-coded (0–N) so FCI treats it as a single node.
    Causes are binary presence indicators.
    high_lethality is a binary outcome: 1 if FatalityRate > 0.5, else 0.
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
    subregion_names = sorted(df["Subregion"].unique())
    codes = {s: i for i, s in enumerate(subregion_names)}
    df["subregion"] = df["Subregion"].map(codes)
    feature_cols = ["subregion"] + feature_cols

    # Outcome — binary: 1 = high fatality rate (>0.5), 0 = low fatality rate (<=0.5)
    rate = pd.to_numeric(df["FatalityRate"], errors="coerce")
    df["high_lethality"] = (rate > 0.5).astype(float)
    df.loc[rate.isna(), "high_lethality"] = np.nan
    feature_cols = feature_cols + ["high_lethality"]

    data_df = df[feature_cols].dropna().reset_index(drop=True)
    if max_samples is not None and len(data_df) > max_samples:
        data_df = data_df.sample(max_samples, random_state=42).reset_index(drop=True)
    return data_df, feature_cols, subregion_names


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
    # sabotage → cfit (disabling instruments/controls leads to CFIT)
    # bk.add_required_by_pattern("sabotage", "cfit")
    bk.add_forbidden_by_pattern("cfit", "sabotage")
    # sabotage → fire (explosives/arson cause fire; fire doesn't cause sabotage)
    bk.add_forbidden_by_pattern("fire", "sabotage")
    # pilot_error → mechanical (improper operation damages equipment; mechanical failure doesn't cause pilot error)
    bk.add_forbidden_by_pattern("mechanical", "pilot_error")
    # high_lethality is the outcome — it cannot cause anything
    bk.add_forbidden_by_pattern("high_lethality", ".*")
    # mechanical → high_lethality (not the reverse)
    bk.add_forbidden_by_pattern("high_lethality", "mechanical_trouble")
    return bk


def pag_to_dowhy_graph(nodes: list[str], adj: np.ndarray) -> nx.DiGraph:
    """Convert a causal-learn PAG adjacency matrix to a networkx DiGraph for DoWhy.

    causal-learn convention: adj[i][j] is the mark AT node i on the edge between
    i and j.

      -1  tail  (—)
       1  arrowhead  (>)
       2  circle  (o)

    Conversion rules:
      mark_at_i=-1, mark_at_j=1  →  directed edge i → j
      mark_at_i=1,  mark_at_j=1  →  bidirected i ↔ j: latent node U with U→i and U→j
      All other combinations (circle marks, undirected) are skipped.
    """
    from causallearn.graph.Endpoint import Endpoint
    assert Endpoint.TAIL.value == -1 and Endpoint.ARROW.value == 1 and Endpoint.CIRCLE.value == 2, \
        "causal-learn Endpoint enum values have changed; review decoder."

    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)

    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            mi = adj[i][j]  # mark at node i
            mj = adj[j][i]  # mark at node j
            if mi == 0 and mj == 0:
                continue
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
            # all other mark combinations (circle marks) skipped

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
    """Use DoWhy to identify the backdoor adjustment set, then estimate via OLS.

    DoWhy is used only for identification — finding the valid set of confounders
    to condition on. The OLS coefficient on the treatment variable (holding the
    adjustment set constant) is returned. This is the standard partial regression
    coefficient, not an ATE estimand.

    Returns the OLS coefficient, or None if identification or estimation failed.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*assumed unobserved.*", category=UserWarning)
        model = CausalModel(data=data_df, treatment=treatment, outcome=outcome, graph=dag)
        identified = model.identify_effect(proceed_when_unidentifiable=True)

    backdoor_sets   = identified.backdoor_variables
    frontdoor_sets  = identified.frontdoor_variables
    iv_vars         = identified.instrumental_variables

    if backdoor_sets:
        first_key = next(iter(backdoor_sets))
        adjustment_vars = list(backdoor_sets[first_key])
        regressors = [treatment] + adjustment_vars
        try:
            X = sm.add_constant(data_df[regressors])
            y = data_df[outcome]
            result = sm.OLS(y, X).fit()
            coef = result.params[treatment]
            pval = result.pvalues[treatment]
            conf = result.conf_int().loc[treatment]
            print(f"  OLS coef({treatment}) = {coef:+.4f}  "
                  f"95% CI [{conf[0]:+.4f}, {conf[1]:+.4f}]  p={pval:.4f}  [backdoor]")
            print(f"  Adjustment set: {adjustment_vars}")
            return coef
        except Exception as e:
            print(f"  Estimation failed: {e}")
            return None
    elif frontdoor_sets:
        first_key = next(iter(frontdoor_sets))
        mediators = list(frontdoor_sets[first_key])
        print(f"  Identified via front-door (mediators: {mediators}) — not estimated via OLS.")
        return None
    elif iv_vars:
        first_key = next(iter(iv_vars))
        instruments = list(iv_vars[first_key])
        print(f"  Identified via IV (instruments: {instruments}) — not estimated via OLS.")
        return None
    else:
        print(f"  Not identified -- no valid backdoor, front-door, or IV set found.")
        return None


if __name__ == "__main__":
    OUTCOME = "high_lethality"
    TREATMENTS = ["subregion", "is_military",
                  "cfit", "collision", "fire", "fuel", "mechanical",
                  "pilot_error", "sabotage", "shot_down", "weather"]

    print("Loading and encoding data...")
    data_df, feature_cols, subregion_names = prepare_data(exclude_solo=True, max_samples=5000)
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
    directed  = [(u, v) for u, v in dag.edges() if not dag.nodes[u].get("observed") is False]
    latent    = [n for n, d in dag.nodes(data=True) if d.get("observed") is False]
    print(f"  Directed edges : {directed}")
    print(f"  Latent nodes   : {latent}\n")
    render_graph(dag, filename="output/dowhy/pag")

    total_effects = {}
    for treatment in TREATMENTS:
        print(f"{'='*60}")
        print(f"Treatment: {treatment} -> {OUTCOME}")
        print(f"{'='*60}")
        total_effects[treatment] = identify_and_estimate(dag, data_df, treatment, OUTCOME)
        print()

    print(f"{'='*60}")
    print("TOTAL CAUSAL IMPACT SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Cause':<15} {'OLS coef (high_lethality)':>26}")
    print(f"  {'-'*42}")
    for treatment, effect in sorted(total_effects.items(),
                                    key=lambda x: x[1] if x[1] is not None else 0,
                                    reverse=True):
        val = f"{effect:>+.4f}" if effect is not None else "   n/a"
        print(f"  {treatment:<15} {val:>22}")

    # Per-subregion effect on high_lethality
    # Adjustment set for subregion was empty (exogenous root), so each dummy
    # is regressed on the outcome with no additional controls.
    print(f"\n{'='*60}")
    print("SUBREGION CAUSAL IMPACT (per-region dummies)")
    print(f"{'='*60}")
    print(f"  {'Subregion':<30} {'OLS coef':>10}  {'95% CI':>22}  {'p-value':>8}")
    print(f"  {'-'*74}")
    subregion_effects = {}
    for name in subregion_names:
        dummy = (data_df["subregion"] == subregion_names.index(name)).astype(float)
        X = sm.add_constant(dummy)
        result = sm.OLS(data_df[OUTCOME], X).fit()
        coef = result.params.iloc[1]
        pval = result.pvalues.iloc[1]
        conf = result.conf_int().iloc[1]
        subregion_effects[name] = coef
        print(f"  {name:<30} {coef:>+.4f}  [{conf.iloc[0]:>+.4f}, {conf.iloc[1]:>+.4f}]  {pval:>8.4f}")

    print(f"\n  Ranked by effect:")
    for name, coef in sorted(subregion_effects.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name:<30} {coef:>+.4f}")