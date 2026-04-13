# Timothy Mascal
# April 4, 2026
# Created with the Aid of Claude Code

import pandas as pd
import numpy as np
import statsmodels.api as sm
import graphviz
import networkx as nx
from dowhy import CausalModel
from causallearn.search.ConstraintBased.PC import pc
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


class RegressionModel:

    def __init__(self):
        self.df = pd.read_csv("data/labeled_accidents.csv")

    def _prepare_data(self, include_flight_type: bool = True, exclude_solo: bool = False) -> tuple[np.ndarray, list[str]]:
        """Compact encoding for causal graph discovery (PC / FCI).

        Subregion is a single integer node (0–21) so the graph stays readable.
        FlightType is a single binary is_military node.
        Causes are binary presence indicators.
        """
        df = self.df.dropna(subset=["Cause"]).copy()
        df = df[~df["Cause"].str.lower().str.strip().isin(["unknown", "undetermined"])].copy()
        if exclude_solo:
            df = df[~((df["Aboard"] == 1) & (df["Fatalities"] == 1))].copy()

        all_causes: set[str] = set()
        for cell in df["Cause"]:
            for c in str(cell).split(","):
                all_causes.add(c.strip().lower())
        all_causes.discard("unknown")
        all_causes.discard("undetermined")

        for cause in sorted(all_causes):
            df[cause] = df["Cause"].apply(
                lambda raw, _cause=cause: int(_cause in [tok.strip().lower() for tok in str(raw).split(",")])
            )

        feature_cols = sorted(all_causes)

        if include_flight_type:
            df["is_military"] = (df["FlightType"].str.lower().str.strip() == "military").astype(int)
            feature_cols = ["is_military"] + feature_cols

        if "Subregion" in df.columns:
            # Single integer-coded node for UN M49 subregion (0–21).
            # The chi-square test handles multi-category variables natively, avoiding
            # the mutual-exclusivity corruption that one-hot encoding would introduce.
            # Limitation: this node can only reveal *whether* subregion associates with
            # another variable — not *which* subregion drives the association.
            # Further targeted analysis (e.g. stratified regression) is needed for that.
            known = df["Subregion"] != "unknown"
            df = df[known].copy()
            codes = {s: i for i, s in enumerate(sorted(df["Subregion"].unique()))}
            df["subregion"] = df["Subregion"].map(codes)
            feature_cols = ["subregion"] + feature_cols

        if "FatalityRate" in df.columns:
            rate = pd.to_numeric(df["FatalityRate"], errors="coerce")
            df["high_lethality"] = (rate > 0.5).astype(float)
            df.loc[rate.isna(), "high_lethality"] = np.nan
            feature_cols = feature_cols + ["high_lethality"]

        data = df[feature_cols].dropna().to_numpy(dtype=float)
        return data, feature_cols

    def _prepare_regression_data(self, exclude_solo: bool = False) -> pd.DataFrame:
        """Expanded one-hot encoding used exclusively for linear regression.

        Produces a fully expanded binary feature matrix suitable for estimating
        causal impacts via OLS.  Each coefficient can be interpreted as an
        Average Treatment Effect (ATE) — the expected change in FatalityRate
        when that indicator goes from 0 to 1, holding all others constant.

        Column layout (34 predictors + FatalityRate outcome):
          rgn_*         — 22 binary columns, one per UN M49 subregion
          ft_military   — 1 if military
          ft_commercial — 1 if commercial airline
          ft_civilian   — 1 if private / civilian
          <cause>       — 9 binary cause-present columns
          FatalityRate  — continuous outcome (normalised fatalities)

        Note: because subregion and flight-type columns are mutually exclusive
        within their group, including all dummies without dropping one reference
        category creates perfect multicollinearity with the OLS intercept.
        sm.add_constant is therefore NOT used here — the intercept is absorbed
        by the full set of dummies (each group's dummies sum to 1).
        """
        rate = pd.to_numeric(self.df["FatalityRate"], errors="coerce")
        df = self.df[rate.notna() & np.isfinite(rate)].copy()
        df = df.dropna(subset=["Cause"])
        df = df[~df["Cause"].str.lower().str.strip().isin(["unknown", "undetermined"])].copy()
        if exclude_solo:
            df = df[~((df["Aboard"] == 1) & (df["Fatalities"] == 1))].copy()
        df = df[df["Subregion"] != "unknown"].copy()
        cols: list[str] = []

        # Subregion one-hot (22 columns)
        for sr in sorted(df["Subregion"].unique()):
            col = "rgn_" + sr.replace(" ", "_").replace("-", "_")
            df[col] = (df["Subregion"] == sr).astype(int)
            cols.append(col)

        # Flight type one-hot (3 columns)
        ft = df["FlightType"].str.lower().str.strip()
        df["ft_military"]   = (ft == "military").astype(int)
        df["ft_commercial"] = (ft == "commercial").astype(int)
        df["ft_civilian"]   = (ft == "civilian").astype(int)
        cols += ["ft_military", "ft_commercial", "ft_civilian"]

        # Cause binary indicators (9 columns)
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
        cols += sorted(all_causes)

        df["FatalityRate"] = pd.to_numeric(df["FatalityRate"], errors="coerce")
        return df[cols + ["FatalityRate"]].dropna()

    def _build_background_knowledge(self) -> BackgroundKnowledge:
        """Background knowledge shared by both algorithms.

        Treats 'weather' and 'is_military' as root nodes — exogenous variables
        that cannot be caused by anything else in the graph.
        """
        bk = BackgroundKnowledge()
        # Weather is exogenous except for subregion — the region an aircraft
        # flies in predicts local weather patterns, not the reverse.
        bk.add_forbidden_by_pattern("(?!subregion$).*", "weather")
        bk.add_forbidden_by_pattern("weather", "subregion")
        bk.add_forbidden_by_pattern("cfit", "subregion")
        bk.add_required_by_pattern("pilot_error", "cfit")
        bk.add_forbidden_by_pattern("cfit", "pilot_error")
        bk.add_forbidden_by_pattern(".*", "is_military")
        bk.add_forbidden_by_pattern("high_lethality", ".*")
        # Subregion causes shot_down (not the reverse)
        bk.add_forbidden_by_pattern("shot_down", "subregion")
        # Mechanical trouble causes high fatality (not the reverse)
        bk.add_forbidden_by_pattern("high_lethality", "mechanical_trouble")
        # Subregion is parent of pilot_error (not the reverse)
        bk.add_forbidden_by_pattern("pilot_error", "subregion")
        return bk

    def ols_fatality(self, x_vars: list[str] | None = None, exclude_solo: bool = False):
        """OLS regression of FatalityRate on selected columns from _prepare_regression_data.

        Args:
            x_vars: Column names to use as predictors. Pass None to use all available
                    predictor columns. Available columns include rgn_*, ft_*, and cause
                    indicators. Run model._prepare_regression_data().columns to inspect.

        Returns:
            The fitted statsmodels OLS RegressionResults object.
        """
        df = self._prepare_regression_data(exclude_solo=exclude_solo)
        predictor_cols = [c for c in df.columns if c != "FatalityRate"]
        predictors = x_vars if x_vars is not None else predictor_cols
        missing = [v for v in predictors if v not in df.columns]
        if missing:
            raise ValueError(f"Predictors not found: {missing}. Available: {predictor_cols}")

        subset = df[predictors + ["FatalityRate"]].dropna()
        X = subset[predictors].to_numpy()
        y = subset["FatalityRate"].to_numpy()

        fitted = sm.OLS(y, X).fit()
        print(fitted.summary(xname=predictors, yname="FatalityRate"))
        return fitted

    def derive_cpdag(self, alpha: float = 0.05, include_flight_type: bool = True, exclude_solo: bool = False) -> CausalGraph:
        """Run the PC algorithm to produce a CPDAG (Markov Equivalence Class).

        Returns:
            CausalGraph whose .G.graph is the CPDAG adjacency matrix.
        """
        data, feature_cols = self._prepare_data(include_flight_type, exclude_solo=exclude_solo)
        bk = self._build_background_knowledge()
        cg = pc(data, alpha=alpha, indep_test="chisq", background_knowledge=bk,
                show_progress=False, node_names=feature_cols)
        return cg

    def render_mec(self, graph: GeneralGraph | CausalGraph, filename: str = "output/mec") -> graphviz.Digraph:
        """Render a PAG (from FCI) to a PNG file.

        PAG adjacency matrix encoding — graph[i][j] is the mark AT node i:
          -1  tail  (—)
           1  arrowhead  (>)
           2  circle  (o)

        Edge type is determined by the marks at both endpoints:
          tail — arrowhead  →  directed  (i → j)
          arrowhead — arrowhead  →  bidirected  (i ↔ j)  latent confounder
          tail — tail  →  undirected  (i — j)
          circle — arrowhead  →  i o→ j  (uncertain tail/arrowhead)
          circle — circle  →  i o-o j  (fully uncertain)
          circle — tail  →  i o— j
        """
        _MARK = {-1: "none", 1: "normal", 2: "odot"}

        dot = graphviz.Digraph(name="PAG", graph_attr={"rankdir": "LR"})

        # CausalGraph (PC) wraps the GeneralGraph in a .G attribute; FCI returns one directly
        g = graph.G if isinstance(graph, CausalGraph) else graph
        nodes = [node.get_name() for node in g.get_nodes()]
        adj = g.graph
        n = len(nodes)

        for name in nodes:
            dot.node(name)

        for i in range(n):
            for j in range(i + 1, n):
                mark_at_i = adj[i][j]   # mark on the i-side of edge i–j
                mark_at_j = adj[j][i]   # mark on the j-side of edge i–j
                if mark_at_i == 0 and mark_at_j == 0:
                    continue            # no edge
                dot.edge(
                    nodes[i], nodes[j],
                    dir="both",
                    arrowtail=_MARK.get(mark_at_i, "none"),
                    arrowhead=_MARK.get(mark_at_j, "none"),
                )

        dot.render(filename, format="png", cleanup=True)
        return dot

    def backdoor_paths(self, graph: GeneralGraph | CausalGraph, treatment: str, outcome: str,
                       exclude_solo: bool = False) -> None:
        """Use DoWhy to identify backdoor paths and valid adjustment sets.

        Converts the PC-learned DAG to a networkx DiGraph, builds a DoWhy
        CausalModel aligned to the same node names, and prints all identified
        estimands including backdoor adjustment sets.

        Args:
            graph:        CausalGraph (from PC) or GeneralGraph (from FCI).
            treatment:    Name of the treatment node.
            outcome:      Name of the outcome node.
            exclude_solo: Whether to exclude single-aboard single-fatality rows.
        """
        g = graph.G if isinstance(graph, CausalGraph) else graph
        nodes = [node.get_name() for node in g.get_nodes()]
        adj = g.graph

        # Build DiGraph — only directed edges (tail at src, arrowhead at dst)
        dag = nx.DiGraph()
        dag.add_nodes_from(nodes)
        for i, src in enumerate(nodes):
            for j, dst in enumerate(nodes):
                if adj[i][j] == -1 and adj[j][i] == 1:
                    dag.add_edge(src, dst)

        # Build aligned DataFrame using same encoding as _prepare_data
        data_arr, feature_cols = self._prepare_data(exclude_solo=exclude_solo)
        data_df = pd.DataFrame(data_arr, columns=feature_cols)

        causal_model = CausalModel(
            data=data_df,
            treatment=treatment,
            outcome=outcome,
            graph=dag,
        )
        identified = causal_model.identify_effect(proceed_when_unidentifiable=True)
        print(identified)

    def build_scm(self, edges: str, filename: str = "output/scm") -> graphviz.Digraph:
        """Generate a SCM diagram from a CSV edge string like 'A>B, B>C, C>A'.

        Renders the graph to a file and returns the Digraph object.
        """
        dot = graphviz.Digraph(name="SCM", graph_attr={"rankdir": "LR"})

        for token in edges.split(","):
            token = token.strip()
            if ">" not in token:
                continue
            source, target = token.split(">", 1)
            dot.edge(source.strip(), target.strip())

        dot.render(filename, format="png", cleanup=True)
        return dot



if __name__ == "__main__":
    print("Beginning Regression Model")

    model = RegressionModel()

    # Naive hand-specified SCM
    model.build_scm(edges="Mechanical Trouble>Total Fatalities, Pilot Error>Total Fatalities, Weather>Total Fatalities")

    # PC — CPDAG (Markov Equivalence Class)
    print("\nDeriving CPDAG via PC algorithm...")
    cpdag = model.derive_cpdag(alpha=0.05, exclude_solo=True)
    node_names = [node.get_name() for node in cpdag.G.get_nodes()]
    print("Node labels:", node_names)
    print("CPDAG adjacency matrix:\n", cpdag.G.graph)
    model.render_mec(cpdag, filename="output/cpdag")

    # Backdoor paths — DoWhy identification for cfit → high_lethality
    print("\nBackdoor identification: cfit → high_lethality")
    model.backdoor_paths(cpdag, treatment="cfit", outcome="high_lethality", exclude_solo=True)

    # OLS — effect of fuel-related accidents on fatality rate
    # Possible x_vars options:
    #   Subregions: rgn_Australia_and_New_Zealand, rgn_Caribbean, rgn_Central_America,
    #               rgn_Central_Asia, rgn_Eastern_Africa, rgn_Eastern_Asia, rgn_Eastern_Europe,
    #               rgn_Melanesia, rgn_Micronesia, rgn_Middle_Africa, rgn_Northern_Africa,
    #               rgn_Northern_America, rgn_Northern_Europe, rgn_Polynesia, rgn_South_America,
    #               rgn_South_eastern_Asia, rgn_Southern_Africa, rgn_Southern_Asia,
    #               rgn_Southern_Europe, rgn_Western_Africa, rgn_Western_Asia, rgn_Western_Europe
    #   Flight type: ft_military, ft_commercial, ft_civilian
    #   Causes:      cfit, collision, fire, fuel, mechanical, pilot_error, sabotage, shot_down, weather
    print("\nOLS: Fuel cause → FatalityRate")
    model.ols_fatality(x_vars=["fuel"], exclude_solo=True)

    print("\nOLS: Sabotage cause → FatalityRate")
    model.ols_fatality(x_vars=["sabotage"], exclude_solo=True)

    print("\nOLS: CFIT cause → FatalityRate")
    model.ols_fatality(x_vars=["cfit"], exclude_solo=True)

    print("\nOLS: All causes → FatalityRate")
    model.ols_fatality(x_vars=["cfit", "collision", "fire", "fuel", "mechanical", "pilot_error", "sabotage", "shot_down", "weather"], exclude_solo=True)

    print("\nOLS: Weather + all subregions → FatalityRate")
    subregions = [c for c in model._prepare_regression_data(exclude_solo=True).columns if c.startswith("rgn_")]
    model.ols_fatality(x_vars=["weather"] + subregions, exclude_solo=True)