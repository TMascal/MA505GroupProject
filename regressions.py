# Timothy Mascal
# April 4, 2026
# Created with the Aid of Claude Code

import pandas as pd
import numpy as np
import statsmodels.api as sm
import graphviz
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


class RegressionModel:

    def __init__(self):
        self.df = pd.read_csv("data/labeled_accidents.csv")

    def _prepare_data(self, include_flight_type: bool = True) -> tuple[np.ndarray, list[str]]:
        """One-hot encode causes and optional features into a binary numpy array."""
        df = self.df.dropna(subset=["Cause"]).copy()

        all_causes: set[str] = set()
        for cell in df["Cause"]:
            for c in str(cell).split(","):
                all_causes.add(c.strip().lower())
        all_causes.discard("unknown")

        for cause in sorted(all_causes):
            df[cause] = df["Cause"].apply(
                lambda raw, _cause=cause: int(_cause in [tok.strip().lower() for tok in str(raw).split(",")])
            )

        feature_cols = sorted(all_causes)

        if include_flight_type:
            df["is_military"] = (df["FlightType"].str.lower().str.strip() == "military").astype(int)
            feature_cols = ["is_military"] + feature_cols

        if "FatalityRate" in df.columns:
            rate = pd.to_numeric(df["FatalityRate"], errors="coerce")
            df["high_lethality"] = (rate >= 0.5).astype("Int64")
            feature_cols = feature_cols + ["high_lethality"]

        data = df[feature_cols].dropna().to_numpy(dtype=int)
        return data, feature_cols

    def _build_background_knowledge(self) -> BackgroundKnowledge:
        """Background knowledge shared by both algorithms.

        Treats 'weather' and 'is_military' as root nodes — exogenous variables
        that cannot be caused by anything else in the graph.
        """
        bk = BackgroundKnowledge()
        bk.add_forbidden_by_pattern(".*", "weather")
        bk.add_forbidden_by_pattern(".*", "is_military")
        return bk

    def derive_cpdag(self, alpha: float = 0.05, include_flight_type: bool = True) -> CausalGraph:
        """Run the PC algorithm to produce a CPDAG (Markov Equivalence Class).

        Returns:
            CausalGraph whose .G.graph is the CPDAG adjacency matrix.
        """
        data, feature_cols = self._prepare_data(include_flight_type)
        bk = self._build_background_knowledge()
        cg = pc(data, alpha=alpha, indep_test="chisq", background_knowledge=bk,
                show_progress=False, node_names=feature_cols)
        return cg

    def derive_pag(self, alpha: float = 0.05, include_flight_type: bool = True) -> GeneralGraph:
        """Run the FCI algorithm to produce a PAG (Partial Ancestral Graph).

        Unlike the CPDAG, the PAG also represents possible latent confounders
        via bidirected edges (i ↔ j) and circle marks (o).

        Returns:
            GeneralGraph whose .graph is the PAG adjacency matrix.
        """
        data, feature_cols = self._prepare_data(include_flight_type)
        bk = self._build_background_knowledge()
        graph, _ = fci(data, independence_test_method="chisq", alpha=alpha,
                       background_knowledge=bk, show_progress=False, node_names=feature_cols)
        return graph

    def render_mec(self, graph: GeneralGraph | CausalGraph, filename: str = "mec") -> graphviz.Digraph:
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

    def build_scm(self, edges: str, filename: str = "scm") -> graphviz.Digraph:
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
    cpdag = model.derive_cpdag(alpha=0.05)
    node_names = [node.get_name() for node in cpdag.G.get_nodes()]
    print("Node labels:", node_names)
    print("CPDAG adjacency matrix:\n", cpdag.G.graph)
    model.render_mec(cpdag, filename="cpdag")

    # FCI — PAG (Partial Ancestral Graph, allows latent confounders)
    print("\nDeriving PAG via FCI algorithm...")
    pag = model.derive_pag(alpha=0.05)
    node_names = [node.get_name() for node in pag.get_nodes()]
    print("Node labels:", node_names)
    print("PAG adjacency matrix:\n", pag.graph)
    model.render_mec(pag, filename="pag")
