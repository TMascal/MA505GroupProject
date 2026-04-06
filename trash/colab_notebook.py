# File that combines all of the workspace into one file to be moved to collab.

# ============================================================
# MA505 Group Project — Single Colab Notebook
# Each "# %% CELL N" comment is a new cell in Colab.
# Run cells in order top to bottom.
# ============================================================


# %% CELL 1 — Install dependencies
# --------------------------------------------------------
# !pip install causal-learn graphviz statsmodels


# %% CELL 2 — Imports
# --------------------------------------------------------
import re
import os
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api as sm
import graphviz

from IPython.display import Image, display

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


# %% CELL 3 — Mount Google Drive and set paths
# --------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

# Change this to the folder in your Drive where the CSVs live
DATA_DIR = "/content/drive/MyDrive/MA505/"
os.makedirs(DATA_DIR + "output", exist_ok=True)


# %% CELL 4 — AccidentCauseClassifier class
# --------------------------------------------------------
class AccidentCauseClassifier:
    """Classifies accident summaries into cause categories using keyword matching."""

    STOPWORDS = {
        "the", "a", "an", "and", "of", "in", "was", "to", "by", "at", "on",
        "it", "its", "that", "this", "with", "from", "for", "as", "or", "be",
        "but", "not", "into", "after", "when", "had", "were", "is", "are",
        "has", "have", "he", "she", "they", "who", "which", "while", "all",
        "also", "about", "one", "two", "three", "four", "five", "been", "no",
        "their", "during", "near", "over", "under", "than", "more", "other",
        "both", "then", "there", "would", "could", "upon", "out", "up", "down",
        "an", "due", "may", "caused", "shortly", "flight", "aircraft", "plane",
        "airplane", "crew",
    }

    CAUSE_RULES: list[tuple[str, list[str]]] = [
        ("sabotage",      ["hijacked", "hijackers", "hijacker", "explosive", "bomb", "dynamite",
                           "sabotage", "suicide", "stolen", "grenade"]),
        ("shot_down",     ["shot down", "shot by", "enemy fire", "anti-aircraft", "shot by fighter", "fighter fire", "gunfire", "bullet"]),
        ("fire",          ["fire", "flames", "burned", "burnt", "exploded", "explosion"]),
        ("fuel",          ["fuel exhaustion", "fuel starvation", "ran out of fuel", "low fuel", "out of fuel"]),
        ("mechanical",    ["engine failure", "engine failed", "mechanical failure",
                           "engine", "engines", "failure", "failed", "power loss", "power failure",
                           "broken", "malfunctioning", "control problems",
                           "landing gear", "gear", "wing", "fatigue", "rudder", "elevator", "icing"]),
        ("weather",       ["weather", "fog", "rain", "visibility", "adverse",
                           "conditions", "vfr", "thunderstorm", "ice", "wind", "storm", "snow",
                           "turbulence", "windshear", "crosswind", "rainstorm", "snowstorm", "clouds"]),
        ("collision",     ["mid-air collision", "midair collision", "collided with another aircraft", "collided with another airplane", "collided"]),
        ("cfit",          ["mountain", "mountains", "mountainside", "mount", "mountainous", "terrain",
                           "trees", "struck", "hillside", "hill", "controlled flight",
                           "ocean", "lake", "river", "sea", "wooded", "jungle", "mt", "ravine",
                           "ridge", "peak", "volcano", "swamp", "channel", "foothills"]),
        ("pilot_error",   ["pilot error", "tailspin", "tail spin", "lost control", "nose dive",
                           "nosedive", "overran", "overloaded", "uncontrolled", "unable", "premature",
                           "procedure", "spin", "stall", "improperly", "inadequate", "alcohol",
                           "unqualified", "sharp turn", "turned over", "error", "improper", "procedures",
                           "stalled", "maintain", "failed to", "neglected"]),
        ("undetermined",  ["cause unknown", "cause undetermined", "undetermined", "reasons unknown",
                           "disappeared", "never found", "cause not determined", "missing", "wreckage",
                           "disintegrated"]),
    ]

    def __init__(self):
        self._patterns: list[tuple[str, re.Pattern]] = [
            (cause, re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b', re.IGNORECASE))
            for cause, keywords in self.CAUSE_RULES
        ]

    def classify(self, summary: str) -> str:
        if not isinstance(summary, str) or not summary.strip():
            return "unknown"
        matches = [cause for cause, pattern in self._patterns if pattern.search(summary)]
        if not matches:
            return "unknown"
        if len(matches) > 1 and "undetermined" in matches:
            matches.remove("undetermined")
        return ", ".join(matches)

    def classify_series(self, series: pd.Series) -> pd.Series:
        return series.map(self.classify)

    _MILITARY_PATTERN = re.compile(
        r'\bmilitary\b|air force|army|navy|marine corps|coast guard|royal air force|'
        r'royal australian|royal canadian|royal new zealand|luftwaffe|'
        r'air corps|national guard|air command',
        re.IGNORECASE,
    )
    _CIVILIAN_PATTERN = re.compile(r'\bprivate\b|\bair taxi\b', re.IGNORECASE)

    def classify_operator(self, operator: str) -> str:
        if not isinstance(operator, str) or not operator.strip():
            return "unknown"
        if self._MILITARY_PATTERN.search(operator):
            return "military"
        if self._CIVILIAN_PATTERN.search(operator):
            return "civilian"
        return "commercial"

    def classify_operator_series(self, series: pd.Series) -> pd.Series:
        return series.map(self.classify_operator)

    def word_frequency(self, series: pd.Series, top_n: int = 100) -> list[tuple[str, int]]:
        counts = Counter()
        for text in series.dropna():
            tokens = re.findall(r'\b[a-z]+\b', text.lower())
            counts.update(t for t in tokens if t not in self.STOPWORDS)
        return counts.most_common(top_n)


# %% CELL 5 — Run classifier and save labeled_accidents.csv
# --------------------------------------------------------
df = pd.read_csv(DATA_DIR + "Airplane_Crashes_and_Fatalities_Since_1908.csv")
print(f"Loaded {len(df)} rows.\n")

classifier = AccidentCauseClassifier()

aboard       = pd.to_numeric(df["Aboard"],      errors="coerce")
fatalities   = pd.to_numeric(df["Fatalities"],  errors="coerce")
ground       = pd.to_numeric(df["Ground"],      errors="coerce").fillna(0)
fatality_rate = (fatalities + ground) / aboard

labeled = pd.DataFrame({
    "Date":         df["Date"],
    "Time":         df["Time"],
    "Location":     df["Location"],
    "Operator":     df["Operator"],
    "FlightType":   classifier.classify_operator_series(df["Operator"]),
    "Cause":        classifier.classify_series(df["Summary"]),
    "FatalityRate": fatality_rate,
})

labeled.to_csv(DATA_DIR + "labeled_accidents.csv", index=False)
print("Saved to labeled_accidents.csv\n")

print("Cause distribution:")
print(labeled["Cause"].value_counts().to_string())
print()
print("Cause ratios:")
print(labeled["Cause"].value_counts(normalize=True).map(lambda x: f"{x:.2%}").to_string())
print()
print("Sample:")
print(labeled.head(10).to_string())


# %% CELL 6 — Classifier evaluation helpers
# --------------------------------------------------------
CATEGORIES = [
    "sabotage", "shot_down", "fire", "fuel", "weather",
    "collision", "cfit", "mechanical", "pilot_error",
    "undetermined", "unknown",
]

def parse_labels(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return set()
    return {label.strip() for label in str(cell).split(",")}




# %% CELL 8 — RegressionModel class
# --------------------------------------------------------
class RegressionModel:

    def __init__(self):
        self.df = pd.read_csv(DATA_DIR + "labeled_accidents.csv")

    def _prepare_data(self, include_flight_type: bool = True) -> tuple[np.ndarray, list[str]]:
        df = self.df.dropna(subset=["Cause"]).copy()

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

        if "FatalityRate" in df.columns:
            rate = pd.to_numeric(df["FatalityRate"], errors="coerce")
            df["high_lethality"] = (rate >= 0.5).astype("Int64")
            feature_cols = feature_cols + ["high_lethality"]

        data = df[feature_cols].dropna().to_numpy(dtype=int)
        return data, feature_cols

    def _build_background_knowledge(self) -> BackgroundKnowledge:
        bk = BackgroundKnowledge()
        bk.add_forbidden_by_pattern(".*", "weather")
        bk.add_forbidden_by_pattern(".*", "is_military")
        bk.add_forbidden_by_pattern("high_lethality", ".*")
        return bk

    def linear_regression(self, x_vars: list[str], y_var: str, include_flight_type: bool = True):
        data, feature_cols = self._prepare_data(include_flight_type)
        df = pd.DataFrame(data, columns=feature_cols)

        if "FatalityRate" in self.df.columns:
            rate = pd.to_numeric(self.df["FatalityRate"], errors="coerce")
            df["FatalityRate"] = rate.reindex(df.index)

        available = list(df.columns)
        for var in x_vars + [y_var]:
            if var not in df.columns:
                raise ValueError(f"'{var}' not found. Available: {available}")

        subset = df[x_vars + [y_var]].dropna()
        X = sm.add_constant(subset[x_vars].to_numpy())
        y = subset[y_var].to_numpy()

        fitted = sm.OLS(y, X).fit()
        print(fitted.summary(xname=["const"] + x_vars, yname=y_var))
        return fitted

    def derive_cpdag(self, alpha: float = 0.05, include_flight_type: bool = True) -> CausalGraph:
        data, feature_cols = self._prepare_data(include_flight_type)
        bk = self._build_background_knowledge()
        cg = pc(data, alpha=alpha, indep_test="chisq", background_knowledge=bk,
                show_progress=False, node_names=feature_cols)
        return cg

    def derive_pag(self, alpha: float = 0.05, include_flight_type: bool = True) -> GeneralGraph:
        data, feature_cols = self._prepare_data(include_flight_type)
        bk = self._build_background_knowledge()
        graph, _ = fci(data, independence_test_method="chisq", alpha=alpha,
                       background_knowledge=bk, show_progress=False, node_names=feature_cols)
        return graph

    def render_mec(self, graph: GeneralGraph | CausalGraph, filename: str = "mec") -> graphviz.Digraph:
        _MARK = {-1: "none", 1: "normal", 2: "odot"}
        dot = graphviz.Digraph(name="PAG", graph_attr={"rankdir": "LR"})

        g = graph.G if isinstance(graph, CausalGraph) else graph
        nodes = [node.get_name() for node in g.get_nodes()]
        adj = g.graph
        n = len(nodes)

        for name in nodes:
            dot.node(name)

        for i in range(n):
            for j in range(i + 1, n):
                mark_at_i = adj[i][j]
                mark_at_j = adj[j][i]
                if mark_at_i == 0 and mark_at_j == 0:
                    continue
                dot.edge(
                    nodes[i], nodes[j],
                    dir="both",
                    arrowtail=_MARK.get(mark_at_i, "none"),
                    arrowhead=_MARK.get(mark_at_j, "none"),
                )

        out_path = DATA_DIR + "output/" + filename
        dot.render(out_path, format="png", cleanup=True)
        display(Image(out_path + ".png"))
        return dot

    def build_scm(self, edges: str, filename: str = "scm") -> graphviz.Digraph:
        dot = graphviz.Digraph(name="SCM", graph_attr={"rankdir": "LR"})
        for token in edges.split(","):
            token = token.strip()
            if ">" not in token:
                continue
            source, target = token.split(">", 1)
            dot.edge(source.strip(), target.strip())

        out_path = DATA_DIR + "output/" + filename
        dot.render(out_path, format="png", cleanup=True)
        display(Image(out_path + ".png"))
        return dot


# %% CELL 9 — Run regression models
# --------------------------------------------------------
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
