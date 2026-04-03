# Timothy Mascal
# April 1st, 2026
# Perform Data Labeling by finding key terms within an excel sheet
# Created with the aid of claude code.

# Concept
# Input: CSV File of All Aircraft Accidents in History
# Output: A pandas dataframe that can classify each accident by cause

import re
from collections import Counter

import pandas as pd


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

    # Keywords for each cause, checked in priority order.
    # The first cause whose keywords match wins.
    CAUSE_RULES: list[tuple[str, list[str]]] = [
        ("sabotage",      ["hijacked", "hijackers", "hijacker", "explosive", "bomb", "dynamite",
                           "sabotage", "suicide", "stolen"]),
        ("shot_down",     ["shot down", "shot by", "enemy fire", "anti-aircraft", "shot by fighter", "fighter fire"]),
        ("fire",          ["fire", "flames", "burned", "burnt", "exploded", "explosion"]),
        ("fuel",          ["fuel exhaustion", "fuel starvation", "ran out of fuel", "low fuel", "out of fuel"]),
        ("mechanical",    ["engine failure", "engine failed", "mechanical failure",
                           "engine", "engines", "failure", "failed", "power loss", "power failure",
                           "broken", "malfunctioning", "control problems",
                           "landing gear", "gear", "wing", "fatigue", "rudder", "elevator", "icing"]),
        ("weather",       ["weather", "fog", "rain", "visibility", "adverse",
                           "conditions", "vfr", "thunderstorm", "ice", "wind", "storm", "snow",
                           "turbulence", "windshear", "crosswind", "rainstorm", "snowstorm", "clouds"]),
        ("collision",     ["mid-air collision", "midair collision", "collided with another", "collided", "mid-air", "midair"]),
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
        # Drop undetermined if more specific causes were also matched
        if len(matches) > 1 and "undetermined" in matches:
            matches.remove("undetermined")
        return ", ".join(matches)

    def classify_series(self, series: pd.Series) -> pd.Series:
        return series.map(self.classify)

    def word_frequency(self, series: pd.Series, top_n: int = 100) -> list[tuple[str, int]]:
        counts = Counter()
        for text in series.dropna():
            tokens = re.findall(r'\b[a-z]+\b', text.lower())
            counts.update(t for t in tokens if t not in self.STOPWORDS)
        return counts.most_common(top_n)


if __name__ == "__main__":
    df = pd.read_csv("data/Airplane_Crashes_and_Fatalities_Since_1908.csv")
    print(f"Loaded {len(df)} rows.\n")

    classifier = AccidentCauseClassifier()

    labeled = pd.DataFrame({
        "Date":     df["Date"],
        "Time":     df["Time"],
        "Location": df["Location"],
        "Operator": df["Operator"],
        "Cause":    classifier.classify_series(df["Summary"]),
    })

    labeled.to_csv("data/labeled_accidents.csv", index=False)
    print("Saved to data/labeled_accidents.csv\n")

    print("Cause distribution:")
    print(labeled["Cause"].value_counts().to_string())
    print()
    print("Cause ratios:")
    print(labeled["Cause"].value_counts(normalize=True).map(lambda x: f"{x:.2%}").to_string())
    print()
    print("Sample:")
    print(labeled.head(10).to_string())