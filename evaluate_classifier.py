import pandas as pd
from collections import defaultdict

CATEGORIES = [
    "sabotage", "shot_down", "fire", "fuel", "weather",
    "collision", "cfit", "mechanical", "pilot_error",
    "undetermined", "unknown",
]


def parse_labels(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return set()
    return {label.strip() for label in str(cell).split(",")}


def main():
    df = pd.read_csv("data/manual_review_sample_accidents.csv")

    algo_col = df.columns[-2]   # second to last: algorithm output
    human_col = df.columns[-1]  # last: human review labels

    print(f"Algorithm column : '{algo_col}'")
    print(f"Human review column: '{human_col}'")
    print(f"Total rows: {len(df)}\n")

    counts = {cat: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for cat in CATEGORIES}

    for _, row in df.iterrows():
        algo_labels = parse_labels(row[algo_col])
        human_labels = parse_labels(row[human_col])

        for cat in CATEGORIES:
            predicted = cat in algo_labels
            actual = cat in human_labels

            if predicted and actual:
                counts[cat]["TP"] += 1
            elif not predicted and not actual:
                counts[cat]["TN"] += 1
            elif predicted and not actual:
                counts[cat]["FP"] += 1
            elif not predicted and actual:
                counts[cat]["FN"] += 1

    # Print results
    header = f"{'Category':<15} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}  {'Precision':>10} {'Recall':>8} {'F1':>6}"
    print(header)
    print("-" * len(header))

    for cat in CATEGORIES:
        tp = counts[cat]["TP"]
        tn = counts[cat]["TN"]
        fp = counts[cat]["FP"]
        fn = counts[cat]["FN"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else float("nan"))

        prec_str = f"{precision:.3f}" if precision == precision else "  N/A"
        rec_str  = f"{recall:.3f}"    if recall == recall    else "  N/A"
        f1_str   = f"{f1:.3f}"        if f1 == f1            else "  N/A"

        print(f"{cat:<15} {tp:>4} {tn:>4} {fp:>4} {fn:>4}  {prec_str:>10} {rec_str:>8} {f1_str:>6}")


if __name__ == "__main__":
    main()