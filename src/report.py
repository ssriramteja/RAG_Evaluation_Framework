import json
import pandas as pd
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent.parent / "results"


def generate_report():
    """
    Loads eval_results.csv and produces a clean
    markdown + JSON report for GitHub README.
    """
    csv_path = RESULTS_DIR / "eval_results.csv"
    if not csv_path.exists():
        print("❌ No eval_results.csv found. Run evaluator.py first.")
        return

    df = pd.read_csv(csv_path)

    ragas_cols = [
        "faithfulness", "answer_relevancy",
        "context_recall", "context_precision", "answer_correctness"
    ]
    custom_cols = [
        "token_overlap_vs_gt", "fuzzy_similarity_vs_gt",
        "context_hit_rate", "answer_length_score", "grounding_proxy"
    ]

    report = {
        "generated_at"   : datetime.now().isoformat(),
        "num_questions"  : len(df),
        "ragas_scores"   : {
            col: round(df[col].mean(), 4)
            for col in ragas_cols if col in df.columns
        },
        "custom_scores"  : {
            col: round(df[col].mean(), 4)
            for col in custom_cols if col in df.columns
        },
        "per_question"   : df[["question"] + ragas_cols].round(4).to_dict(orient="records")
    }

    report_path = RESULTS_DIR / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Report saved to {report_path}")

    # Print markdown table for README
    print("\n## 📊 Evaluation Results (copy into README.md)\n")
    print("| Metric | Score |")
    print("|---|---|")
    for metric, score in {**report["ragas_scores"], **report["custom_scores"]}.items():
        emoji = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
        print(f"| {metric} | {score} {emoji} |")


if __name__ == "__main__":
    generate_report()
