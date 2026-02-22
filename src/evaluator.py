import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness
)
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from metrics import compute_all_custom_metrics

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RESULTS_DIR  = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class RAGEvaluator:
    """
    Automated RAG evaluation engine using RAGAS and custom lexical metrics.

    RAGAS metrics are selected to provide a multi-dimensional view of 
    performance:
    - Faithfulness & Answer Relevancy: Measure generation quality.
    - Context Recall & Precision: Measure retrieval effectiveness.
    """

    def __init__(self):
        print("Initializing RAGAS evaluator (Groq Llama 3.3 + MiniLM)...")

        # LLM Judge for RAGAS semantic evaluation
        self.llm = ChatGroq(
            model       = "llama-3.3-70b-versatile",
            api_key     = GROQ_API_KEY,
            temperature = 0.0
        )

        # Embedding model for semantic similarity calculations in RAGAS
        self.embeddings = HuggingFaceEmbeddings(
            model_name    = "sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True}
        )

        print("RAGAS evaluator initialization complete.\n")


    def run_ragas(self, rag_results: list[dict]) -> pd.DataFrame:
        """
        Run RAGAS 5-metric evaluation over all RAG outputs.
        Returns a DataFrame with one row per question + all scores.
        """

        # RAGAS expects a HuggingFace Dataset with these exact column names
        dataset = Dataset.from_dict({
            "question"  : [r["question"]          for r in rag_results],
            "answer"    : [r["answer"]             for r in rag_results],
            "contexts"  : [r["contexts"]           for r in rag_results],
            "ground_truth": [r["ground_truth"]     for r in rag_results],
        })

        print("Executing RAGAS semantic evaluation (LLM-as-judge)...")
        print("Note: This process is computationally intensive and may take several minutes.\n")

        result = evaluate(
            dataset = dataset,
            metrics = [
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
                answer_correctness
            ],
            llm        = self.llm,
            embeddings = self.embeddings,
            raise_exceptions = False
        )

        ragas_df = result.to_pandas()
        print("✅ RAGAS evaluation complete!")
        return ragas_df


    def run_custom_metrics(self, rag_results: list[dict]) -> pd.DataFrame:
        """
        Run lightweight custom metrics (no LLM needed).
        """
        rows = []
        for item in rag_results:
            custom = compute_all_custom_metrics(item)
            custom["question"] = item["question"]
            rows.append(custom)
        return pd.DataFrame(rows)


    def evaluate_all(self, rag_results: list[dict]) -> pd.DataFrame:
        """
        Full evaluation: RAGAS + custom metrics, merged into one DataFrame.
        """
        # RAGAS metrics
        ragas_df  = self.run_ragas(rag_results)

        # Custom metrics
        custom_df = self.run_custom_metrics(rag_results)

        # Merge on question
        merged_df = ragas_df.merge(custom_df, on="question", how="left")

        # Save to CSV
        csv_path = RESULTS_DIR / "eval_results.csv"
        merged_df.to_csv(csv_path, index=False)
        print(f"\n✅ Full results saved to {csv_path}")

        # Save summary stats
        numeric_cols = merged_df.select_dtypes(include="number").columns
        summary = merged_df[numeric_cols].describe().round(4)
        summary_path = RESULTS_DIR / "eval_summary.csv"
        summary.to_csv(summary_path)
        print(f"✅ Summary stats saved to {summary_path}")

        return merged_df


    def print_summary(self, df: pd.DataFrame):
        """Print a readable summary table to terminal."""
        print("\n" + "=" * 60)
        print("  RAG EVALUATION SUMMARY")
        print("=" * 60)

        metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
            "answer_correctness",
            "token_overlap_vs_gt",
            "grounding_proxy",
            "context_hit_rate"
        ]

        for col in metrics:
            if col in df.columns:
                score = df[col].mean()
                bar   = "█" * int(score * 20)
                print(f"  {col:<30} {score:.3f}  {bar}")

        print("=" * 60)


if __name__ == "__main__":
    # Load processed RAG outputs for evaluation
    outputs_path = RESULTS_DIR / "rag_outputs.json"
    if not outputs_path.exists():
        print("Error: RAG output file not found. Ensure rag_runner.py has been executed.")
        exit(1)

    with open(outputs_path) as f:
        rag_results = json.load(f)

    evaluator = RAGEvaluator()
    df        = evaluator.evaluate_all(rag_results)
    evaluator.print_summary(df)
