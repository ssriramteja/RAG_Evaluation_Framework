import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import itertools

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

# Collect all available Groq API keys from environment
API_KEYS = [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4")
]
# Filter out empty/None keys
API_KEYS = [k for k in API_KEYS if k]

RESULTS_DIR  = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class GroqMultiKeyLLM(ChatGroq):
    """
    Custom wrapper for ChatGroq that rotates through multiple API keys
    to bypass rate limits and ensures n=1 compatibility with RAGAS.
    """
    def __init__(self, *args, keys=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Bypassing Pydantic validation for internal state
        object.__setattr__(self, "key_cycle", itertools.cycle(keys) if keys else None)

    def _rotate_key(self):
        if hasattr(self, "key_cycle") and self.key_cycle:
            new_key = next(self.key_cycle)
            # Use object.__setattr__ for api_key as well to be safe with Pydantic
            object.__setattr__(self, "api_key", new_key)

    def generate(self, prompts, stop=None, callbacks=None, **kwargs):
        self._rotate_key()
        if "n" in kwargs and kwargs["n"] > 1:
            kwargs["n"] = 1
        return super().generate(prompts, stop=stop, callbacks=callbacks, **kwargs)

    async def agenerate(self, prompts, stop=None, callbacks=None, **kwargs):
        self._rotate_key()
        if "n" in kwargs and kwargs["n"] > 1:
            kwargs["n"] = 1
        return await super().agenerate(prompts, stop=stop, callbacks=callbacks, **kwargs)


class RAGEvaluator:
    """
    Automated RAG evaluation engine using RAGAS and custom lexical metrics.
    Utilizes multi-key rotation for high-performance clinical validation.
    """

    def __init__(self):
        print(f"Initializing RAGAS evaluator (Multi-Key Support: {len(API_KEYS)} keys)...")

        # LLM Judge with Multi-Key Rotation
        self.llm = GroqMultiKeyLLM(
            model       = "llama-3.1-8b-instant",
            keys        = API_KEYS,
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
        Run RAGAS evaluation over all RAG outputs.
        """
        dataset = Dataset.from_dict({
            "question"  : [r["question"]          for r in rag_results],
            "answer"    : [r["answer"]             for r in rag_results],
            "contexts"  : [r["contexts"]           for r in rag_results],
            "ground_truth": [r["ground_truth"]     for r in rag_results],
        })

        print("Executing parallel RAGAS semantic evaluation...")
        
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
        Run lightweight custom metrics.
        """
        rows = []
        for item in rag_results:
            custom = compute_all_custom_metrics(item)
            custom["question"] = item["question"]
            rows.append(custom)
        return pd.DataFrame(rows)


    def evaluate_all(self, rag_results: list[dict]) -> pd.DataFrame:
        """
        Full evaluation: RAGAS + custom metrics.
        """
        ragas_df  = self.run_ragas(rag_results)
        
        if "user_input" in ragas_df.columns and "question" not in ragas_df.columns:
            ragas_df = ragas_df.rename(columns={"user_input": "question"})

        custom_df = self.run_custom_metrics(rag_results)

        if "question" in ragas_df.columns:
            merged_df = ragas_df.merge(custom_df, on="question", how="left")
        else:
            print("Warning: Contextual merge failed. Using direct concatenation.")
            custom_cols = custom_df.drop(columns=["question"], errors="ignore")
            merged_df = pd.concat([ragas_df, custom_cols], axis=1)

        csv_path = RESULTS_DIR / "eval_results.csv"
        merged_df.to_csv(csv_path, index=False)
        
        numeric_cols = merged_df.select_dtypes(include="number").columns
        summary = merged_df[numeric_cols].describe().round(4)
        summary.to_csv(RESULTS_DIR / "eval_summary.csv")

        return merged_df


    def print_summary(self, df: pd.DataFrame):
        """Print a readable summary table to terminal."""
        print("\n" + "=" * 60)
        print("  RAG EVALUATION SUMMARY (Multi-Key Result)")
        print("=" * 60)

        metrics = [
            "faithfulness", "answer_relevancy", "context_recall",
            "context_precision", "answer_correctness", "token_overlap_vs_gt"
        ]

        for col in metrics:
            if col in df.columns:
                score = df[col].mean()
                bar   = "█" * int(score * 20)
                print(f"  {col:<30} {score:.3f}  {bar}")
        print("=" * 60)


if __name__ == "__main__":
    outputs_path = RESULTS_DIR / "rag_outputs.json"
    if not outputs_path.exists():
        print("Error: RAG output file not found.")
        exit(1)

    with open(outputs_path) as f:
        rag_results = json.load(f)

    evaluator = RAGEvaluator()
    df        = evaluator.evaluate_all(rag_results)
    evaluator.print_summary(df)
