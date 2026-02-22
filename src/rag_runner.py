import sys
import os
import json
from pathlib import Path

# Add project root to path - Adjusted for actual workspace structure
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Clinical NLP RAG Chatbot" / "01-clinical-rag-chatbot" / "src"))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"

# Medical-specific embedding model for clinical context preservation
EMBED_MODEL  = "emilyalsentzer/Bio_ClinicalBERT"

# Path to FAISS index from Clinical NLP RAG Chatbot project
FAISS_INDEX_PATH = Path(__file__).parent.parent.parent / "Clinical NLP RAG Chatbot" / "01-clinical-rag-chatbot" / "faiss_index"

EVAL_DATA_PATH   = Path(__file__).parent.parent / "data" / "eval_dataset.json"
RESULTS_DIR      = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# RAG Runner Implementation
class RAGRunner:
    """
    RAG pipeline runner for clinical evaluation.

    Encapsulates retrieval using ClinicalBERT-based vector search and 
    generation using high-parameter LLMs.
    """

    def __init__(self):
        print("Loading ClinicalBERT embeddings...")
        # ClinicalBERT is preferred over general-purpose embeddings to better 
        # capture medical terminology and semantic relationships in EHR data.
        self.embeddings = HuggingFaceEmbeddings(
            model_name   = EMBED_MODEL,
            model_kwargs = {"device": "cpu"},
            encode_kwargs= {"normalize_embeddings": True}
        )

        print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Ensure indexer has been run in the chatbot project."
            )
        self.vectorstore = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        print("Initializing LLM client (Groq)...")
        self.llm = ChatGroq(
            model       = GROQ_MODEL,
            api_key     = GROQ_API_KEY,
            temperature = 0.1,
            max_tokens  = 512
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a clinical AI assistant. Answer ONLY based on the "
             "provided context. If the answer is not in the context, say "
             "'I don't have enough information.'\n\nContext:\n{context}"),
            ("human", "{question}")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()
        print("RAG Runner initialization complete.\n")


    def retrieve(self, question: str, k: int = 4) -> list[str]:
        """
        Retrieves relevant context using Max Marginal Relevance (MMR).
        
        MMR balanced relevance with diversity, which is critical in clinical 
        retrieval to avoid redundant snippets and capture a broader set of 
        clinical signs/symptoms.
        """
        docs = self.vectorstore.max_marginal_relevance_search(
            question, k=k, fetch_k=20, lambda_mult=0.5
        )
        return [doc.page_content for doc in docs]


    def answer(self, question: str, contexts: list[str]) -> str:
        """Generate answer from Groq given retrieved contexts."""
        context_str = "\n\n".join(contexts)
        return self.chain.invoke({
            "context" : context_str,
            "question": question
        })


    def run_eval_dataset(self) -> list[dict]:
        """
        Runs the RAG pipeline over all eval questions.
        Returns list of dicts ready for RAGAS evaluation.
        """
        with open(EVAL_DATA_PATH) as f:
            eval_data = json.load(f)

        print(f"Processing {len(eval_data)} evaluation items...\n")
        results = []

        for i, item in enumerate(eval_data, 1):
            question = item["question"]
            print(f"[{i}/{len(eval_data)}] Processing: {question[:60]}...")

            # Context Retrieval
            retrieved_contexts = self.retrieve(question, k=4)

            # Response Generation
            answer = self.answer(question, retrieved_contexts)

            results.append({
                "question"           : question,
                "answer"             : answer,
                "contexts"           : retrieved_contexts,
                "ground_truth"       : item["ground_truth"],
                "reference_contexts" : item["reference_contexts"]
            })

        # Persist raw evaluation results
        out_path = RESULTS_DIR / "rag_outputs.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nEvaluation outputs persisted to {out_path}")

        return results


if __name__ == "__main__":
    runner  = RAGRunner()
    results = runner.run_eval_dataset()
    print(f"\nSample answer for Q1:\n{results[0]['answer']}")
