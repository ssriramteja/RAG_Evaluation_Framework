# Clinical RAG Evaluation Framework

This framework provides a robust, multi-dimensional evaluation suite for clinical RAG (Retrieval-Augmented Generation) pipelines. It integrates RAGAS for semantic evaluation and a set of custom lexical proxies to ensure clinical accuracy and grounding.

## Conceptual Understanding

### The Challenge of Clinical RAG
Clinical RAG systems must maintain high "faithfulness" to source Electronic Health Records (EHR) to avoid potentially dangerous hallucinations. Standard RAG evaluation often fails to capture the nuance of clinical entities and the specific requirement for medical grounding.

### Technical Foundation
1.  **ClinicalBERT Embeddings**: We utilize `Bio_ClinicalBERT` instead of general-purpose models. This model was pre-trained on a large corpus of EHR notes from MIMIC-III, allowing it to better understand clinical nomenclature, medical abbreviations, and the semantic relationships between symptoms and pathologies.
2.  **Max Marginal Relevance (MMR)**: During retrieval, we use MMR to balance document relevance with context diversity. In clinical scenarios, multiple notes may redundantly mention the same symptom; MMR ensures the LLM receives a broader set of unique clinical signs by penalizing redundancy.
3.  **LLM-as-a-Judge (RAGAS)**: We leverage high-parameter models (Llama 3.3-70b) as "judges" to evaluate metrics that require semantic reasoning, such as answer relevance and context recall.

## Component Architecture

-   **Execution Layer (`rag_runner.py`)**: Interacts with the FAISS vector store and Groq LLM to generate responses for the evaluation dataset.
-   **Validation Layer (`evaluator.py`)**: Orchestrates the evaluation process, invoking RAGAS metrics and local lexical calculations.
-   **Metric Suite (`metrics.py`)**: Computes lightweight indicators such as Jaccard overlap and grounding proxies.
## How it Works: The Evaluation Flow

The framework operates as a decoupled multi-stage pipeline. This design allows for independent retrieval (RAG iteration) and validation (metric calculation) phases.

1.  **Ingestion**: The system reads a seed dataset of clinical questions and ground truth answers.
2.  **Retrieval & Inference**: For each question, the system retrieves relevant clinical context using the `ClinicalBERT` + `FAISS` stack and generates a response using a high-density LLM (Llama 3.3).
3.  **Lexical Scoring**: The generated response is immediately compared to the ground truth using local, fast lexical metrics (Jaccard, SequenceMatcher).
4.  **Semantic Adjudication**: The results are passed to an LLM-based judge (RAGAS) which performs a semantic "sanity check" to verify faithfulness and recall.
5.  **Visualization**: Final scores are cached and rendered into a Streamlit dashboard for stakeholder review.

## Input Data Specification

The system expects a structured JSON file located at `data/eval_dataset.json`.

**Required Fields:**
- `question`: The clinical query to be tested.
- `ground_truth`: The ideal clinical response (provided by a medical professional).
- `reference_contexts`: (Optional) The gold-standard document snippets that *should* be retrieved.

**Example Input:**
```json
{
  "question": "What are the symptoms of patient P001?",
  "ground_truth": "Patient P001 presents with acute chest pain and nausea...",
  "reference_contexts": ["PATIENT_ID: P001. History of hypertension..."]
}
```

## Expected Output Artifacts

The pipeline generates several artifacts in the `results/` directory:

1.  **`rag_outputs.json`**: Contains the raw model responses and the retrieved contexts for every test case.
2.  **`eval_results.csv`**: A structured table containing all computed RAGAS and lexical scores per question.
3.  **`eval_summary.csv`**: Statistical summary (mean, std, min, max) of the entire evaluation run.
4.  **`eval_report.json`**: A machine-readable summary used to drive the Streamlit dashboard and external reporting tools.

### 1. Environment Configuration
Ensure you have the necessary dependencies and API access.
```bash
# Install required libraries
pip install -r requirements.txt
```
The `.env` file must contain a valid `GROQ_API_KEY`. The current project is pre-configured with the project key.

### 2. Pipeline Execution
The evaluation follows a sequential three-stage pipeline to transform raw data into actionable insights.

#### Stage A: Data Generation
Run the RAG runner to process the evaluation dataset and generate model responses.
```bash
python src/rag_runner.py
```
**Process**: This script loads the ClinicalBERT embeddings, connects to the FAISS index from the sibling project, and generates a `rag_outputs.json` file in the `results/` directory.

#### Stage B: Metric Calculation
Invoke the evaluation engine to compute semantic and lexical scores.
```bash
python src/evaluator.py
```
**Process**: This script uses RAGAS to analyze the `rag_outputs.json`. It calculates semantic metrics (faithfulness, recall) using Llama 3.3 as an adjudicator and merges them with custom lexical proxies into `results/eval_results.csv`.

#### Stage C: Report Synthesis
Generate a summary report for documentation purposes.
```bash
python src/report.py
```
**Process**: Aggregates the results into a JSON summary (`results/eval_report.json`) and prints a Markdown performance table.

### 3. Visual Analytics
Launch the Streamlit dashboard to perform deep-dives into specific cases and aggregate trends.
```bash
streamlit run ui/dashboard.py
```

## Metric Definitions

| Metric | Category | Clinical Significance |
|---|---|---|
| **Faithfulness** | Semantic | Ensures the answer is derived *only* from the retrieved context (avoids hallucination). |
| **Answer Relevancy** | Semantic | Measures how well the response addresses the specific clinical query. |
| **Context Recall** | Semantic | Checks if all information required to answer the question was successfully retrieved. |
| **Grounding Proxy** | Lexical | Measures the percentage of answer keywords present in the context; a rapid indicator of trust. |
| **Token Overlap** | Lexical | Measures lexical consistency between the model response and the clinical ground truth. |
