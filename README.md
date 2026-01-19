# Benchmarking Retrieval-Supported Large Language Models for Open-Domain Question Answering

This project implements and evaluates various Information Retrieval (IR) techniques for Retrieval-Augmented Generation (RAG) systems. It compares traditional lexical search against semantic, hypothetical (HyDE), and agentic retrieval strategies (PRF, RAG-Fusion) using MS MARCO Question Answering dataset.

## 🚀 Implemented Techniques
1. **Lexical Retrieval:** Sparse keywoard search using **BM25 Okapi**
2. **Semantic Retrieval:** Dense vector using **Sentence Transformers** (`all-MiniLM-L6-v2`) and Cosine Similarity
3. **HyDE (Hypothetical Document Embeddings):** Generates a hypothetical answer using an LLM to ground the vector search
4. **Pseudo-Relevance Feedback (PRF):** Uses an LLM to analyze initial results and rewrite the search query for better precision.
5. **RAG-Fusion (RRF):** Generates multiple query perspectives and re-ranks results using **Reciprocal Rank Fusion**

## 📂 Project Structure
```text
.
├── data/                       # Dataset storage
│   ├── dev_v1.1.json           # Raw MS MARCO dataset
│   └── ms_marco_qna_dataset.csv # Parsed CSV used by the pipeline
├── outputs/                    # Generated answers and evaluation metrics
├── retrieval_techniques/       # Logic for specific retrieval strategies
│   ├── lexical_retrieval_based_conf.py
│   ├── semantic_retrieval_based_conf.py
│   ├── hyde_retrieval_based_conf.py
│   ├── pseudo_relevance_feedback_based_retrieval_conf.py
│   └── rag_fusion_based_retrieval_conf.py
├── evaluation/                 # Metrics and Judging scripts
│   ├── evaluate_confs.py       # BLEU, ROUGE, BERTScore
│   ├── llm_as_judge.py         # LLM-as-a-Judge (GPT-4o)
│   └── calculate_avg_metrics.py # Final averaging script
├── utils/                      # Helper utilities
│   ├── model_factory.py        # LLM Factory
│   └── ms_marco_qna_dataset_parser.py
├── main.py                     # Entry point for retrieval experiments
└── requirements.txt            # Dependencies
```

## Setup & Installation

### Clone the Repository

```bash
git clone https://github.com/GokayGulsoy/CENG-543-Information-Retrieval-Project.git
cd CENG 543 Information Retrieval Course Project
```

### Install Dependencies

Ensure you have Python 3.10+ installed

```bash
pip install -r requirements.txt
```

### Set Environment Variables

Create a `.env` file or set the following variables in your terminal for the LLM provider you intend to use. If you want persistent approach, then set environment variables system wide.

```bash
# Windows (Poweshell)
$env:OPENAI_API_KEY="sk-..."

#Mac/Linux
export OPENAI_API_KEY="sk-..."
```

## Data Preparation 

1. Download the MS [MARCO Question Answering dataset](https://microsoft.github.io/msmarco/).
2. Place the `dev_v1.1.json` file inside the data directory
3. Run the parser from the **root directory** to generate the CSV file (subset of MSC MARCO QNA dataset)


```bash
python utils.ms_marco_qna_dataset_parser
```

## Usage & Execution Order

To perform a complete experiment, you must run the scripts in the following order.

### Step 1: Run Retrieval Experiment

Generates answers for the dataset using a specific technique

```bash
python main.py --technique <TECHNIQUE> --llm-model-id <MODEL>
```

Available Techniques: `lexical` | `semantic` | `hyde` | `prf` | `rrf` 

### Critical Dependency for PRF 

The **Pseudo-Relevance Feedback (PRF)** technique relies on the initial context provided by the Semantic Retrieval. You **must** run the semantic retrieval technique before running PRF.

```bash
# first, run the semantic retrieval to generate the base results 
python main.py --technique semantic --llm-model-id <MODEL>

# Then, run PRF (it reads outputs/ms_marco_qna_with_generated_answers_semantic.csv)
python main.py --technique prf --llm-model-id <MODEL>
```

### Example (RAG Fusion)

```bash
python main.py --technique rrf --llm-model-id gpt-3.5-turbo
```

## Step 2: Calculate Standard Metrics

Computes traditional NLP metrics (BLEU,ROUGE,BERTScore).

```bash
python evaluation.evaluate.py --retrieval-technique <TECHNIQUE>
```

Output: `outputs/ms_marco_qna_with_generated_answers_metrics_<TECHNIQUE>.csv`

## Step 3: Run LLM-as-a-Judge

Uses GPT-4o to grade answers on **Correctness**, **Faithfulness**, and **Context Quality** (Scale 1-5)

```bash
python evaluation.llm_as_judge --retrieval-technique <TECHNIQUE>
```

Output: `outputs/ms_marco_qna_with_generated_answers_metrics_judge_scores_<TECHNIQUE>.csv`

## Step 4: Calculate Final Averages

Aggregates all scores into final report.

```bash
python evaluation.calculate_avg_metrics --retrieval-technique <TECHNIQUE>
```

### Evaluation Metrics Explained

- BLEU / ROUGE: Measures lexical overlap with ground truth answers.
- BERTScore: Measures semantic similarity using contextual embeddings.
- LLM Judge Correctness: Does the AI answer convey the same meaning as the human answer?
- LLM Judge Faithfulness: Is the answer derived only from the retrieved context (hallucination check)?
- LLM Judge Context Quality: Did the retrieval step find the relevant information?


Research Paper Link: [Benchmarking Retrieval-Supported Large Language Models for Open-Domain Question Answering]()
