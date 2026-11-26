# Larkfield Policy Assistant — RAG from Scratch

A Retrieval-Augmented Generation (RAG) system built from first principles to answer employee questions using Larkfield's enterprise policy documentation with strict citations and zero hallucination.

In this repository, I document my hands-on journey exploring and implementing each component of the RAG pipeline—from raw document ingestion and parsing to lexical search (BM25), dense semantic retrieval (vector embeddings), cross-lingual support, and quantitative evaluation.

---

## Architecture & Technical Roadmap

```
                    [ Enterprise Documents ]
          (Markdown, HTML, PDF | English & French)
                             │
                             ▼
                 [ 1. Multi-Format Parser ]
             (Extracts text, headings, pages)
                             │
                             ▼
              [ 2. Chunking & Indexing Pipeline ]
          ├── Fixed-size windows
          ├── Overlapping windows
          └── Structure-aware chunking (Heading-based)
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
   [ Lexical Index ]                 [ Dense Index ]
  SQLite FTS5 (BM25)             multilingual-e5-small
                                (NumPy Cosine Similarity)
             │                               │
             └───────────────┬───────────────┘
                             ▼
                     [ Hybrid Search ]
              (Reciprocal Rank Fusion - RRF)
                             │
                             ▼
             [ Metadata & Access Filtering ]
              (as_of date & staff vs public)
                             │
                             ▼
                  [ Cross-Encoder Rerank ] (Next step)
                             │
                             ▼
             [ Context Assembly & Generation ] (Next step)
             (Strict citation & refusal on missing data)
```

---

## What I Have Built & Explored

### 1. Multi-Format Document Parsing (`policy_assistant/parse.py`)
- Handled heterogeneous enterprise sources: **36 documents** across **Markdown**, **HTML**, and **PDF**.
- Extracted semantic blocks while retaining physical and logical layout: section hierarchies, character spans, and page numbers for PDF citations.

### 2. Chunking Strategies & Trade-offs (`policy_assistant/chunk.py`)
Explored and benchmarked three distinct chunking algorithms:
- **`fixed`**: Fixed-word window. Fast, but arbitrarily cuts through sentences, tables, and section headings.
- **`overlap`**: Sliding window with word overlap to preserve sentence boundaries across splits.
- **`structure`**: Section-aware splitting rooted in document headings. Prefixes each chunk with the document title and section hierarchy to preserve localized context (e.g. knowing which product a "Specifications" table belongs to).

### 3. Dual Index Storage (`policy_assistant/store.py`)
- Engineered a single-file SQLite store (`index.sqlite`) capturing:
  - Document metadata & access control labels (`public` vs `staff-only`).
  - Chunks with character offsets, page coordinates, and section paths.
  - Full-Text Search (FTS5) table for fast lexical BM25 matching.
  - BLOB storage for 32-bit float vector embeddings.
  - Index metadata to enforce model version integrity (preventing embedding model mismatch).

### 4. Dense Retrieval & Semantic Embeddings (`policy_assistant/dense.py`, `policy_assistant/embed.py`)
- Implemented brute-force vector search using **NumPy** matrix multiplication over L2-normalized embeddings (cosine similarity).
- **Multilingual Support**: Integrated `intfloat/multilingual-e5-small` (384 dimensions) to support cross-lingual queries (e.g., retrieving French policy documents like `Livret-d-accueil.md` from English questions).
- **Recorded Provider**: Integrated recorded vectors (`embed-small`, 1536 dimensions, based on OpenAI `text-embedding-3-small`) to enable reproducible testing without requiring external API keys.

### 5. Hybrid Search & Fusion (`policy_assistant/hybrid.py`)
- Implemented **Reciprocal Rank Fusion (RRF)** with standard smoothing constant $k=60$ to combine rank-based hits from lexical (BM25) and dense (vector) retrieval without needing score normalization across differing scales.

### 6. Metadata & Access-Control Filtering (`policy_assistant/filters.py`)
- Implemented temporal filtering (`as_of` date vs `effective_from` / `effective_to`) to answer policy questions as of a particular date.
- Enforced document permission gating (`staff` vs `public` audience) to ensure unprivileged queries never receive restricted company policies.

### 7. Empirical Observations & Key Learnings
- **Table vs. Prose Representation**: Small embedding models (e.g., e5-small) can struggle with dense Markdown tables compared to natural prose, sometimes favoring distractor passages that sound linguistically fluent. Larger models (or BM25) significantly alleviate this issue.
- **Cross-Lingual Retrieval**: Verified that semantic embeddings bridge language gaps, achieving high cosine similarity (`>0.81`) between English refund queries and French refund policy clauses without translation.
- **The Case for Hybrid Search**: Lexical BM25 excels at exact technical identifiers and product SKUs (`sprinklers`, `HT-550`), whereas Dense Retrieval excels at synonym matching (`money back` $\leftrightarrow$ `refund`). Combining them is the optimal path forward.

---

## Repository Structure

```
├── corpus/
│   ├── documents/          # 36 enterprise policy documents (MD, HTML, PDF)
│   ├── formats/            # Parallel document formats for parser benchmarking
│   └── inventory.csv       # Document registry (versions, dates, access, languages)
├── questions/
│   └── questions.jsonl     # Benchmark dataset of 67 real-world test queries
├── recordings/
│   └── embeddings_embed-small.npz  # Pre-recorded embeddings for offline testing
├── policy_assistant/
│   ├── __init__.py
│   ├── __main__.py         # Central CLI dispatcher
│   ├── parse.py            # Multi-format document parser
│   ├── chunk.py            # Chunking algorithms (fixed, overlap, structure)
│   ├── documents.py        # Inventory and corpus loader
│   ├── metadata.py         # Data models and chunk schemas
│   ├── store.py            # SQLite index manager (FTS5 + Vector store)
│   ├── lexical.py          # BM25 Keyword Search
│   ├── dense.py            # NumPy Vector Cosine Similarity Search
│   ├── embed.py            # Embedding model adapters (Local, Recorded, OpenAI)
│   └── evaluate.py         # Retrieval evaluation engine (Hit@k)
├── tests/                  # Pytest test suite (31 automated unit tests)
├── requirements.txt        # Python dependencies (Python >= 3.12)
└── pytest.ini
```

---

## Getting Started

### Prerequisites
- Python **3.12** or later.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/andyvox105/rag-nextia-ai.git
cd rag-nextia-ai

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Verification & Testing

Run the automated test suite:

```bash
python -m pytest
```

All 31 unit tests should pass.

---

## CLI Usage

The system exposes a unified CLI via `python -m policy_assistant`:

```bash
# Inspect the document inventory
python -m policy_assistant inventory

# Test the parser on a specific file
python -m policy_assistant parse corpus/documents/Acceptable-use-policy.md

# Build the index with structural chunking and vector embeddings
python -m policy_assistant ingest --chunker structure --embedder intfloat/multilingual-e5-small

# Inspect chunking results for a document
python -m policy_assistant chunks warranty-policy

# Perform lexical (BM25) search
python -m policy_assistant search "What is the warranty period for garden tools?" --k 5

# Evaluate Hit@k on the 67-question benchmark
python -m policy_assistant eval --k 5
```

---

## Next Steps

- [ ] Add **Cross-Encoder Reranking** (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`).
- [ ] Build the **Generation & Citation Layer** with verifiable attribution and refusal triggers.
- [ ] Evaluate end-to-end question answering pipeline against Grace's reference benchmarks.

---

## License

Code is licensed under the [MIT License](LICENSE). Documents and evaluation questions are released under CC0.
