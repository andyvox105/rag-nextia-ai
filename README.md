# Larkfield Policy Assistant — RAG from Scratch

A Retrieval-Augmented Generation (RAG) system built from first principles to answer employee questions using Larkfield's enterprise policy documentation with strict citations and zero hallucination.

In this repository, I document my hands-on journey exploring and implementing each component of the RAG pipeline—from raw document ingestion and parsing to lexical search (BM25), dense semantic retrieval (vector embeddings), hybrid fusion (RRF), cross-encoder reranking, optional LLM query rewriting, and quantitative evaluation (`hit@k`, `MRR@k`).

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
                             ▼
              [ 3. Optional LLM Query Rewriting ]
         (Mock replay / OpenAI-compatible: Ollama, Gemini, OpenAI)
                             │
                             ▼
              [ 4. Metadata & Access Pre-Filtering ]
               (as_of effective date & staff vs public)
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
   [ Lexical Index ]                 [ Dense Index ]
  SQLite FTS5 (BM25)             multilingual-e5-small
    (Top 20 hits)               (NumPy Cosine, Top 20)
             │                               │
             └───────────────┬───────────────┘
                             ▼
                     [ 5. Hybrid Search ]
              (Reciprocal Rank Fusion - RRF, Top 20)
                             │
                             ▼
                 [ 6. Cross-Encoder Rerank ]
         (mmarco-mMiniLMv2-L12-H384-v1 → Top 5)
                             │
                             ▼
             [ 7. Context Assembly & Generation ] (Next step)
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

### 7. Cross-Encoder Reranking & Unified Retriever (`policy_assistant/rerank.py`, `policy_assistant/search.py`)
- Built a unified `Retriever` entry point orchestrating `bm25`, `dense`, `hybrid`, and `rerank` with pre-filtering (`as_of`, `audience`) applied before ranking.
- Added a second-stage multilingual cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) that jointly reads `(query, chunk)` pairs over the top 20 hybrid candidates to re-score and select the top 5 evidence chunks.

### 8. Optional LLM Query Rewriting & Provider Layer (`policy_assistant/rewrite.py`, `config.py`, `providers.py`)
- Added optional `--rewrite` query rewriting driven by `policy_assistant/prompts/rewrite.md` to translate/normalize colloquial or French questions into English policy terminology while preserving product codes and dates.
- Built a clean provider abstraction (`MockProvider` replaying `recordings/rewrites.jsonl` for offline/reproducible runs, and `OpenAICompatibleProvider` supporting local Ollama, Google Gemini, or OpenAI APIs configured via `.env`).

### 9. Quantitative Evaluation & Benchmark Results (`policy_assistant/evaluate.py`)
Evaluated on Grace's 67-question benchmark (59 answerable questions with relevant passages, `top 5`, `audience=staff`):

| Method | `hit@5` | `MRR@5` | Notes |
|---|---:|---:|---|
| `bm25` | 0.831 | 0.655 | Strong on `exact-code` (1.000 / 0.955), weaker on `table` (0.625 / 0.354) |
| `hybrid` (BM25 + Dense + RRF) | 0.864 | 0.745 | Combines lexical and semantic signals across languages |
| `rerank` (Hybrid top-20 → Cross-Encoder) | **0.898** | **0.831** | Achieves 1.000 / 1.000 on `exact-code`, `contradiction`, and `staff-only` |
| `rerank --rewrite` (recorded `chat-small`) | **0.932** | **0.882** | Lifts `table` to 1.000 / 0.938; kept opt-in due to latency and edge cases (e.g., Q60 Quebec French doc) |

---

## Repository Structure

```
├── .env.example                    # Configuration template (mock, Ollama, or OpenAI-compatible API)
├── corpus/
│   ├── documents/                  # 36 enterprise policy documents (MD, HTML, PDF)
│   ├── formats/                    # Parallel document formats for parser benchmarking
│   └── inventory.csv               # Document registry (versions, dates, access, languages)
├── questions/
│   └── questions.jsonl             # Benchmark dataset of 67 real-world test queries
├── recordings/
│   ├── embeddings_embed-small.npz  # Pre-recorded embeddings for offline testing
│   └── rewrites.jsonl              # Pre-recorded query rewrites for reproducible evaluation
├── policy_assistant/
│   ├── __init__.py
│   ├── __main__.py                 # Central CLI dispatcher
│   ├── config.py                   # .env loader, Settings, prompt loader, provider factory
│   ├── providers.py                # MockProvider & OpenAICompatibleProvider
│   ├── prompts/
│   │   └── rewrite.md              # System/user prompt for query rewriting
│   ├── parse.py                    # Multi-format document parser
│   ├── chunk.py                    # Chunking algorithms (fixed, overlap, structure)
│   ├── documents.py                # Inventory and corpus loader
│   ├── metadata.py                 # Data models and chunk schemas
│   ├── store.py                    # SQLite index manager (FTS5 + Vector store)
│   ├── filters.py                  # Temporal (as_of) and access (staff/public) filters
│   ├── lexical.py                  # BM25 Keyword Search (SQLite FTS5)
│   ├── dense.py                    # NumPy Vector Cosine Similarity Search
│   ├── embed.py                    # Embedding model adapters (Local, Recorded, OpenAI)
│   ├── hybrid.py                   # Reciprocal Rank Fusion (RRF)
│   ├── rerank.py                   # Multilingual Cross-Encoder reranker
│   ├── rewrite.py                  # Optional LLM query rewriting
│   ├── search.py                   # Unified Retriever funnel
│   └── evaluate.py                 # Retrieval evaluation engine (Hit@k & MRR@k)
├── tests/                          # Automated Pytest test suite
├── requirements.txt                # Python dependencies (Python >= 3.12)
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

### Configuration (Optional — for live LLM query rewriting)

By default, the assistant uses `ASSISTANT_PROVIDER=mock` and requires no API keys. To use a live LLM (Ollama, Google Gemini, OpenAI, etc.), copy `.env.example` to `.env` (never committed to git) and set:

```env
ASSISTANT_PROVIDER=openai_compatible
ASSISTANT_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
ASSISTANT_MODEL=gemini-3.5-flash-lite
ASSISTANT_API_KEY=<your_api_key>
```

### Verification & Testing

Run the automated test suite (forcing `mock` ensures offline, deterministic runs):

```bash
ASSISTANT_PROVIDER=mock ASSISTANT_MODEL=chat-small python -m pytest
```

---

## CLI Usage

The system exposes a unified CLI via `python -m policy_assistant`:

```bash
# Inspect the document inventory
python -m policy_assistant inventory

# Build the index with structural chunking and vector embeddings
python -m policy_assistant ingest --chunker structure --embedder intfloat/multilingual-e5-small

# Inspect chunking results for a document
python -m policy_assistant chunks warranty-policy

# Search with cross-encoder reranking (default method), date filter, and audience gating
python -m policy_assistant search "Combien de temps dure la garantie d'un barbecue?" --method rerank --k 5
python -m policy_assistant search "return window" --method hybrid --as-of 2026-10-09 --audience public

# Search with optional LLM query rewriting
python -m policy_assistant search "Puis-je retourner une carte-cadeau?" --method rerank --rewrite

# Evaluate Hit@k and MRR@k on the 67-question benchmark
python -m policy_assistant eval --method rerank --k 5
ASSISTANT_PROVIDER=mock ASSISTANT_MODEL=chat-small python -m policy_assistant eval --method rerank --rewrite
```

---

## Next Steps

- [x] Add **Cross-Encoder Reranking** (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`).
- [x] Add optional **LLM Query Rewriting** and `MRR@k` retrieval evaluation.
- [ ] Build the **Generation & Citation Layer** (Module 4) with verifiable attribution and refusal triggers.
- [ ] Evaluate end-to-end question answering pipeline against Grace's reference benchmarks.

---

## License

Code is licensed under the [MIT License](LICENSE). Documents and evaluation questions are released under CC0.
