# Indian Legal RAG — Citation-grounded chatbot

A Streamlit-hosted Retrieval-Augmented Generation app over Indian statutes and cases that **forces every answer to carry verifiable paragraph-level citations** to the indexed corpus.

Built to address a gap surfaced by IL-TUR (ACL 2024): frontier LLMs underperform Indian-domain SOTA models on retrieval-heavy Indian legal tasks. This project pairs an Indian-legal corpus with hybrid retrieval (vector embeddings + BM25, RRF-fused) and a synthesis step that refuses to make claims its retrieval didn't ground.

Two primary modes:
- **Q&A mode** — ask questions about Indian statutes, cases, and doctrines; get cited answers
- **Case Outcome Prediction mode** — describe your case; find similar past cases; see their verdicts + an assessment of what you might expect

---

## Table of Contents

1. [How it works](#how-it-works)
2. [Stack](#stack)
3. [Setup](#setup)
4. [Corpus & Datasets](#corpus--datasets)
5. [Case Outcome Prediction feature](#case-outcome-prediction-feature)
   - [CJPE test cases](#cjpe-test-cases)
   - [BAIL test cases](#bail-test-cases)
6. [Build commands](#build-commands)
7. [ChromaDB index options](#chromadb-index-options)
8. [Embedding backend options](#embedding-backend-options)
9. [Query router](#query-router)
10. [Streamlit UI](#streamlit-ui)
11. [Evaluation (PCR benchmark)](#evaluation)
12. [Adding your own corpus](#adding-your-own-corpus)
13. [Repository layout](#repository-layout)
14. [Disclaimers](#disclaimers)

---

## How it works

Given a question about Indian law, the pipeline runs four stages:

```
User question
      │
      ▼
┌─────────────┐    OpenAI API / Groq fallback   ┌──────────────────────┐
│   ROUTER    │ ──────────────────────────────▶  │  RouteDecision       │
└─────────────┘                                  │  intent, doc filter, │
                                                 │  rewritten query     │
                                                 └──────────┬───────────┘
                                                            │
                                                            ▼
                                              ┌─────────────────────────┐
                                              │   HYBRID RETRIEVER      │
                                              │  vector (ChromaDB) +    │
                                              │  BM25 fused by RRF k=60 │
                                              └──────────┬──────────────┘
                                                         │ top-8 chunks
                                                         ▼
                                              ┌─────────────────────────┐
                                              │   SYNTHESIZER           │
                                              │  OpenAI API primary     │
                                              │  Gemini 2.5 Flash/Pro   │
                                              │  forced JSON + citation  │
                                              │  verification           │
                                              └──────────┬──────────────┘
                                                         │
                                                         ▼
                                              Answer with [S#] citations
                                              each marked verified / unverified
```

1. **Route** — The router classifies intent (`statute_lookup` / `case_research` / `legal_concept` / `procedure` / `general_legal_qa`) and rewrites the query for better retrieval. Out-of-scope queries are rejected early.
2. **Retrieve** — Hybrid: top-k vector search (ChromaDB) + top-k BM25 (`rank_bm25`), scores combined via Reciprocal Rank Fusion. Optional `doc_type_filter` (statute / case / rule / constitution).
3. **Synthesize** — The LLM sees the question + numbered `[S#]` source paragraphs and must output a JSON object: answer text, citations (each with `chunk_id` + supporting quote), confidence, and caveats. Every `chunk_id` is cross-checked against the retrieved set — citations pointing to chunks not retrieved are flagged **[unverified]** in the UI.
4. **Display** — Streamlit shows the grounded answer, a collapsible Sources panel, and per-turn routing/timing telemetry.

The pipeline never falls back to ungrounded model knowledge. If retrieval returns nothing, it says so.

---

## Stack

| Layer | Choice |
|---|---|
| **Frontend** | [Streamlit](https://streamlit.io) — `streamlit run app.py` |
| **Primary LLM** | OpenAI-compatible API — primary for routing + synthesis |
| **Routing fallback** | Groq **Llama 3.3 70B Versatile** — fast structured JSON |
| **Synthesis fallback** | Google **Gemini 2.5 Flash** (default) / **Gemini 2.5 Pro** (heavy) |
| **Embeddings** | **fastembed** (ONNX Runtime, `all-MiniLM-L6-v2`, 384-dim) — local, no API cost, no PyTorch |
| **Vector DB** | [ChromaDB](https://www.trychroma.com) — local persistent, cosine similarity |
| **Lexical retrieval** | `rank_bm25` (BM25Okapi) over in-memory chunk corpus |
| **Fusion** | Reciprocal Rank Fusion (Cormack et al., 2009), k=60 |
| **Datasets** | Curated seeds in `data/raw/` + IL-TUR (ACL 2024, `Exploration-Lab/IL-TUR`) |
| **Eval** | IL-TUR Prior Case Retrieval (Recall@1/5/10, MRR@10) |

**LLM provider priority:**
1. **OpenAI-compatible API** (primary) — tried first for every call; auto-skipped on 401/403
2. **Groq Llama 3.3 70B** — fallback for routing/classification calls
3. **Gemini 2.5 Flash/Pro** — fallback for synthesis calls

---

## Setup

```bash
# From the LegalTech root (shared venv for all projects)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r legal-rag/requirements.txt

# Configure .env inside legal-rag/
cp legal-rag/.env.example legal-rag/.env   # or edit directly
```

**Required `.env` keys:**

```env
# Primary LLM — OpenAI-compatible endpoint
OPENAI_API_KEY=<your key>
OPENAI_BASE_URL=<your endpoint base URL>    # e.g. https://api.openai.com/v1
OPENAI_MODEL=gpt-4o                         # model name at your endpoint

# Fallback LLMs (optional — app degrades gracefully without them)
GROQ_API_KEY=<groq key>          # for Llama 3.3 70B routing fallback
GOOGLE_API_KEY=<gemini key>      # for Gemini synthesis fallback

# HuggingFace token — recommended for faster IL-TUR dataset downloads
HF_TOKEN=<hf token>

# Embedder — keep as "local" (uses fastembed ONNX, no API cost)
LEGAL_RAG_EMBEDDER=local

# ChromaDB index — points to the active 19K-chunk corpus
LEGAL_RAG_CHROMA_DIR=data/chroma
LEGAL_RAG_CHROMA_COLLECTION=indian_legal_corpus
```

**Build the corpus and run:**

```bash
cd legal-rag

# Full recommended corpus (~19K chunks, ~5 min, datasets cached from HuggingFace)
python -m data.ingest.build_all --reset --il-tur --pcr-max 300 --cjpe --bail

# Start the app
streamlit run app.py --server.fileWatcherType none

# Windows shortcut (handles port conflicts automatically)
run.bat
```

Open [http://localhost:8501](http://localhost:8501).

> **Windows note:** Always start with `run.bat` or the `--server.fileWatcherType none` flag. This prevents Streamlit's file watcher from inspecting PyTorch internals and crashing. The `run.bat` script handles this automatically.

---

## Corpus & Datasets

The index is built from multiple sources. The full recommended build produces **19,144 chunks**:

| Dataset | Source | Chunks | Language | Flag |
|---|---|---|---|---|
| Seed corpus | `data/raw/` | ~35 | English | Always included |
| PCR cases | IL-TUR `pcr` (300 of 7,070) | ~8,687 | English | `--il-tur` |
| LSI statutes | IL-TUR `lsi` (99 of 100) | ~103 | English | `--il-tur` |
| CJPE judgments | IL-TUR `cjpe` (500 of ~42K) | ~9,580 | English | `--cjpe` |
| BAIL applications | IL-TUR `bail` (500 of ~337K) | ~739 | **Hindi** | `--bail` |

### 1. Seed corpus (local)

Curated summaries of landmark Indian legal documents — key constitutional provisions, landmark Supreme Court cases (Kesavananda Bharati, Maneka Gandhi, Vishaka, Shreya Singhal, Puttaswamy), and fundamental statutes (IPC, Consumer Protection Act, IT Act).

**Example questions:**
- *"What is the basic structure doctrine and which case established it?"*
- *"What are the Vishaka guidelines for sexual harassment at the workplace?"*
- *"Is the right to privacy a fundamental right under the Indian Constitution?"*
- *"What does Article 21 say and how was it expanded in Maneka Gandhi?"*

### 2. IL-TUR PCR — Prior Case Retrieval

Full Supreme Court of India judgment texts. These are the "candidate" cases from the IL-TUR PCR benchmark — actual court judgments covering civil, criminal, and constitutional matters.

**Example questions:**
- *"What did the Supreme Court hold about the scope of Article 142?"*
- *"Find cases where the court interpreted 'public purpose' under land acquisition law."*
- *"How has the Supreme Court interpreted Section 302 IPC in cases of circumstantial evidence?"*
- *"Find precedents on the doctrine of res judicata in civil suits."*

### 3. IL-TUR LSI — Legal Statute Identification

The 100 Indian statute texts from the LSI task. Only the `statutes` split is indexed (actual statute text); the query splits with anonymized `<SECTION>/<ACT>` placeholders are skipped.

**Example questions:**
- *"What does the Indian Evidence Act say about admissibility of confessions?"*
- *"Explain the provisions of the Arbitration and Conciliation Act on interim relief."*
- *"What are the punishments under the NDPS Act for drug trafficking?"*

### 4. IL-TUR CJPE — Court Judgment Prediction with Explanation

Full court judgment texts with binary outcome labels. `outcome=1` means the petitioner/appellant won; `outcome=0` means they lost. Used by the Case Outcome Prediction feature.

**Example questions (Q&A mode):**
- *"What factors do courts consider when assessing whether a civil appeal has merit?"*
- *"In what circumstances will a court reverse a lower court's factual findings?"*
- *"How do courts decide landlord-tenant disputes under rent control law?"*

### 5. IL-TUR BAIL — Bail Prediction

Bail application texts from Indian district courts (primarily Uttar Pradesh). Each record contains facts, arguments, and the judge's opinion, with a binary label (bail granted/denied). Text is in **Hindi (Devanagari)**.

> ⚠️ English queries retrieve Hindi documents with reduced accuracy. A multilingual embedding model improves BAIL retrieval significantly.

**Example questions (Q&A mode):**
- *"What factors do courts consider when deciding bail in non-bailable offences?"*
- *"Under what circumstances is bail typically denied in assault cases?"*

---

## Case Outcome Prediction feature

Switch to **Case Outcome Prediction** mode in the sidebar. Describe your case, choose the dataset (CJPE or BAIL), and the system:

1. Runs a filtered vector search over only that dataset's chunks
2. Deduplicates results to one entry per unique case
3. Reads the stored `outcome` label from ChromaDB metadata (no extra API call)
4. Calls the LLM to summarise the similar cases and assess your likely outcome

```
Your case description
        │
        ▼
┌──────────────────────────────┐
│  Filtered vector search       │  source_task = "cjpe" or "bail"
└──────────┬───────────────────┘
           │ top similar chunks
           ▼
┌──────────────────────────────┐
│  Dedup by case               │  one result per unique case
│  Read outcome from metadata  │  0 / 1 from ChromaDB, no API
└──────────┬───────────────────┘
           │ similar cases + outcome stats
           ▼
┌──────────────────────────────┐
│  LLM analysis                │  summary · key factors · your likely outcome
└──────────────────────────────┘
```

**Label meanings:**

| Dataset | `outcome=1` | `outcome=0` |
|---|---|---|
| CJPE | Judgment accepted — petitioner/appellant **won** | Judgment rejected — petitioner/appellant **lost** |
| BAIL | Bail **granted** | Bail **denied** |

---

### CJPE test cases

The CJPE corpus contains Supreme Court of India appeals (civil and criminal). Use these ready-to-paste descriptions to test the feature.

**How:** Sidebar → Case Outcome Prediction → Task: **CJPE** → paste description → submit.

---

**Test 1 — Lease/property dispute (expected: petitioner likely wins)**

> Theatre owner in Ahmedabad leased their property to a film distribution company. The lessee stopped paying rent and refused to vacate after the lease period ended. The owner filed a suit for eviction and recovery of arrears. The trial court decreed in favour of the owner, but the High Court of Bombay reversed the decree and dismissed the suit, holding that no valid lease termination had occurred. The owner now appeals to the Supreme Court.

---

**Test 2 — Lease holding-over dispute (expected: petitioner likely loses)**

> Province of Bengal owns agricultural land leased to a private company for industrial purposes. The lease expired by efflux of time. The lessee continued in possession and paid rent, which the lessor accepted, claiming this created a holding-over under Section 116 of the Transfer of Property Act. The lessor disputes that acceptance of rent implies renewal and seeks eviction, arguing the acceptance was inadvertent.

---

### BAIL test cases

The BAIL corpus is from Indian district courts, primarily in Hindi. English descriptions still retrieve structurally similar applications, though match quality is lower than CJPE. Use these to test the feature.

**How:** Sidebar → Case Outcome Prediction → Task: **BAIL** → paste description → submit.

---

**Test 3 — Bail application (expected: likely granted)**

> Accused is charged under IPC Section 304 Part II (culpable homicide not amounting to murder) arising from a road accident. The accused is a 38-year-old truck driver with no prior criminal record. He has been in judicial custody for 4 months. The chargesheet has been filed and trial is expected to continue for over a year. The accused has a family and is the sole breadwinner. Sureties are available. No allegation of tampering with evidence or threatening witnesses.

---

**Test 4 — Bail application (expected: likely denied)**

> Accused is charged under IPC Section 302 (murder) and Section 120B (criminal conspiracy). The accused is alleged to be part of an organised gang and the crime was premeditated. There are six eyewitnesses. The accused has two prior bail violations in a separate matter. The victim's family has filed a threat complaint against the accused's associates, indicating a risk of witness intimidation.

---

### Outcome Prediction — limitations

| Limitation | Detail |
|---|---|
| Corpus size | 500 CJPE + 500 BAIL indexed by default (out of 42K+ / 337K+ available). More cases → better predictions. Scale with `--cjpe-max` / `--bail-max`. |
| Not a classifier | The system finds similar cases — not a trained outcome predictor. Use for research, not certainty. |
| BAIL is in Hindi | English queries find Hindi documents with reduced accuracy. A multilingual embedder improves this. |
| Binary labels only | Courts weigh many factors. A 60/40 split in similar cases means the outcome is genuinely uncertain. |
| Not legal advice | Always consult a lawyer for actual legal matters. |

---

## Build commands

```bash
# Seed corpus only (~35 chunks, always fast)
python -m data.ingest.build_all

# Recommended: seed + PCR 300 + LSI 100  (~9K chunks)
python -m data.ingest.build_all --il-tur

# Full recommended corpus — PCR + LSI + CJPE + BAIL (~19K chunks, ~5 min)
python -m data.ingest.build_all --reset --il-tur --pcr-max 300 --cjpe --bail

# Scale up PCR for more precedent coverage
python -m data.ingest.build_all --reset --il-tur --pcr-max 1000 --cjpe --bail

# Maximum PCR (~7K cases, ~55 min — produces ~256K chunks)
python -m data.ingest.build_all --reset --il-tur --pcr-max 7070 --cjpe --bail

# More CJPE cases (42K+ available)
python -m data.ingest.build_all --reset --il-tur --cjpe --cjpe-max 2000 --bail

# Custom limits
python -m data.ingest.build_all --reset --il-tur --pcr-max 500 --cjpe --cjpe-max 1000 --bail --bail-max 1000
```

> **`--reset` is required when switching embedding models.** ChromaDB stores dimension-specific vectors. Switching from 384-dim (local) to 768-dim (Gemini) without `--reset` causes a dimension mismatch error at insert time.

---

## ChromaDB index options

The active index is controlled by two `.env` variables:

```env
LEGAL_RAG_CHROMA_DIR=data/chroma           # path relative to legal-rag/
LEGAL_RAG_CHROMA_COLLECTION=indian_legal_corpus
```

Two indexes ship with this repo:

| Directory | Collection | Chunks | Dim | Notes |
|---|---|---|---|---|
| `data/chroma` | `indian_legal_corpus` | **19,144** | 384 | **Default** — full corpus (PCR 300 + CJPE 500 + BAIL 500 + LSI + seed) |
| `data/chroma_v2` | `indian_legal_v2` | 8,825 | 384 | Earlier build — smaller corpus, good as fallback |

To switch indexes, edit the two lines in `.env` and restart the app — no code changes needed. Both use 384-dim fastembed vectors so they are interchangeable without rebuilding.

---

## Embedding backend options

Set `LEGAL_RAG_EMBEDDER` in `.env`:

| Backend | `.env` value | Dim | Notes |
|---|---|---|---|
| **fastembed / local** *(active)* | `local` | 384 | ONNX Runtime, `all-MiniLM-L6-v2`. **No PyTorch, no API cost.** Safe in Streamlit threads on Python 3.14+. Recommended. |
| **Gemini** | `gemini` | 768 | Requires `GOOGLE_API_KEY`. High quality but costs API credits. Index must be rebuilt when switching. |
| **Voyage AI** | `voyage` | 1024 | Requires `VOYAGE_API_KEY`. `voyage-law-2` is fine-tuned for legal text — best retrieval quality. |
| **HF Inference API** | `hf_api` | varies | Requires network access to HuggingFace. Rate-limited on free tier. |

> **Switching embedders requires a full index rebuild** (`--reset`). Vectors from different models are not compatible. The current active index (`data/chroma`, 384-dim) must be rebuilt if you switch to Gemini (768-dim) or Voyage (1024-dim).

**Why fastembed instead of sentence-transformers:**
The `local` backend uses [fastembed](https://github.com/qdrant/fastembed) (ONNX Runtime) rather than `sentence-transformers` (PyTorch). On Python 3.14, PyTorch causes a hard segfault when loaded inside Streamlit's script runner thread. fastembed avoids PyTorch entirely while producing equivalent embedding quality from the same model weights.

---

## Query router

The router runs before every Q&A query. It does three things:

1. **Intent classification** — decides the query type:
   - `statute_lookup` — looking for specific statute text
   - `case_research` — searching for precedent cases
   - `legal_concept` — asking about a doctrine or principle
   - `procedure` — asking about a legal procedure
   - `general_legal_qa` — general Indian law question
   - `out_of_scope` — not about Indian law → rejected immediately

2. **Query rewriting** — rewrites the question into a retrieval-optimised form.

3. **Doc-type filtering** — restricts retrieval to `statute` or `case` based on intent.

**When to skip the router (sidebar toggle):**
- Your question is already precise and well-formed
- You're debugging retrieval directly
- Your question spans multiple doc types
- You want to avoid the API call latency (~0.5s)

| With router | Without router |
|---|---|
| Query rewritten for retrieval | Raw question used as-is |
| Auto doc-type filter applied | No filter — all types searched |
| Out-of-scope queries rejected | All queries reach retrieval |
| +0.5–1s latency | Faster |

---

## Streamlit UI

```bash
streamlit run app.py --server.fileWatcherType none
# Windows: double-click run.bat
```

**Sidebar controls:**

| Control | What it does |
|---|---|
| **Case Outcome Prediction** toggle | Switch between Q&A and Outcome Prediction modes |
| Dataset | CJPE (English court judgments) or BAIL (Hindi bail applications) |
| Restrict to doc types | Filter retrieval to statute / case / rule / constitution |
| Use Gemini 2.5 Pro | Switch to heavier synthesis model (slower, higher quality) |
| Inspect retrieval only | Skip LLM synthesis — shows retrieved chunks only (Q&A mode) |
| Skip router | Bypass routing/rewriting — use raw question for retrieval |
| Clear chat history | Wipe the session |

**Reading the Q&A answer:**
- `[S1]`, `[S2]`, ... are citation markers in the answer text
- `✅` = citation verified (chunk_id matched a retrieved chunk)
- `•` = citation unverified (model cited a chunk_id it didn't actually retrieve)
- Footer shows: intent, rewritten query, confidence, elapsed time

**Reading the Outcome Prediction result:**
- `✅` = favorable (judgment accepted / bail granted)
- `❌` = unfavorable (judgment rejected / bail denied)
- Outcome counts shown prominently: "X favorable / Y unfavorable"
- LLM assessment below the statistics
- Expandable panel with actual snippets from similar cases

---

## Evaluation

Benchmark the retrieval system against IL-TUR's held-out PCR queries:

```bash
# Fast run — 100 queries (~1-2 min)
python -m eval.il_tur_pcr

# Tighter confidence intervals
python -m eval.il_tur_pcr --n-queries 500

# Keep the eval index after run (for inspection)
python -m eval.il_tur_pcr --keep-index
```

The script builds an isolated ChromaDB collection from PCR candidates, runs held-out query cases through the retriever, and computes **Recall@1**, **Recall@5**, **Recall@10**, and **MRR@10**. Results are written to `eval/results/`.

> Fill in your results after running:
> ```
> Dataset:    Exploration-Lab/IL-TUR :: pcr
> Embedder:   fastembed / all-MiniLM-L6-v2 (384-dim)
> Recall@1:   <fill>
> Recall@5:   <fill>
> Recall@10:  <fill>
> MRR@10:     <fill>
> ```

---

## Adding your own corpus

Drop any `.txt` file into `data/raw/statutes/` or `data/raw/cases/` with this header format:

```
TITLE: <human-readable title>
DOC_TYPE: <constitution | statute | rule | case>
JURISDICTION: IN
CITATION: <e.g. (2017) 10 SCC 1>
SOURCE_URL: <link to authoritative source>
SOURCE_NOTE: <optional note about verification quality>

<blank line>
<body text>
```

Then rebuild (no `--reset` needed — upsert is safe):

```bash
python -m data.ingest.build_all --il-tur --cjpe --bail
```

The chunker recognises Indian legal section markers (`Section`, `Article`, `(1)`, `Chapter`) and splits on paragraph boundaries with a ~350-token target chunk size and 60-token overlap.

---

## Repository layout

```
legal-rag/
├── app.py                          # Main Streamlit UI (Q&A + Outcome Prediction)
├── app2.py                         # Simplified Q&A-only UI (no outcome prediction)
├── run.bat                         # Windows launcher (kills port 8501, starts app)
├── core/
│   ├── config.py                   # .env loading, paths, model names
│   ├── llm.py                      # OpenAI / Groq / Gemini clients with retries + fallback
│   ├── embeddings.py               # Embedding backends (fastembed/local/gemini/voyage/hf_api)
│   ├── citation.py                 # Chunk dataclass + stable SHA1 chunk_id
│   ├── chunker.py                  # Paragraph-aware chunker with section markers
│   ├── index.py                    # ChromaDB persistent index (upsert, query, all_chunks)
│   ├── retriever.py                # Hybrid vector + BM25, RRF fusion
│   ├── prompts.py                  # Router few-shot + synthesis system prompt
│   ├── router.py                   # LLM classifier → RouteDecision
│   └── synthesis.py                # answer_question() and predict_outcome() pipelines
├── data/
│   ├── raw/
│   │   ├── statutes/               # Curated statute summaries
│   │   └── cases/                  # Curated case summaries
│   ├── ingest/
│   │   ├── load_local.py           # Reads every .txt under data/raw/
│   │   ├── fetch_il_tur.py         # IL-TUR loader (PCR / LSI / CJPE / BAIL)
│   │   ├── scrape_indiankanoon.py  # ToS-safe stub (raises by default)
│   │   └── build_all.py            # CLI orchestrator for corpus build
│   ├── chroma/                     # Active index — 19,144 chunks, 384-dim (default)
│   └── chroma_v2/                  # Backup index — 8,825 chunks, 384-dim
├── scripts/
│   └── rebuild_gemini.py           # One-off: rebuilds index with Gemini 768-dim embeddings
├── eval/
│   ├── il_tur_pcr.py               # PCR benchmark: Recall@k, MRR
│   └── results/                    # JSON eval outputs
├── requirements.txt
├── .env                            # API keys — do not commit
└── .gitignore
```

---

## Disclaimers

- **Not legal advice.** This is a research and demonstration tool. Verify every citation against the official source before relying on it.
- **Curated statute seeds are SUMMARIES.** Several `data/raw/statutes/*.txt` files are short curated summaries clearly marked with `SOURCE_NOTE`. For production use, replace with full bare-act text from [indiacode.nic.in](https://indiacode.nic.in).
- **Indian Kanoon is intentionally not scraped.** See `data/ingest/scrape_indiankanoon.py` for the rationale and lawful alternatives.
- **BAIL dataset is in Hindi.** English-only embedding models will have limited effectiveness retrieving Hindi documents from English queries.
- **CJPE and BAIL outcome predictions are statistical, not deterministic.** The system retrieves similar historical cases — it does not run a trained classifier. A 6/4 split in retrieved outcomes means the matter is genuinely uncertain.

---

## License

MIT for the code. Dataset usage governed by upstream sources — `Exploration-Lab/IL-TUR` is CC-BY-NC-SA per the IL-TUR paper (ACL 2024, arXiv:2407.05399); check before redistribution.
