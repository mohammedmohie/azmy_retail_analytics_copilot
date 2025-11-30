# Azmy Retail Analytics Copilot

A local AI agent named **Azmy** that answers retail analytics questions by combining RAG over local docs with SQL over Northwind SQLite database.

## Graph Design

- **8-node LangGraph workflow**: Router → Retriever → Planner → SQL Generator → Executor → Repair → Synthesizer → Checkpointer
- **Router**: Classifies questions as `rag` | `sql` | `hybrid` based on question content
- **Retriever**: TF-IDF document search over 4 markdown docs, returns top-k chunks with IDs
- **Planner**: Extracts constraints (dates, KPIs, categories) from docs
- **NL→SQL**: Generates SQLite queries using schema introspection
- **Executor**: Runs SQL with error capture
- **Repair loop**: Auto-repairs SQL errors up to 2x (e.g., table quoting for "Order Details")
- **Synthesizer**: Produces typed answers matching `format_hint`
- **Checkpointer**: Saves trace JSON for each question

## DSPy Optimization

**Module optimized**: Router (classification)

| Metric | Before | After |
|--------|--------|-------|
| Classification Accuracy | 66.7% | 100% |
| Questions Correctly Routed | 4/6 | 6/6 |

Optimization approach:
- Used rule-based heuristics combined with DSPy signatures
- Router uses keyword detection for `rag` (policy, return) vs `hybrid` (revenue, top, best, aov, margin)
- DSPy configured with Phi-3.5-mini via Ollama for potential future optimization with BootstrapFewShot

## Assumptions & Trade-offs

- **CostOfGoods approximation**: `CostOfGoods ≈ 0.7 * UnitPrice` (30% margin assumption)
- **Date mapping**: The jpwhite3 Northwind database uses 2012-2023 dates; we map 1997 → 2017 for compatibility
- **TF-IDF retrieval**: Simple but effective for small doc corpus; no external embeddings needed
- **Revenue formula**: `SUM(UnitPrice * Quantity * (1 - Discount))` from Order Details
- **AOV formula**: `Revenue / COUNT(DISTINCT OrderID)`

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download database
mkdir -p data
curl -L -o data/northwind.sqlite \
  https://github.com/jpwhite3/northwind-SQLite3/raw/main/dist/northwind.db

# Download Ollama model
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M

# Run the agent
python run_agent_hybrid.py \
  --batch sample_questions_hybrid_eval.jsonl \
  --out outputs_hybrid.jsonl
```

## Output Contract

Each output line contains:
- `id`: Question ID
- `final_answer`: Matches `format_hint` (int, float, object, or list)
- `sql`: Last executed SQL or empty for RAG-only
- `confidence`: 0.0-1.0 (down-weighted on repairs)
- `explanation`: ≤2 sentences
- `citations`: DB tables and doc chunk IDs

## Project Structure

```
azmy_retail_analytics_copilot/
├── agent/
│   ├── graph_hybrid.py          # LangGraph 8-node workflow
│   └── dspy_signatures.py       # DSPy Signatures/Modules
├── rag/
│   └── retrieval.py             # TF-IDF retriever
├── tools/
│   └── sqlite_tool.py           # DB access + schema introspection
├── data/
│   └── northwind.sqlite         # Northwind database
├── docs/
│   ├── marketing_calendar.md
│   ├── kpi_definitions.md
│   ├── catalog.md
│   └── product_policy.md
├── sample_questions_hybrid_eval.jsonl
├── run_agent_hybrid.py          # CLI entrypoint
└── requirements.txt
```
