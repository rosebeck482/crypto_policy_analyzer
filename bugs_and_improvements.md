# Bugs and Improvements

Date: 2025-09-21

This file lists concrete bugs and actionable improvements identified across the repository. File references use path:line for quick lookup.

---

## Critical Bugs (Must Fix)

- OpenAI key hard-failure at import time
  - policy_analyzer/web_analyzer.py:54 – Raises ValueError if `OPENAI_API_KEY` is missing, crashing the app on import. Make it lazy/optional and fail per-request with a clear error or provide a fallback.

- Broken imports and module paths
  - policy_analyzer/__init__.py:6 – `from .data_processor import DataProcessor` fails; `data_processor` is a package. Use `from .data_processor.data_processor import DataProcessor` or expose via `policy_analyzer/data_processor/__init__.py`.
  - policy_analyzer/data_processor/data_processor.py:28 – Uses `from document_chunker import ...` (non-package import). Should be `from .document_chunker import ...`.
  - policy_analyzer/data_processor/data_processor.py:30 – Uses `import semantic_metadata` (non-relative). Should be `from . import semantic_metadata`.
  - policy_analyzer/query_analyze_prompt.py:192 – Uses `from web_analyzer import WebAnalyzer` (non-relative). Should be `from .web_analyzer import WebAnalyzer`.

- Undefined method usage in advanced pipeline
  - policy_analyzer/query_analyze_prompt.py:269 – Calls `self.analyzer.hybrid_rerank_retrieve(...)` which does not exist. Should call `hybrid_retrieval` or `enhanced_hybrid_retrieval`.

- Misplaced function with invalid `self`
  - policy_analyzer/query_analyze_prompt.py:319 – Defines `process_query(self, ...)` at module scope, referencing `self`. This will crash if used. Remove or move inside a class.

- Elasticsearch connection config not loaded
  - policy_analyzer/data_processor/data_processor.py:33–41 – ES params are empty string defaults and not loaded from env; code reads `ES_API_KEY` only (line 129). If `ES_URL` is unset, ES client fails. Load all ES config from environment (or `.env`).

- ES field mismatches across components
  - policy_analyzer/web_analyzer.py:198 – BM25 expects `_source.text`.
  - policy_analyzer/web_analyzer.py:728 – `find_document_by_content` searches `_source.page_content`.
  - Ensure the indexing pipeline stores both fields (or align both retrievals to one field).

- README run instruction is wrong
  - README.md: “python -m policy_analyzer.data_processor” – There is no `__main__.py` in `policy_analyzer/data_processor/`. Should be `python -m policy_analyzer.data_processor.data_processor` or add a `__main__.py`.

---

## Security Issues

- TLS verification disabled for Elasticsearch
  - policy_analyzer/web_analyzer.py:103, 120 – `verify_certs=False`. Enable certificate verification and allow configuring CA bundle for production.

- Risky default Elasticsearch URL
  - app.py:30 – Defaults to a real Elastic Cloud endpoint. Use `http://localhost:9200` as default and document how to configure cloud URLs.

---

## Functional Bugs / Inconsistencies

- GraphRAG imports and symbols don’t align
  - policy_analyzer/web_analyzer.py:27 – Imports `GraphSearch` from `graphrag_core.search`, but that module defines `KGSearch`. Also depends on external `api`, `rag`, `graphrag` modules (not present). Graph mode remains unusable.

- UI expects fields not returned by backend
  - policy_analyzer/templates/index.html:415 – Uses `source.url`; backend returns `{"content", "metadata"}` and likely `metadata.source`. Update to `source.metadata.source` or include a `url` key in API response.

- Advanced mode UI and backend mismatch
  - policy_analyzer/templates/index.html:394–397 – Expects `sub_queries` with `{hop, query}` objects; advanced backend returns a list of strings. Normalize shape on either side.

- Ingestion uses internal data_links, not top-level
  - policy_analyzer/data_processor/data_processor.py:101–107 – Loads links from `policy_analyzer/data_processor/data_links`. README and top-level `data_links` suggest project root. Clarify and unify.

---

## Dependency Problems

- Missing deps for advertised features
  - DSPy not listed (needed by `dspy_multihop.py` and `query_analyze_prompt.py`).
  - GraphRAG stack dependencies (`graphrag`, `rag`, `api`, `redis`, `xxhash`, etc.) absent.

- Likely invalid or too-new pins
  - requirements.txt – Pins like `flask==3.1.0`, `openai==1.68.2`, and multiple `langchain-*` 0.3.x may not exist or be untested together. Validate against PyPI and lock to a known-good set.

- Unused heavy dependencies
  - `llama-index*`, `sentence-transformers`, `pandas` appear unused in current code paths. Remove unless required.

---

## Robustness & Performance

- No network timeouts/retries for Docling fetch
  - policy_analyzer/data_processor/document_chunker.py: URL timeout parameter isn’t actually applied to Docling’s `convert`. Add explicit timeout/retries at the request layer or via Docling config if supported.

- Dedup uses Python’s non-stable `hash()`
  - policy_analyzer/web_analyzer.py:451, 465, 475 – Dedup keys use `hash(doc.page_content)`. Use a stable digest like SHA-256 of normalized content.

- Threaded ES indexing lacks backpressure/retry
  - policy_analyzer/data_processor/data_processor.py:315–339 – Consider retries on batch failure with exponential backoff; surface partial failures.

---

## Code Quality / Maintainability

- Deprecated embedding model
  - policy_analyzer/web_analyzer.py:80 – Uses `text-embedding-ada-002`. Migrate to `text-embedding-3-small/large` and re-index accordingly.

- Misleading label in index metadata
  - policy_analyzer/data_processor/data_processor.py:348 – Writes “Chunking method: semantic” while only header splitting is active.

- Duplicate file
  - Two copies of `policy_analyzer/graphrag_core/entity_resolution_prompt.py`. Remove duplicate.

---

## Documentation Improvements

- Correct run commands and structure diagram
  - README.md – Update module path for the processor and directory layout to match reality (data_processor under a subpackage).

- Document configuration
  - Explain which `data_links` file is used by which component and how to change it.
  - Clarify ES field schema used for vector and BM25 search (`page_content` vs `text`).

---

## Suggested Fix Plan (Prioritized)

1) Make runtime resilient
- Gracefully handle missing `OPENAI_API_KEY` (defer checks, offer fallback).
- Align ES schema across indexing and retrieval; verify BM25 field and vector field.
- Enable TLS verification for ES with configurable CA bundle.

2) Repair imports and module layout
- Switch to relative imports within the package; fix `policy_analyzer/__init__.py` export.
- Adjust README commands or add `policy_analyzer/data_processor/__main__.py`.

3) Triage features
- Gate or remove advanced/DSPy and GraphRAG paths until dependencies are added and code is wired correctly.

4) Dependencies
- Audit and lock to known-good versions; remove unused packages; add missing ones if features are kept.

5) Ingestion hardening
- Add HTTP timeouts, retries with jitter around Docling fetching; optional per-domain limits.

6) UI corrections
- Use `source.metadata.source` (or emit `url` server-side) and align `sub_queries` shape with backend.

