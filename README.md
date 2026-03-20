# OpenRAG

> **Open-Source Universal Knowledge Intelligence Framework**
>
> Multimodal RAG with a built-in API server, declarative YAML pipelines, and enterprise-grade access control.

[![CI](https://github.com/nksnaresh/openrag/actions/workflows/codeql.yml/badge.svg)](https://github.com/nksnaresh/openrag/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Security Policy](https://img.shields.io/badge/security-policy-brightgreen)](SECURITY.md)

---

## What is OpenRAG?

OpenRAG is a production-ready knowledge intelligence framework designed for building, operating, and observing multimodal RAG systems at scale. More than a library, it is a **complete platform** providing:

| Feature | OpenRAG |
|---|---|
| Native modalities | Text, Image, Table, Equation, Code, Audio/Video |
| Built-in API server | ✅ REST + GraphQL + WebSocket (FastAPI) |
| Enterprise UI | ✅ Query Studio + Live Observability Hub |
| Security & RBAC | ✅ JWT + Admin/Viewer Roles + Secured Credentials |
| Automated Governance | ✅ CodeQL Scanning + Secret Push Protection |
| Observability | ✅ OTel Tracing + Prometheus Metrics + Log Streaming |
| Multi-tenancy + ACLs | ✅ Namespace isolation, per-doc ACLs |
| Streaming responses | ✅ SSE + WebSocket |
| Structured output | ✅ JSON schema-constrained answers |
| Pluggable storage | 5 vector DBs · 3 graph DBs · 3 doc stores |
| License | Apache 2.0 |

---

## Quickstart (5 minutes)

```bash
pip install openrag
openrag init          # creates pipeline.yaml + .env
# Fill in your OPENAI_API_KEY in .env
openrag serve --dev   # starts API with in-memory storage
```

In another terminal:
```bash
openrag ingest ./my_docs/ --namespace demo
openrag query "What are the main topics?" --namespace demo
```

Or use the Python SDK:
```python
import asyncio
from openrag import OpenRAG, OpenRAGConfig

async def main():
    async with OpenRAG(config=OpenRAGConfig(namespace="demo")) as rag:
        await rag.ingest("report.pdf")
        result = await rag.query("Summarise the key findings")
        print(result.answer)
        for cite in result.citations:
            print(f"  [{cite.document_title}, p.{cite.page_number}]")

asyncio.run(main())
```

---

## Installation

```bash
# Core only (in-memory adapters, no parsers)
pip install openrag

# With recommended parsers + Qdrant + Neo4j
pip install openrag[pdf,office,qdrant,neo4j,postgres]

# Everything
pip install openrag[all]
```

---

## Configuration

Copy `env.example` to `.env` and fill in your values:

```bash
cp env.example .env
```

Or create a `pipeline.yaml` for full declarative control:

```yaml
namespace: my-project
embedding:
  provider: openai
  model: text-embedding-3-large
vector_db:
  adapter: qdrant
  url: http://localhost:6333
graph_db:
  adapter: neo4j
  url: bolt://localhost:7687
llm:
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
```

---

## Project Structure

```
openrag/
├── openrag/              # Main package
│   ├── models/           # Shared data contracts
│   ├── config.py         # OpenRAGConfig
│   ├── registry.py       # Adapter registry
│   ├── core.py           # OpenRAG façade
│   ├── parsers/          # Document parser adapters
│   ├── processors/       # Modality processors
│   ├── pipeline/         # DAG execution engine
│   ├── embeddings/       # Embedding adapters
│   ├── knowledge/        # KG builder
│   ├── search/           # BM25 + hybrid retrieval
│   ├── query/            # Query orchestrator
│   ├── storage/          # Vector/Graph/Doc store adapters
│   ├── server/           # FastAPI + GraphQL + WebSocket
│   ├── auth/             # JWT + API key auth + ACLs
│   ├── observability/    # OTel + Prometheus + Audit
│   └── cli/              # openrag CLI
└── tests/
    ├── unit/             # Unit tests (no external deps)
    └── integration/      # Integration tests (Docker fixtures)
```

---

## Development Setup

```bash
git clone https://github.com/your-org/openrag
cd openrag
pip install uv
uv sync --group dev
pre-commit install
uv run pytest tests/unit/
```

---

## Documentation

Full docs: [openrag.readthedocs.io](https://openrag.readthedocs.io)

- [Quickstart Guide](docs/quickstart.md)
- [YAML Pipeline Reference](docs/pipeline_config.md)
- [Provider Integration Guides](docs/providers/)
- [Adapter Development Guide](docs/extending/adapters.md)
- [Deployment Guide](docs/deployment/)

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
