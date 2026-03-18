# Business Requirements Document (BRD)
## OpenRAG: Open-Source Universal Knowledge Intelligence Framework

| Field | Details |
|---|---|
| **Document Version** | 1.0 |
| **Date** | March 2026 |
| **Framework Name** | OpenRAG |
| **Owner** | [Your Organization] |
| **Status** | Concept / Pre-Alpha |
| **License** | Apache 2.0 |

---

## 1. Executive Summary

**OpenRAG** is a next-generation, full-stack knowledge intelligence framework that enables organizations to build, operate, and query rich multimodal knowledge bases at scale. Unlike narrow, library-only solutions, OpenRAG is designed as a complete platform: it combines a developer SDK, a declarative YAML pipeline engine, a built-in REST and GraphQL API server, cloud-native deployment support, and enterprise-grade access control — all in a single, cohesive system.

OpenRAG is built around the principle that **knowledge doesn't live in text alone**. It natively understands and reasons over six content modalities — text, images, tables, mathematical expressions, source code, and audio/video transcripts — and provides a unified query interface that seamlessly spans all of them. It is designed to be the foundational intelligence layer for AI-native applications across academia, enterprise, and government.

---

## 2. Vision Statement

> *"OpenRAG makes any collection of documents — regardless of format, modality, or scale — instantly queryable by any AI model, any team, and any application, with zero lock-in."*

OpenRAG's mission is to democratize multimodal knowledge retrieval by providing an open, extensible, production-grade framework that any developer can adopt in minutes and any enterprise can trust at scale.

---

## 3. Problem Statement

### 3.1 The Knowledge Fragmentation Crisis

Organizations today accumulate knowledge across a diverse ecosystem of document types:

- Research papers, manuals, and reports in PDF
- Presentations (PPTX), spreadsheets (XLSX), and business documents (DOCX)
- Code repositories, inline diagrams, and technical wikis
- Meeting recordings, training videos, and audio notes
- Dashboard screenshots, architectural diagrams, and schematics

Current AI knowledge solutions fail this reality in compounding ways:

| Problem | Impact |
|---|---|
| **Modality blindness** | Non-text content (charts, diagrams, code, audio) is silently discarded or superficially indexed |
| **Library-only tools** | Require engineering effort to build ingestion servers, APIs, and access control from scratch |
| **Brittle pipelines** | Hardcoded processing pipelines with no runtime reconfigurability |
| **Vendor lock-in** | Tight coupling to a single LLM provider, vector database, or storage backend |
| **No multi-tenancy** | Single knowledge base per deployment; no isolation between teams or clients |
| **Observability gaps** | Black-box processing with no way to monitor pipeline health, latency, or accuracy |
| **No structured output** | Free-text-only answers with no machine-readable structured responses |
| **Weak access control** | All-or-nothing access to the entire knowledge base |

No existing open-source framework addresses all of these problems simultaneously. **OpenRAG is designed to close all of them.**

### 3.2 Quality Gap

Existing solutions that handle multi-modality treat it as an afterthought — a single processor bolt-on to a text-primary pipeline. OpenRAG treats each modality as a first-class citizen with its own dedicated understanding engine, semantic model, and retrieval pathway.

---

## 4. Strategic Objectives

| # | Objective | KPI |
|---|---|---|
| **SO-1** | Enable any developer to build a production multimodal knowledge base in under 30 minutes | Time-to-first-query ≤ 30 min |
| **SO-2** | Support all six content modalities natively without external preprocessing tools | ≥ 6 modalities out-of-box |
| **SO-3** | Provide a built-in, production-ready API server requiring zero additional infrastructure | API server ships with the framework |
| **SO-4** | Achieve provider-agnostic LLM, embedding, and vector DB integration | ≥ 5 LLM providers, ≥ 4 vector DBs |
| **SO-5** | Enable multi-tenant operation with document-level access control | Tenant isolation with per-document ACLs |
| **SO-6** | Give operators full observability into pipeline health and query quality | OpenTelemetry traces + Prometheus metrics |
| **SO-7** | Support declarative YAML pipeline definitions for zero-code pipeline construction | Full pipeline config via YAML |
| **SO-8** | Run natively in cloud, on-premises, and air-gapped environments | Verified in all 3 environments |

---

## 5. Stakeholders

| Stakeholder Group | Role | Core Interest |
|---|---|---|
| **Individual AI Developers** | Primary user | Quick adoption, simple SDK, comprehensive docs |
| **Platform/Infra Engineers** | Primary user | Stable API, cloud-native deployment, scalability |
| **Enterprise AI Teams** | Primary user | Access control, multi-tenancy, SLA compliance |
| **Academic Researchers** | Primary user | Multimodal document understanding, citation support |
| **Data Engineers / MLOps** | Secondary user | Pipeline observability, batch throughput, storage adapters |
| **Security / Compliance Teams** | Secondary user | Access control, audit logging, data residency |
| **Open-Source Community** | Contributors | Extensibility, clear interfaces, permissive license |
| **Project Owner** | Sponsor | Product vision, adoption growth, community governance |

---

## 6. Scope

### 6.1 In Scope

#### Core Framework
- Python SDK (`openrag` package) with declarative and programmatic APIs
- Built-in async pipeline engine with DAG-based task orchestration
- Six native content modality processors: text, image, table, equation, code, audio/video
- Pluggable parser adapters for multiple document ingestion backends
- Dual knowledge representation: dense vector embeddings + sparse knowledge graph
- Cross-modal entity linking and relationship inference
- YAML-based declarative pipeline configuration

#### API & Serving
- Built-in REST API server (FastAPI) for ingestion and query endpoints
- GraphQL query interface for flexible, client-driven queries
- WebSocket endpoint for real-time streaming query responses
- OpenAPI / Swagger documentation generated automatically

#### Storage & Retrieval
- Pluggable vector database adapters: Qdrant, Weaviate, Chroma, pgvector, Pinecone
- Pluggable graph database adapters: Neo4j, ArangoDB, in-memory
- Pluggable document store adapters: PostgreSQL, MongoDB, SQLite
- Hybrid retrieval: vector similarity + graph traversal + keyword (BM25) fusion

#### Query Intelligence
- Five query strategies: dense, sparse, graph, hybrid, and multimodal
- Structured output mode: LLM responses constrained to JSON schemas
- Query-time function calling: trigger external tools during answer generation
- Query rewriting and decomposition for complex multi-hop questions
- Conversational memory for multi-turn dialogue

#### Platform Features
- Multi-tenant architecture with namespace isolation
- Document-level and field-level access control lists (ACLs)
- OpenTelemetry-based distributed tracing
- Prometheus-compatible metrics endpoint
- Structured audit logging for all ingestion and query operations
- Background job scheduler for periodic re-indexing

#### Deployment
- Docker Compose reference deployment
- Kubernetes Helm chart
- Air-gapped / offline operation mode
- Horizontal scaling for API server and worker processes

### 6.2 Out of Scope (v1.0)
- Hosted SaaS managed service
- Native mobile SDK (iOS/Android)
- Real-time document change detection/sync with source systems
- Fine-tuning of embedding or LLM models
- Built-in front-end web UI (community extensions welcome)

---

## 7. Functional Requirements

### 7.1 Document Ingestion

| ID | Priority | Requirement |
|---|---|---|
| FR-ING-01 | Must | The system SHALL accept PDF, DOCX, PPTX, XLSX, EPUB, HTML, and plain text files as ingestion inputs. |
| FR-ING-02 | Must | The system SHALL accept image files (JPG, PNG, BMP, TIFF, WebP, SVG) as standalone documents. |
| FR-ING-03 | Must | The system SHALL accept source code files (.py, .js, .ts, .java, .go, .rs, .sql, etc.) as a distinct modality. |
| FR-ING-04 | Should | The system SHALL accept audio files (MP3, WAV, M4A, OGG) and extract transcripts via configurable ASR models. |
| FR-ING-05 | Should | The system SHALL accept video files (MP4, MOV, MKV) and extract both transcripts and key frame images. |
| FR-ING-06 | Must | The system SHALL provide a URL ingestion endpoint that fetches and parses web pages. |
| FR-ING-07 | Must | The ingestion subsystem SHALL expose a REST API endpoint (`POST /v1/ingest`) accepting files, URLs, or raw JSON content payloads. |
| FR-ING-08 | Must | The system SHALL support batch ingestion of entire directories recursively via API and SDK. |
| FR-ING-09 | Must | The system SHALL detect duplicate documents using content hashing and skip re-ingestion unless forced. |
| FR-ING-10 | Must | The system SHALL support direct injection of pre-structured content payloads without triggering document parsing. |
| FR-ING-11 | Should | The system SHALL support ingestion from cloud object storage (S3, GCS, Azure Blob) via configurable adapters. |
| FR-ING-12 | Must | Ingestion jobs SHALL support pause, resume, and cancellation via job management API. |

### 7.2 Multimodal Content Processing

| ID | Priority | Requirement |
|---|---|---|
| FR-MOD-01 | Must | The system SHALL automatically classify content blocks into one of: text, image, table, equation, code, or audio/video. |
| FR-MOD-02 | Must | Image blocks SHALL be processed by a configurable Vision Language Model (VLM) to produce semantic captions, bounding-box described regions, and spatial relationship extraction. |
| FR-MOD-03 | Must | Table blocks SHALL be parsed into a structured schema (column names, data types, row count) and a natural-language summary. |
| FR-MOD-04 | Must | Mathematical equation blocks SHALL be parsed from LaTeX or MathML into semantic interpretations with domain tagging. |
| FR-MOD-05 | Must | Code blocks SHALL be analyzed for language, purpose, function signatures, and semantic intent using a code-aware LLM. |
| FR-MOD-06 | Should | Audio/video transcript blocks SHALL be processed with speaker diarization and semantic segmentation. |
| FR-MOD-07 | Must | Each content block SHALL receive contextual enrichment from surrounding document content via a configurable sliding context window. |
| FR-MOD-08 | Must | The framework SHALL expose a `ModalityProcessor` interface enabling developers to add custom content type handlers. |
| FR-MOD-09 | Must | Processing pipelines SHALL be definable as YAML documents specifying which processors to run, in what order, with what parameters. |
| FR-MOD-10 | Should | Processors SHALL support parallel execution via a DAG scheduler when there are no data dependencies. |

### 7.3 Knowledge Representation

| ID | Priority | Requirement |
|---|---|---|
| FR-KR-01 | Must | The system SHALL generate dense vector embeddings for every processed content block using a configurable embedding model. |
| FR-KR-02 | Must | The system SHALL build a knowledge graph of entities and relationships extracted from all content modalities. |
| FR-KR-03 | Must | Cross-modal entity linking SHALL associate entities mentioned in text with related images, tables, equations, and code blocks that reference the same concept. |
| FR-KR-04 | Must | The knowledge graph SHALL support weighted, typed edges (e.g., `defines`, `cites`, `illustrates`, `proves`, `implements`). |
| FR-KR-05 | Must | BM25 keyword indices SHALL be maintained for sparse retrieval in addition to dense vector indices. |
| FR-KR-06 | Should | The system SHALL support hierarchical document structure indexing (section → subsection → block) to enable scope-aware retrieval. |
| FR-KR-07 | Must | Entities SHALL carry provenance metadata: source document ID, page number, block index, modality type. |
| FR-KR-08 | Should | The system SHALL support temporal metadata on entities, enabling time-scoped queries ("find all facts added after date X"). |

### 7.4 Retrieval & Query

| ID | Priority | Requirement |
|---|---|---|
| FR-QRY-01 | Must | The system SHALL support five query strategies: `dense` (vector only), `sparse` (BM25 only), `graph` (KG traversal only), `hybrid` (vector + BM25 fusion), and [multimodal](file:///Users/nareshsingh/MEGA-P/dev/RAG-Anything-main/raganything/query.py#825-850) (cross-modal retrieval with VLM). |
| FR-QRY-02 | Must | The system SHALL support real-time streaming of query responses via WebSocket and Server-Sent Events (SSE). |
| FR-QRY-03 | Must | The system SHALL support structured output queries where the response is constrained to a developer-specified JSON schema. |
| FR-QRY-04 | Must | The system SHALL support function-calling queries where the LLM can invoke registered tools during answer generation. |
| FR-QRY-05 | Must | The system SHALL support multi-hop queries: decomposing complex questions into sub-queries, resolving each, and synthesizing the final answer. |
| FR-QRY-06 | Must | The system SHALL maintain conversational session memory enabling multi-turn queries with context carry-forward. |
| FR-QRY-07 | Must | The system SHALL expose `GET /v1/query` (REST) and a GraphQL query endpoint. |
| FR-QRY-08 | Should | The system SHALL support query-time re-ranking of retrieved candidates using a configurable cross-encoder model. |
| FR-QRY-09 | Should | The system SHALL support hypothetical document embedding (HyDE) as an optional query expansion strategy. |
| FR-QRY-10 | Must | Query responses SHALL include full citations: document name, page number, block index, and modality type for each piece of retrieved evidence. |
| FR-QRY-11 | Should | The system SHALL support query analytics: latency, retrieved context length, relevance score distribution, and answer confidence score. |

### 7.5 Multi-Tenancy & Access Control

| ID | Priority | Requirement |
|---|---|---|
| FR-MT-01 | Must | The system SHALL support multiple isolated tenants (namespaces) within a single deployment, each with an independent knowledge base. |
| FR-MT-02 | Must | Each API request SHALL be authenticated using API keys or JWT tokens scoped to a specific tenant. |
| FR-MT-03 | Must | Document-level access control SHALL allow specifying which users, roles, or groups may retrieve information from a given document. |
| FR-MT-04 | Should | Query results SHALL be automatically filtered to exclude documents the requesting identity does not have access to. |
| FR-MT-05 | Should | An admin API SHALL allow managing users, roles, namespaces, and ACL policies. |

### 7.6 API Server

| ID | Priority | Requirement |
|---|---|---|
| FR-API-01 | Must | The system SHALL include an embedded FastAPI server launchable via `openrag serve`. |
| FR-API-02 | Must | The REST API SHALL publish an OpenAPI 3.0 specification at `/docs`. |
| FR-API-03 | Must | The system SHALL provide a GraphQL endpoint at `/graphql` supporting queries, mutations, and subscriptions. |
| FR-API-04 | Must | The API server SHALL support CORS configuration for browser-based API clients. |
| FR-API-05 | Should | The API server SHALL support rate limiting per API key or IP address. |

### 7.7 Declarative Pipeline Configuration

| ID | Priority | Requirement |
|---|---|---|
| FR-PIPE-01 | Must | The system SHALL support definition of full ingestion and query pipelines via YAML configuration files. |
| FR-PIPE-02 | Must | YAML pipelines SHALL specify: parser, modality processors, embedding model, knowledge graph settings, retrieval mode, and LLM function. |
| FR-PIPE-03 | Should | Pipeline definitions SHALL support environment variable interpolation (e.g., `${OPENAI_API_KEY}`). |
| FR-PIPE-04 | Should | The system SHALL validate pipeline YAML at load time and report configuration errors with clear messages. |
| FR-PIPE-05 | Should | Multiple named pipeline profiles SHALL be supported within a single configuration file. |

### 7.8 Observability

| ID | Priority | Requirement |
|---|---|---|
| FR-OBS-01 | Must | The system SHALL emit OpenTelemetry traces for all ingestion and query operations. |
| FR-OBS-02 | Must | The system SHALL expose a Prometheus-compatible `/metrics` endpoint. |
| FR-OBS-03 | Must | The system SHALL produce structured JSON logs for all key events (ingestion start/complete, query start/complete, errors). |
| FR-OBS-04 | Should | The system SHALL produce an audit log of all data access events (documents read, queries executed, user identity). |
| FR-OBS-05 | Should | The system SHALL expose a health check endpoint (`GET /health`) returning component-level status (vector DB, graph DB, LLM connectivity). |

---

## 8. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-01 | **Language** | The SDK SHALL be implemented in Python ≥ 3.10. |
| NFR-02 | **Performance** | The query API SHALL return a first token within 2 seconds for 95th percentile requests on a standard deployment. |
| NFR-03 | **Scalability** | The API server and ingestion workers SHALL support horizontal scaling via stateless process design. |
| NFR-04 | **Extensibility** | All core subsystems (parsers, processors, storage adapters, LLM providers) SHALL be replaceable via registered adapter interfaces. |
| NFR-05 | **Configurability** | Every system parameter SHALL be configurable via YAML, environment variables, or programmatic API — with clear precedence rules. |
| NFR-06 | **Portability** | The system SHALL run on Linux, macOS, and Windows. Official Docker image SHALL be provided. |
| NFR-07 | **Offline Operation** | All AI model inference SHALL be reroutable to local endpoints. Zero mandatory calls to external APIs during operation. |
| NFR-08 | **Reliability** | Failed ingestion jobs SHALL be retried with configurable exponential backoff. Partial failures SHALL not corrupt the knowledge base. |
| NFR-09 | **Security** | API keys SHALL be stored hashed. All inter-service communication SHALL support TLS. LLM prompt injection SHALL be mitigated by input sanitization. |
| NFR-10 | **Testability** | All core modules SHALL have ≥ 80% unit test coverage. Integration tests SHALL run against local stubs for all external services. |
| NFR-11 | **Developer Experience** | CLI SHALL provide `openrag init`, `openrag ingest`, `openrag query`, and `openrag serve` commands. |
| NFR-12 | **Documentation** | Complete API reference, quickstart guide, YAML pipeline reference, and provider integration guides SHALL be published. |

---

## 9. Detailed Use Cases

### UC-01: Enterprise Legal Knowledge Base

**Actor:** Corporate Legal Team  
**Scenario:** A legal department ingests thousands of contracts, regulatory filings, and court rulings (PDFs, DOCX). Some documents contain embedded tables of legal clauses, diagrams of organizational structures, and financial figures.

**Flow:**
1. Legal ops engineer defines an ingestion pipeline YAML specifying the document parser, table processor, and image processor.
2. Ingestion runs as a background batch job against a document folder, with per-document ACLs mapping to attorney role groups.
3. Attorneys query in natural language: *"Which contracts contain force majeure clauses and what are the termination penalties?"*
4. OpenRAG returns cited excerpts from relevant contracts with clause-level provenance.
5. A structured output query returns the findings as a JSON array for downstream case management system integration.

**Key Features Used:** Batch ingestion, ACLs, structured output queries, citation provenance, REST API.

---

### UC-02: Scientific Research Accelerator

**Actor:** Academic Research Institution  
**Scenario:** A university lab ingests 10,000 research papers across materials science. Papers contain equations, figures, comparative tables, and code appendices.

**Flow:**
1. Papers are batch-ingested with equation, image, table, and code modality processors enabled.
2. Cross-modal entity linking connects an equation in paper A with a figure in paper B that illustrates the same formula.
3. A researcher queries: *"Show me all experimental setups that tested the Nernst equation at temperatures above 500K, including the measurement fixtures shown in the figures."*
4. OpenRAG's VLM integration returns text context plus the relevant images, enabling visual inspection without opening the original PDFs.
5. The researcher exports a structured JSON bibliography of relevant papers.

**Key Features Used:** Equation + image + code processors, cross-modal entity linking, multimodal query, structured output.

---

### UC-03: DevOps AI Assistant

**Actor:** Engineering Platform Team  
**Scenario:** An engineering team ingests their internal wikis, runbooks (Markdown), architecture diagrams (PNG/SVG), and source code repositories. The assistant must answer questions about deployment procedures and code behavior.

**Flow:**
1. Markdown files and code files are ingested with the code modality processor extracting function signatures and semantic intent.
2. Diagrams are captioned by the image processor and linked to related wiki pages via entity linking.
3. Developers query: *"What are the rollback steps for a failed Kubernetes deployment and which service handles the DB migration?"* — the system returns text from the runbook plus the architecture diagram showing the service boundaries.
4. The API is integrated into the team's Slack bot via the REST API.

**Key Features Used:** Code processor, image processor, wiki ingestion, REST API integration, multi-turn conversation.

---

### UC-04: Medical Clinical Decision Support

**Actor:** Healthcare Provider  
**Scenario:** A hospital system ingests clinical guidelines, drug interaction tables, procedure diagrams, and medical imaging reports. Strict access control ensures physicians only see data relevant to their specialty.

**Flow:**
1. Documents are ingested with role-based ACLs: oncologists cannot access psychiatry records by default.
2. A physician queries: *"What are the contraindications for metformin in patients with Stage 3 CKD per the latest ADA guidelines?"*
3. OpenRAG retrieves guidelines, relevant tables (dosing recommendations), and cross-references to drug interaction databases.
4. The response is streamed in real time via WebSocket to the clinical portal.
5. All queries are audit-logged for compliance.

**Key Features Used:** Multi-tenancy with ACLs, streaming query, audit logging, table processor, hybrid retrieval.

---

### UC-05: Media & Entertainment Content Intelligence

**Actor:** Streaming Platform Content Team  
**Scenario:** A media company ingests video transcripts, subtitle files, and thumbnail images from their catalogue. Content analysts need to find scenes, themes, or dialogue matches across thousands of hours of content.

**Flow:**
1. Video files are processed with the audio/video modality processor, generating timestamped transcripts and key-frame image captions.
2. A content analyst queries: *"Find all scenes involving a chase through an urban environment with dialogue referencing law enforcement."*
3. OpenRAG returns matched video segments with timestamps, extracted dialogue snippets, and key-frame images.
4. Results are returned via GraphQL query enabling the front-end to render rich previews directly.

**Key Features Used:** Audio/video processor, image processor, GraphQL query interface, timestamp-scoped retrieval.

---

### UC-06: Air-Gapped Government Intelligence

**Actor:** Government Defense / Intelligence Agency  
**Scenario:** A classified environment has no internet access. All AI inference must run locally. Documents include PDFs, images, and spreadsheets with sensitive information.

**Flow:**
1. OpenRAG is deployed with all inference pointing to on-premises LLM and embedding servers.
2. Documents are batch-ingested locally with all parsing and processing using locally-hosted models.
3. Operators query via the built-in API server on the internal network.
4. All operations are traced via local OpenTelemetry collector and metrics shipped to an internal Prometheus instance.

**Key Features Used:** Offline/air-gapped mode, local inference endpoints, built-in API server, OpenTelemetry, Prometheus.

---

## 10. Competitive Positioning

| Capability | OpenRAG | Typical RAG Library |
|---|---|---|
| Built-in API server | ✅ REST + GraphQL + WebSocket | ❌ Library only |
| Declarative YAML pipelines | ✅ YAML-first | ❌ Code-only |
| Native modalities | Text, Image, Table, Equation, Code, Audio/Video | Text, Image, Table, Equation |
| Multi-tenancy + ACLs | ✅ Namespace isolation + per-doc ACLs | ❌ Single knowledge base |
| Streaming responses | ✅ WebSocket + SSE | Partial |
| Structured output / JSON schema | ✅ | ❌ |
| Function calling during query | ✅ | ❌ |
| OpenTelemetry tracing | ✅ | ❌ |
| Prometheus metrics | ✅ | ❌ |
| Pluggable vector DB adapters | ≥ 5 backends | 1-2 backends |
| BM25 hybrid retrieval | ✅ | Partial |
| Knowledge graph support | ✅ Pluggable KG adapter | Hardcoded |
| Cloud-native deployment | Docker + Helm chart | Manual |
| License | Apache 2.0 | MIT / GPL |

---

## 11. Constraints & Assumptions

| Type | Detail |
|---|---|
| **Constraint** | Audio/video processing requires an external ASR model (e.g., Whisper) to be configured |
| **Constraint** | SVG image processing requires an optional rendering library (e.g., CairoSVG) |
| **Constraint** | Multi-tenant access control requires a persistent relational database (Postgres or SQLite) for ACL storage |
| **Assumption** | Users supply their own LLM, VLM, embedding, and ASR model credentials or local endpoints |
| **Assumption** | Vector database and graph database are deployed separately or via Docker Compose reference config |
| **Assumption** | Production deployments use a reverse proxy (e.g., Nginx, Traefik) in front of the OpenRAG API server |

---

## 12. Success Metrics

| Metric | Target (v1.0) |
|---|---|
| **Time to first query** | New developer produces a working multimodal knowledge base in ≤ 30 minutes |
| **Modality coverage** | All 6 modalities process without error on reference documents |
| **API latency (P95)** | First streaming token ≤ 2s on typical hardware with GPT-4o backend |
| **Ingestion reliability** | ≥ 99% of submitted documents successfully indexed; failures logged with recovery path |
| **Multi-tenant isolation** | Tenant A cannot retrieve documents owned by Tenant B — verified by automated tests |
| **Observability** | OpenTelemetry traces visible in Jaeger; Prometheus metrics scrapeable |
| **Community adoption** | Package available on PyPI; Docker image on Docker Hub; Helm chart on Artifact Hub |
| **Test coverage** | ≥ 80% unit test coverage on core modules |

---

## 13. Glossary

| Term | Definition |
|---|---|
| **Modality** | A specific type of content within a document: text, image, table, equation, code, or audio/video |
| **Modal Processor** | A component responsible for understanding and indexing a specific content modality |
| **Knowledge Graph (KG)** | A graph structure where nodes are entities and edges are typed relationships between them |
| **Hybrid Retrieval** | Combining dense vector similarity search with sparse BM25 keyword retrieval for improved recall |
| **Cross-Modal Entity Linking** | Associating the same semantic entity across different content modalities |
| **Namespace** | An isolated partition of the knowledge base corresponding to a tenant, project, or user group |
| **ACL** | Access Control List — a policy specifying which identities may access which resources |
| **DAG** | Directed Acyclic Graph — used to represent processing pipeline steps and their dependencies |
| **VLM** | Vision Language Model — an AI model capable of understanding both images and text |
| **HyDE** | Hypothetical Document Embedding — a query expansion technique where the LLM generates a hypothetical answer to improve retrieval |
| **ASR** | Automatic Speech Recognition — converts audio to text |
| **SSE** | Server-Sent Events — a protocol for streaming real-time data from server to browser |
