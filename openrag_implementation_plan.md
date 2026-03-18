# OpenRAG — Detailed Technical Implementation Plan

| Field | Detail |
|---|---|
| **Version** | 1.0 |
| **Date** | March 2026 |
| **Total Duration** | ~24 weeks (6 months) |
| **Team Size** | 3–6 engineers |
| **Language** | Python ≥ 3.10 |
| **Reference Docs** | [BRD](./openrag_brd.md) · [TAD](./openrag_tad.md) |

---

## Implementation Roadmap

```
Phase 0  ──► Phase 1  ──► Phase 2  ──► Phase 3  ──► Phase 4  ──► Phase 5  ──► Phase 6  ──► Phase 7/8
Bootstrap   Foundation   Ingestion   Knowledge    Query       API+Auth    Observ/      DX &
& Standards  Layer       Pipeline    Layer        Intelligence Server     Deploy       Release
  Wk 1-2     Wk 3-5      Wk 6-10     Wk 11-14    Wk 15-18   Wk 19-21   Wk 22-23     Wk 24
```

---

## Phase 0: Project Bootstrap & Standards (Weeks 1–2)

### Goals
Establish the project scaffold, toolchain, code quality gates, and shared data contracts that every subsequent phase depends on.

### 0.1 Repository & Toolchain Setup

**Steps:**
1. Initialize git repository with trunk-based branching strategy (`main`, `dev`, `release/*`)
2. Create [pyproject.toml](file:///Users/nareshsingh/MEGA-P/dev/RAG-Anything-main/pyproject.toml) with build config using `hatchling` or `setuptools`:
   ```toml
   [project]
   name = "openrag"
   version = "0.1.0"
   requires-python = ">=3.10"
   dependencies = ["pydantic>=2.0", "pydantic-settings>=2.0", "pyyaml>=6.0", "anyio>=4.0", "structlog>=24.0"]
   ```
3. Configure `uv` as the package and venv manager (`uv sync`, `uv run`)
4. Set up `ruff` (linting + formatting), `mypy` (strict type checking), `pre-commit` hooks
5. Configure `pytest` with `pytest-asyncio`, `pytest-cov`, `pytest-mock`
6. Set up GitHub Actions CI: lint → typecheck → test → coverage report

**Deliverables:**
- [pyproject.toml](file:///Users/nareshsingh/MEGA-P/dev/RAG-Anything-main/pyproject.toml) · [.pre-commit-config.yaml](file:///Users/nareshsingh/MEGA-P/dev/RAG-Anything-main/.pre-commit-config.yaml) · `.github/workflows/ci.yml`

### 0.2 Core Data Contracts

Define all shared dataclasses and enums **before** any module implementation. These are the "lingua franca" of OpenRAG.

**File: `openrag/models/content.py`**
```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

class BlockType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    CODE = "code"
    AUDIO_TRANSCRIPT = "audio_transcript"
    VIDEO_FRAME = "video_frame"

@dataclass
class BoundingBox:
    x0: float; y0: float; x1: float; y1: float; page: int

@dataclass
class ContentBlock:
    block_id: str
    block_type: BlockType
    sequence_index: int
    raw_content: Any
    page_number: Optional[int] = None
    bounding_box: Optional[BoundingBox] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentMeta:
    title: Optional[str] = None
    author: Optional[str] = None
    language: str = "en"
    page_count: Optional[int] = None
    source_url: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentPayload:
    document_id: str           # SHA-256 of file content
    source_path: str
    tenant_id: str
    metadata: DocumentMeta
    blocks: List[ContentBlock] = field(default_factory=list)
```

**File: `openrag/models/processing.py`**
```python
@dataclass
class ProcessingContext:
    payload: ContentPayload
    llm_func: Callable
    vlm_func: Optional[Callable]
    context_config: ContextWindowConfig
    tenant_config: Dict[str, Any]

@dataclass
class ProcessedBlock:
    source_block: ContentBlock
    natural_language_description: str
    embedding_text: str
    entity_candidates: List[EntityCandidate]
    structured_data: Optional[Any] = None
    confidence_score: float = 1.0
```

**File: `openrag/models/query.py`**
```python
class QueryMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    GRAPH = "graph"
    HYBRID = "hybrid"
    MULTIMODAL = "multimodal"

@dataclass
class QueryRequest:
    text: str
    mode: QueryMode = QueryMode.HYBRID
    namespace: str = "default"
    top_k: int = 20
    output_schema: Optional[Dict] = None
    session_id: Optional[str] = None
    stream: bool = False

@dataclass
class Citation:
    document_id: str; document_title: str
    page_number: Optional[int]; block_id: str; block_type: BlockType; score: float

@dataclass
class QueryResponse:
    answer: str
    citations: List[Citation]
    query_mode: QueryMode
    latency_ms: float
    session_id: Optional[str] = None
```

**Acceptance Criteria:** All models importable; mypy strict passes; 100% unit test coverage on models.

---

## Phase 1: Core Foundation (Weeks 3–5)

### Goals
Build the configuration system, adapter registry, and all storage adapters. Everything in later phases depends on this layer being stable.

### 1.1 Configuration System

**File: `openrag/config.py`**

```python
from pydantic_settings import BaseSettings
from pydantic import Field
import yaml

class ParserConfig(BaseModel):
    pdf: str = "pymupdf"
    docx: str = "docling"
    audio: str = "whisper"
    code: str = "treesitter"

class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    model: str = "text-embedding-3-large"
    dimensions: int = 3072
    batch_size: int = 100

class VectorDBConfig(BaseModel):
    adapter: str = "in_memory"
    url: Optional[str] = None
    collection: str = "openrag_chunks"

class GraphDBConfig(BaseModel):
    enabled: bool = True
    adapter: str = "networkx"
    url: Optional[str] = None

class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_output_tokens: int = 2048

class OpenRAGConfig(BaseSettings):
    namespace: str = "default"
    tenant_id: str = "default"
    working_dir: str = "./openrag_storage"
    parsers: ParserConfig = Field(default_factory=ParserConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    graph_db: GraphDBConfig = Field(default_factory=GraphDBConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "OpenRAGConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    model_config = SettingsConfigDict(env_prefix="OPENRAG_", env_nested_delimiter="__")
```

**YAML interpolation:** Support `${ENV_VAR}` via a pre-processor that runs `os.path.expandvars` on the raw YAML string before parsing.

### 1.2 Adapter Registry

**File: `openrag/registry.py`**

```python
class AdapterRegistry:
    """Central registry for all pluggable adapters."""
    _parsers: Dict[str, Type[BaseParserAdapter]] = {}
    _vector_dbs: Dict[str, Type[BaseVectorDBAdapter]] = {}
    _graph_dbs: Dict[str, Type[BaseGraphDBAdapter]] = {}
    _embeddings: Dict[str, Type[BaseEmbeddingAdapter]] = {}
    _processors: Dict[BlockType, Type[BaseModalityProcessor]] = {}

    @classmethod
    def register_parser(cls, mime_or_ext: str, adapter_cls: Type): ...
    @classmethod
    def get_parser(cls, mime_or_ext: str) -> BaseParserAdapter: ...
    # Same pattern for all adapter types
```

Register built-in adapters at package import time in `openrag/__init__.py`.

### 1.3 Storage Adapters

Implement all storage adapters behind their abstract interfaces. Start with the in-process adapters (for testing), then add production adapters.

#### Vector DB

**File: `openrag/storage/vector/base.py`**
```python
class BaseVectorDBAdapter(ABC):
    @abstractmethod
    async def upsert(self, namespace: str, records: List[VectorRecord]) -> None: ...
    @abstractmethod
    async def query(self, namespace: str, vector: np.ndarray, top_k: int,
                    filters: Optional[Dict] = None) -> List[SearchResult]: ...
    @abstractmethod
    async def delete(self, namespace: str, ids: List[str]) -> None: ...
    @abstractmethod
    async def initialize(self) -> None: ...
```

**Adapters to implement (in order):**

| Order | Adapter | File | Notes |
|---|---|---|---|
| 1 | `InMemoryVectorAdapter` | `vector/in_memory.py` | `numpy` dot-product; for dev/test |
| 2 | `QdrantAdapter` | `vector/qdrant.py` | `qdrant-client` async API |
| 3 | `ChromaAdapter` | `vector/chroma.py` | `chromadb` embedded mode |
| 4 | `PgVectorAdapter` | `vector/pgvector.py` | `asyncpg` + pgvector extension |
| 5 | `WeaviateAdapter` | `vector/weaviate.py` | `weaviate-client` v4 |

#### Graph DB

| Order | Adapter | File |
|---|---|---|
| 1 | `NetworkXAdapter` | `graph/networkx.py` |
| 2 | `Neo4jAdapter` | `graph/neo4j.py` |
| 3 | `ArangoDBAdapter` | `graph/arangodb.py` |

#### Document Store

| Order | Adapter | File |
|---|---|---|
| 1 | `SQLiteDocumentAdapter` | `document/sqlite.py` |
| 2 | `PostgreSQLDocumentAdapter` | `document/postgresql.py` |
| 3 | `MongoDBDocumentAdapter` | `document/mongodb.py` |

**Acceptance Criteria:** All adapters pass a shared `AdapterContractTestSuite` that runs identical read/write/delete/query operations against each implementation.

---

## Phase 2: Ingestion Pipeline (Weeks 6–10)

### Goals
Build the full document ingestion path: parsers → DAG engine → modality processors → context enricher.

### 2.1 Parser Adapters

**Base interface: `openrag/parsers/base.py`**
```python
class BaseParserAdapter(ABC):
    @abstractmethod
    async def parse(self, file_path: str, options: ParseOptions) -> ContentPayload: ...
    @abstractmethod
    def supported_extensions(self) -> List[str]: ...
    @classmethod
    def health_check(cls) -> bool: return True
```

**Implementation order:**

| Week | Parser | Library | Key Logic |
|---|---|---|---|
| 6 | `PyMuPDFAdapter` | `pymupdf` | Extract text spans, image bytes, table rects, page numbers |
| 6 | `DoclingAdapter` | `docling` | DOCX/PPTX/HTML → ContentPayload |
| 7 | `HTMLAdapter` | `beautifulsoup4` + `httpx` | Fetch URL, strip nav/footer, extract article body |
| 7 | `CodeAdapter` | `tree-sitter` | Per-language grammar, extract functions/classes as `CODE` blocks |
| 8 | `WhisperAdapter` | `openai-whisper` | Audio → timestamped transcript segments |
| 8 | `FFmpegAdapter` | `ffmpeg-python` | Video → transcript + key frames at N-second intervals |

**Deduplication (`openrag/ingestion/deduplicator.py`):**
```python
async def is_duplicate(document_hash: str, namespace: str, doc_store: BaseDocumentAdapter) -> bool:
    existing = await doc_store.find_by_hash(namespace, document_hash)
    return existing is not None
```
Content hash = `SHA-256(file_bytes)`, stored in document store on first ingest.

### 2.2 DAG Pipeline Engine

**File: `openrag/pipeline/dag_engine.py`**

```python
@dataclass
class PipelineStage:
    name: str
    processor: BaseModalityProcessor
    depends_on: List[str] = field(default_factory=list)
    timeout_seconds: float = 120.0
    block_types: Optional[List[BlockType]] = None  # None = all types

class DAGPipelineEngine:
    def __init__(self, stages: List[PipelineStage], config: OpenRAGConfig): ...

    async def execute(self, payload: ContentPayload, context: ProcessingContext
                      ) -> List[ProcessedBlock]:
        """
        1. Topological sort of stages by depends_on
        2. Group independent stages → execute concurrently via anyio.create_task_group
        3. Pass outputs of dependency stages as inputs to dependent stages
        4. Collect all ProcessedBlocks
        """
```

**DAG construction from YAML:**
```python
def build_from_config(config: OpenRAGConfig) -> DAGPipelineEngine:
    stages = []
    if config.processors.text.enabled:
        stages.append(PipelineStage("context_enrich", ContextEnrichStage(), []))
        stages.append(PipelineStage("text", TextProcessor(), ["context_enrich"],
                                    block_types=[BlockType.TEXT]))
    if config.processors.image.enabled:
        stages.append(PipelineStage("image", ImageProcessor(), ["context_enrich"],
                                    block_types=[BlockType.IMAGE]))
    # ... etc
    return DAGPipelineEngine(stages, config)
```

### 2.3 Context Enricher

**File: `openrag/pipeline/context_enricher.py`**

Implements 4 strategies (per TAD §3.5):

```python
class ContextEnricher:
    strategies = {
        "page_window": PageWindowStrategy,
        "block_window": BlockWindowStrategy,
        "section_ancestor": SectionAncestorStrategy,
        "semantic_neighbour": SemanticNeighbourStrategy,
    }

    def enrich(self, block: ContentBlock, payload: ContentPayload,
               config: ContextWindowConfig) -> str:
        strategy = self.strategies[config.strategy]()
        raw_context = strategy.extract(block, payload, config)
        return self._truncate_to_budget(raw_context, config.max_tokens)
```

### 2.4 Modality Processors

**Base: `openrag/processors/base.py`**
```python
class BaseModalityProcessor(ABC):
    @abstractmethod
    async def process(self, block: ContentBlock, context: ProcessingContext) -> ProcessedBlock: ...
    @abstractmethod
    def supported_block_types(self) -> List[BlockType]: ...
```

**Implementation per processor:**

#### `TextProcessor` (`processors/text_processor.py`)
- Chunking strategies: `sentence` (using `nltk.sent_tokenize`), `token` (tiktoken), `recursive_character`, `semantic` (embed + cluster)
- Each chunk becomes one `ProcessedBlock`; `embedding_text = chunk_text`
- Entity candidates extracted via lightweight spaCy NER (`en_core_web_sm`)

#### `ImageProcessor` (`processors/image_processor.py`)
```python
async def process(self, block, ctx) -> ProcessedBlock:
    img_bytes = block.raw_content  # bytes
    b64 = base64.b64encode(img_bytes).decode()
    surrounding_text = ctx_enricher.enrich(block, ctx.payload, ctx.context_config)
    prompt = IMAGE_PROMPT_TEMPLATE.format(context=surrounding_text)
    response_json = await ctx.vlm_func(prompt, image_b64=b64)
    parsed = self._parse_vlm_response(response_json)
    return ProcessedBlock(
        source_block=block,
        natural_language_description=parsed["description"],
        embedding_text=parsed["description"] + " " + parsed["ocr_text"],
        entity_candidates=parsed["entities"],
        confidence_score=parsed.get("confidence", 0.9),
    )
```

Prompt template (stored in `openrag/prompts/image.py`):
```
You are an expert image analyst. Given this image and surrounding document context:
CONTEXT: {context}
Respond in JSON:
{ "description": "...", "ocr_text": "...", "entities": [{"name":"...","type":"..."}], "confidence": 0.0-1.0 }
```

#### `TableProcessor` (`processors/table_processor.py`)
- Detect format: HTML table → `pandas.read_html`, Markdown → custom parser, CSV → `pandas.read_csv`
- Extract schema: column names, dtypes, row count, null percentage
- Numeric columns: compute min/max/mean as structured metadata
- LLM prompt: table schema + 5 sample rows + caption → natural language summary + domain tags

#### `EquationProcessor` (`processors/equation_processor.py`)
- Parse LaTeX via `sympy.parsing.latex.parse_latex()` → simplified symbolic form
- LLM prompt: LaTeX string + simplified form + surrounding text → English interpretation + variable definitions + domain classification
- Store `latex_source` in `ProcessedBlock.structured_data`

#### `CodeProcessor` (`processors/code_processor.py`)
- `tree-sitter`: parse AST → extract function defs, class defs, import statements, docstrings
- Each function/class = one sub-block
- LLM prompt: function signature + body + docstring → semantic intent + algorithmic category
- Entity candidates: function names (`FUNCTION`), class names (`CLASS`), imported modules (`MODULE`)

#### `AudioVideoProcessor` (`processors/audio_video_processor.py`)
- Audio: `WhisperAdapter.transcribe()` → `List[{"start": float, "end": float, "text": str}]`
- Video: `FFmpegAdapter.extract_frames(interval_sec=30)` → `List[PIL.Image]`; each frame → `ImageProcessor`
- Final `ProcessedBlock` contains transcript segments; frames are processed as child IMAGE blocks

**Acceptance Criteria (Phase 2):**
- Each processor handles its block type end-to-end in isolation
- Integration test: parse a PDF with text + image + table + equation → all blocks produce `ProcessedBlock` with non-empty `natural_language_description`
- DAG correctly runs image/table/equation processors in parallel (verified via timing test)

### 2.5 Ingestion Orchestrator

**File: `openrag/ingestion/orchestrator.py`**

```python
class IngestionOrchestrator:
    def __init__(self, config, registry, dag_engine, doc_store, deduplicator, job_manager): ...

    async def ingest_file(self, path: str, metadata: IngestMetadata,
                          acl: Optional[ACLPolicy] = None) -> JobResult:
        job = await self.job_manager.create(path, metadata.tenant_id)
        try:
            # 1. Hash check
            doc_hash = sha256_file(path)
            if await self.deduplicator.is_duplicate(doc_hash, metadata.namespace):
                return JobResult(job_id=job.id, status="skipped", reason="duplicate")

            # 2. Select parser
            ext = Path(path).suffix.lower()
            parser = self.registry.get_parser(ext)

            # 3. Parse
            payload = await parser.parse(path, ParseOptions.from_config(self.config))
            payload.document_id = doc_hash

            # 4. Attach ACL
            if acl: payload.metadata.custom["acl"] = acl.to_dict()

            # 5. Run DAG pipeline
            context = ProcessingContext.build(payload, self.config)
            processed_blocks = await self.dag_engine.execute(payload, context)

            # 6. Store document record
            await self.doc_store.save_document(payload, metadata.namespace)

            await self.job_manager.complete(job.id)
            return JobResult(job_id=job.id, status="completed", block_count=len(processed_blocks))

        except Exception as e:
            await self.job_manager.fail(job.id, str(e))
            raise

    async def ingest_batch(self, paths, options: BatchOptions) -> BatchJobResult:
        sem = anyio.Semaphore(options.max_workers)
        async def _bounded(path):
            async with sem: return await self.ingest_file(path, options.metadata)
        async with anyio.create_task_group() as tg:
            results = [tg.start_soon(_bounded, p) for p in paths]
        return BatchJobResult(results=results)
```

---

## Phase 3: Knowledge Layer (Weeks 11–14)

### Goals
Build the embedding engine, knowledge graph builder, BM25 indexer, and wire them into the ingestion pipeline.

### 3.1 Embedding Engine

**File: `openrag/embeddings/engine.py`**

```python
class EmbeddingEngine:
    def __init__(self, adapter: BaseEmbeddingAdapter, cache: EmbeddingCache,
                 batch_size: int = 100): ...

    async def embed_blocks(self, blocks: List[ProcessedBlock],
                           namespace: str) -> List[EmbeddedBlock]:
        # 1. Check cache by content hash
        uncached = [b for b in blocks if not await self.cache.get(hash(b.embedding_text))]
        # 2. Batch uncached
        for batch in chunks(uncached, self.batch_size):
            texts = [b.embedding_text for b in batch]
            vectors = await self.adapter.embed(texts)
            for block, vector in zip(batch, vectors):
                await self.cache.set(hash(block.embedding_text), vector)
        # 3. Assemble EmbeddedBlock list
        ...

    async def embed_query(self, text: str) -> np.ndarray:
        cached = await self.cache.get(hash(text))
        if cached: return cached
        vector = await self.adapter.embed([text])
        await self.cache.set(hash(text), vector[0])
        return vector[0]
```

**Embedding Adapters:**

| Adapter | Key Details |
|---|---|
| `OpenAIEmbeddingAdapter` | `AsyncOpenAI.embeddings.create()`; handles rate limit with `tenacity` retry |
| `CohereEmbeddingAdapter` | `cohere.AsyncClient.embed()`; supports `input_type=search_document / search_query` |
| `HuggingFaceEmbeddingAdapter` | `sentence-transformers`; runs in `asyncio.get_event_loop().run_in_executor()` |
| `OllamaEmbeddingAdapter` | `httpx.AsyncClient` to local Ollama REST endpoint |

**Embedding Cache:** Redis adapter using `{namespace}:emb:{content_hash}` key; in-memory [dict](file:///Users/nareshsingh/MEGA-P/dev/RAG-Anything-main/raganything/modalprocessors.py#238-264) for dev.

### 3.2 Knowledge Graph Builder

**File: `openrag/knowledge/graph_builder.py`**

```python
class KnowledgeGraphBuilder:
    def __init__(self, graph_adapter: BaseGraphDBAdapter, llm_func: Callable,
                 entity_extractor: EntityExtractor, rel_extractor: RelationshipExtractor,
                 cross_modal_linker: CrossModalLinker): ...

    async def build_from_blocks(self, blocks: List[ProcessedBlock],
                                payload: ContentPayload, namespace: str) -> GraphBuildResult:
        # 1. Upsert Document node
        await self.graph.upsert_node(DocumentNode(id=payload.document_id, ...))

        # 2. Upsert Section nodes from document hierarchy
        sections = self._extract_sections(payload)
        for section in sections:
            await self.graph.upsert_node(SectionNode(...))
            await self.graph.upsert_edge(Edge(payload.document_id, section.id, "CONTAINS"))

        # 3. For each block: upsert Chunk node
        for block in blocks:
            chunk_node = ChunkNode.from_processed_block(block, namespace)
            await self.graph.upsert_node(chunk_node)
            await self.graph.upsert_edge(Edge(section_id, chunk_node.id, "CONTAINS"))
            if prev_chunk: await self.graph.upsert_edge(Edge(prev_chunk.id, chunk_node.id, "NEXT"))

        # 4. Entity extraction + dedup
        all_entities = [e for b in blocks for e in b.entity_candidates]
        resolved = await self.entity_extractor.resolve(all_entities, namespace)
        for entity in resolved:
            await self.graph.upsert_node(EntityNode(**entity))
            await self.graph.upsert_edge(Edge(chunk_id, entity.id, typed_edge(block.block_type)))

        # 5. Cross-modal linking
        await self.cross_modal_linker.link(resolved, blocks, namespace)
```

**Entity Extractor (`openrag/knowledge/entity_extractor.py`):**
- Stage 1: Fast spaCy NER (`PERSON`, `ORG`, `GPE`, `PRODUCT`)
- Stage 2: LLM refinement for domain entities (`ALGORITHM`, `DATASET`, `EQUATION`, `CONCEPT`)
- Deduplication: fuzzy match on canonical name using `rapidfuzz`

**Relationship Extractor (`openrag/knowledge/relationship_extractor.py`):**
- LLM prompt: entity pair + sentence context → typed relationship
- Supported types: `REFERENCES`, `DEFINES`, `IMPLEMENTS`, `CITES`, `PROVES`, `ILLUSTRATES`

**Cross-Modal Linker (`openrag/knowledge/cross_modal_linker.py`):**
- For each entity in a text block, search for same entity name in image/table/equation block entity candidates
- If match found, create `ILLUSTRATES` / `PROVES` / `CONTAINS_DATA_ABOUT` edge

### 3.3 BM25 Indexer

**File: `openrag/search/bm25_indexer.py`**

```python
class BM25Indexer:
    def __init__(self, doc_store: BaseDocumentAdapter): ...

    async def index_blocks(self, blocks: List[ProcessedBlock], namespace: str) -> None:
        for block in blocks:
            tokens = self._tokenize(block.embedding_text)
            await self.doc_store.save_bm25_tokens(namespace, block.source_block.block_id, tokens)

    async def search(self, namespace: str, query_tokens: List[str], top_k: int) -> List[SearchResult]:
        corpus_tokens = await self.doc_store.get_all_bm25_tokens(namespace)
        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query_tokens)
        return self._top_k_results(scores, top_k)
```

For production: delegate to **Elasticsearch** or **OpenSearch** via an adapter that wraps their REST API.

### 3.4 Wiring Phase 3 into Ingestion

Update `IngestionOrchestrator.ingest_file()` to run the three indexing steps **in parallel** after the DAG pipeline:

```python
async with anyio.create_task_group() as tg:
    tg.start_soon(self.embedding_engine.embed_blocks, processed_blocks, namespace)
    tg.start_soon(self.kg_builder.build_from_blocks, processed_blocks, payload, namespace)
    tg.start_soon(self.bm25_indexer.index_blocks, processed_blocks, namespace)
```

**Acceptance Criteria (Phase 3):**
- Ingest a mixed PDF → 3 storage systems populated: vector DB has chunk embeddings, graph DB has nodes/edges, doc store has BM25 tokens
- Embedding cache: second ingest of same document uses 0 embedding API calls

---

## Phase 4: Query Intelligence (Weeks 15–18)

### Goals
Build the complete query path: rewriting → retrieval → re-ranking → synthesis → citation.

### 4.1 Query Rewriter

**File: `openrag/query/rewriter.py`**

```python
class QueryRewriter:
    async def rewrite(self, request: QueryRequest, llm_func: Callable) -> QueryRequest:
        if request.mode == QueryMode.DENSE and self.config.query.hyde_enabled:
            hypothesis = await llm_func(HYDE_PROMPT.format(query=request.text))
            return replace(request, text=hypothesis)
        # Expansion: synonym injection via WordNet for sparse mode
        if request.mode == QueryMode.SPARSE:
            expanded = self._expand_with_synonyms(request.text)
            return replace(request, text=expanded)
        return request
```

### 4.2 Multi-Hop Decomposer

**File: `openrag/query/decomposer.py`**

```python
class MultiHopDecomposer:
    async def decompose(self, request: QueryRequest, llm_func) -> List[QueryRequest]:
        classification = await llm_func(COMPLEXITY_CLASSIFIER_PROMPT.format(q=request.text))
        if classification["is_complex"]:
            sub_queries = await llm_func(DECOMPOSE_PROMPT.format(q=request.text))
            return [replace(request, text=sq) for sq in sub_queries["sub_queries"]]
        return [request]   # Single hop
```

### 4.3 Hybrid Retriever

**File: `openrag/query/retriever.py`**

```python
class HybridRetriever:
    async def retrieve(self, request: QueryRequest) -> List[RetrievedChunk]:
        # Run all applicable retrievers in parallel
        tasks = {}
        if request.mode in (DENSE, HYBRID, MULTIMODAL):
            tasks["dense"] = self._dense_retrieve(request)
        if request.mode in (SPARSE, HYBRID):
            tasks["sparse"] = self._sparse_retrieve(request)
        if request.mode in (GRAPH, HYBRID, MULTIMODAL):
            tasks["graph"] = self._graph_retrieve(request)

        results = {}
        async with anyio.create_task_group() as tg:
            for name, coro in tasks.items():
                tg.start_soon(self._run_and_store, name, coro, results)

        # ACL filter on all results
        filtered = {k: self._apply_acl_filter(v, request) for k, v in results.items()}

        # RRF Fusion
        return self._rrf_fusion(filtered, k=60)

    async def _dense_retrieve(self, request: QueryRequest) -> List[SearchResult]:
        query_vector = await self.embed_engine.embed_query(request.text)
        return await self.vector_db.query(request.namespace, query_vector, request.top_k,
                                          filters=self._acl_filters(request))

    async def _sparse_retrieve(self, request: QueryRequest) -> List[SearchResult]:
        tokens = self.tokenize(request.text)
        return await self.bm25_indexer.search(request.namespace, tokens, request.top_k)

    async def _graph_retrieve(self, request: QueryRequest) -> List[SearchResult]:
        entities = await self.entity_extractor.extract_from_query(request.text)
        chunks = []
        for entity in entities[:5]:   # Anchor on top 5 entities
            subgraph = await self.graph_db.traverse(entity.id, depth=2,
                                                     edge_types=["REFERENCES","ILLUSTRATES"])
            chunks.extend(self._subgraph_to_chunks(subgraph))
        return chunks

    def _rrf_fusion(self, results_by_retriever: Dict, k: int = 60) -> List[RetrievedChunk]:
        scores: Dict[str, float] = defaultdict(float)
        for retriever_results in results_by_retriever.values():
            for rank, result in enumerate(retriever_results):
                scores[result.chunk_id] += 1.0 / (k + rank + 1)
        return sorted(all_chunks, key=lambda c: scores[c.chunk_id], reverse=True)
```

### 4.4 Re-Ranker

**File: `openrag/query/reranker.py`**

```python
class CrossEncoderReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)   # sentence-transformers

    async def rerank(self, query: str, chunks: List[RetrievedChunk],
                     top_n: int = 10) -> List[RetrievedChunk]:
        pairs = [(query, chunk.content) for chunk in chunks]
        scores = await asyncio.get_event_loop().run_in_executor(
            None, self.model.predict, pairs
        )
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_n]]
```

### 4.5 Answer Synthesizer

**File: `openrag/query/synthesizer.py`**

```python
class AnswerSynthesizer:
    async def synthesize(self, request: QueryRequest, context_chunks: List[RetrievedChunk],
                         session_memory: Optional[str] = None) -> SynthesisResult:
        context_str = self._assemble_context(context_chunks)
        prompt = SYNTHESIS_PROMPT.format(
            query=request.text,
            context=context_str,
            memory=session_memory or "",
            output_schema=json.dumps(request.output_schema) if request.output_schema else "none"
        )

        if request.output_schema:
            return await self._synthesize_structured(prompt, request.output_schema)
        elif self.config.llm.tools_enabled:
            return await self._synthesize_with_tools(prompt, request)
        else:
            answer = await self.llm_func(prompt)
            return SynthesisResult(answer=answer, chunks_used=context_chunks)

    async def _synthesize_structured(self, prompt, schema, max_retries=3) -> SynthesisResult:
        for attempt in range(max_retries):
            raw = await self.llm_func(prompt + STRUCTURED_OUTPUT_SUFFIX.format(schema=schema))
            try:
                parsed = json.loads(raw); jsonschema.validate(parsed, schema)
                return SynthesisResult(answer=json.dumps(parsed), structured=parsed, ...)
            except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                prompt += f"\nValidation error: {e}. Please fix and retry."
        raise SynthesisError("Structured output failed after max retries")
```

### 4.6 Citation Formatter + Output

**File: `openrag/query/formatter.py`**

```python
class OutputFormatter:
    def format(self, synthesis: SynthesisResult, request: QueryRequest) -> QueryResponse:
        citations = [
            Citation(
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                page_number=chunk.page_number,
                block_id=chunk.block_id,
                block_type=chunk.block_type,
                score=chunk.retrieval_score,
            )
            for chunk in synthesis.chunks_used
        ]
        return QueryResponse(answer=synthesis.answer, citations=citations,
                             query_mode=request.mode, latency_ms=..., session_id=request.session_id)
```

### 4.7 Conversational Session Manager

**File: `openrag/query/session.py`**

```python
class SessionMemoryManager:
    def __init__(self, store: BaseDocumentAdapter, max_turns: int = 10,
                 summarize_after: int = 8): ...

    async def get_context(self, session_id: str, llm_func: Callable) -> str:
        turns = await self.store.get_session(session_id)
        if len(turns) > self.summarize_after:
            summary = await llm_func(SUMMARIZE_PROMPT.format(turns=turns[:-2]))
            await self.store.replace_session(session_id, [{"role": "summary", "content": summary}] + turns[-2:])
            turns = await self.store.get_session(session_id)
        return self._format_turns(turns)

    async def add_turn(self, session_id: str, user_query: str, assistant_answer: str): ...
```

**Acceptance Criteria (Phase 4):**
- Query returns cited answer with ≥ 1 citation referencing an ingested document
- Structured output query: response JSON validates against provided schema
- Hybrid retriever test: result set contains contributions from >1 retriever
- Multi-turn test: second query in session correctly refers back to first turn's context

---

## Phase 5: API Server + Auth + Multi-Tenancy (Weeks 19–21)

### Goals
Expose all ingestion and query capabilities via REST and GraphQL APIs with full authentication and namespace isolation.

### 5.1 FastAPI Application

**File: `openrag/server/app.py`**

```python
def create_app(config: OpenRAGConfig) -> FastAPI:
    app = FastAPI(title="OpenRAG API", version="1.0.0",
                  docs_url="/docs", redoc_url="/redoc")
    app.add_middleware(CORSMiddleware, allow_origins=config.api.cors_origins, ...)
    app.include_router(ingest_router, prefix="/v1")
    app.include_router(query_router, prefix="/v1")
    app.include_router(namespace_router, prefix="/v1")
    app.include_router(health_router)
    app.include_router(metrics_router)
    app.add_route("/graphql", GraphQLApp(schema=strawberry_schema))
    return app
```

**Key endpoints and their handlers:**

| Endpoint | Handler function | Logic |
|---|---|---|
| `POST /v1/ingest` | `handle_ingest_file` | Validate auth → parse `IngestRequest` → call `orchestrator.ingest_file()` → return `JobStatus` |
| `GET /v1/jobs/{id}` | `handle_get_job` | `job_manager.get_status(id)` with tenant ownership check |
| `POST /v1/query` | `handle_query` | Validate auth → build `QueryRequest` → `query_orchestrator.execute()` → return `QueryResponse` |
| `GET /v1/query/stream` | `handle_query_stream` | Same but returns `EventSourceResponse` with token streaming |
| `WS /v1/ws/query` | `handle_ws_query` | Accept WebSocket → receive JSON → stream tokens → send `done` event |

### 5.2 Authentication Engine

**File: `openrag/auth/engine.py`**

```python
class AuthEngine:
    async def resolve_identity(self, request: Request) -> Identity:
        # Try JWT Bearer first
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return await self._resolve_jwt(auth_header[7:])
        # Fall back to API Key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return await self._resolve_api_key(api_key)
        raise HTTPException(status_code=401, detail="Authentication required")

    async def _resolve_jwt(self, token: str) -> Identity:
        payload = jose.jwt.decode(token, self.config.auth.jwt_secret, algorithms=["HS256"])
        return Identity(user_id=payload["sub"], tenant_id=payload["tenant_id"],
                        roles=payload.get("roles", []))

    async def _resolve_api_key(self, key: str) -> Identity:
        hashed = hashlib.sha256(key.encode()).hexdigest()
        record = await self.api_key_store.find_by_hash(hashed)
        if not record or record.revoked: raise HTTPException(401)
        return Identity(**record.identity)
```

**FastAPI dependency injection:**
```python
async def get_identity(request: Request, auth: AuthEngine = Depends(get_auth_engine)) -> Identity:
    return await auth.resolve_identity(request)

@router.post("/query")
async def handle_query(body: QueryRequestBody, identity: Identity = Depends(get_identity)):
    request = QueryRequest(text=body.text, namespace=identity.tenant_id, ...)
    return await query_orchestrator.execute(request)
```

### 5.3 ACL Filter

**File: `openrag/auth/acl.py`**

```python
class ACLFilter:
    def build_vector_filter(self, identity: Identity) -> Dict:
        """Returns a filter dict passed to the vector DB adapter."""
        return {
            "must": [
                {"match": {"tenant_id": identity.tenant_id}},
                {"should": [
                    {"match": {"acl.read": f"role:{r}"}} for r in identity.roles
                ] + [{"match": {"acl.read": f"user:{identity.user_id}"}}]}
            ]
        }
```

Applied inside `HybridRetriever._dense_retrieve()` / `_sparse_retrieve()` / `_graph_retrieve()`.

### 5.4 Streaming Responses

**SSE (Server-Sent Events):**
```python
from sse_starlette.sse import EventSourceResponse

@router.get("/query/stream")
async def handle_query_stream(q: str, ns: str, identity: Identity = Depends(get_identity)):
    async def token_generator():
        async for token in query_orchestrator.stream(QueryRequest(text=q, namespace=ns)):
            yield {"data": json.dumps({"token": token})}
        yield {"data": json.dumps({"done": True, "citations": [...]})}
    return EventSourceResponse(token_generator())
```

**WebSocket:**
```python
@app.websocket("/v1/ws/query")
async def ws_query(ws: WebSocket, identity: Identity = Depends(get_ws_identity)):
    await ws.accept()
    while True:
        data = await ws.receive_json()
        request = QueryRequest(**data, namespace=identity.tenant_id)
        async for token in query_orchestrator.stream(request):
            await ws.send_json({"type": "token", "token": token})
        await ws.send_json({"type": "done", "citations": [...]})
```

### 5.5 GraphQL Schema

**File: `openrag/server/graphql/schema.py`** (using Strawberry)

```python
@strawberry.type
class Query:
    @strawberry.field
    async def query(self, text: str, mode: QueryMode = QueryMode.HYBRID,
                    namespace: str = "default", top_k: int = 10,
                    info: Info) -> QueryResponseType:
        identity = info.context["identity"]
        request = QueryRequest(text=text, mode=mode, namespace=identity.tenant_id, top_k=top_k)
        return await query_orchestrator.execute(request)

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def ingest(self, input: IngestInput, info: Info) -> JobType:
        identity = info.context["identity"]
        return await orchestrator.ingest_file(input.path, IngestMetadata(tenant_id=identity.tenant_id))

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def query_stream(self, request_id: str) -> AsyncGenerator[QueryTokenType, None]:
        async for token in query_orchestrator.get_stream(request_id):
            yield QueryTokenType(token=token)
```

**Acceptance Criteria (Phase 5):**
- `openrag serve` launches API; `curl /health` returns `{"status": "ok"}`
- JWT auth: valid token → 200; invalid token → 401
- ACL: Tenant A API key cannot retrieve Tenant B documents (verified by test)
- WebSocket streaming: client receives incremental tokens

---

## Phase 6: Observability & Deployment (Weeks 22–23)

### Goals
Instrument every pipeline stage with OpenTelemetry traces, emit Prometheus metrics, add Docker and Helm deployment configs.

### 6.1 OpenTelemetry Integration

**File: `openrag/observability/tracing.py`**

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def setup_tracing(endpoint: str) -> None:
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)

tracer = trace.get_tracer("openrag")

# Decorator for automatic span creation:
def traced(span_name: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("openrag.namespace", kwargs.get("namespace", ""))
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e); span.set_status(StatusCode.ERROR)
                    raise
        return wrapper
    return decorator
```

Apply `@traced("openrag.ingest.parse")` on all pipeline stage entry points.

### 6.2 Prometheus Metrics

**File: `openrag/observability/metrics.py`**

```python
from prometheus_client import Counter, Histogram, generate_latest

INGEST_DOCS = Counter("openrag_ingest_documents_total", "Documents ingested",
                       ["tenant", "parser", "status"])
INGEST_BLOCKS = Counter("openrag_ingest_blocks_total", "Blocks processed",
                         ["tenant", "block_type", "status"])
QUERY_REQUESTS = Counter("openrag_query_requests_total", "Queries executed",
                          ["tenant", "mode", "status"])
QUERY_LATENCY = Histogram("openrag_query_latency_seconds", "Query latency",
                           ["tenant", "mode"], buckets=[.1,.25,.5,1,2.5,5,10])
LLM_TOKENS = Counter("openrag_llm_tokens_total", "LLM tokens consumed",
                      ["tenant", "model", "token_type"])

@metrics_router.get("/metrics")
async def metrics_endpoint():
    return Response(generate_latest(), media_type="text/plain")
```

### 6.3 Structured Audit Logger

```python
import structlog

audit_log = structlog.get_logger("openrag.audit")

async def log_query(identity: Identity, request: QueryRequest,
                    response: QueryResponse, docs_accessed: List[str]) -> None:
    audit_log.info("query.executed",
        tenant_id=identity.tenant_id, user_id=identity.user_id,
        query_hash=sha256(request.text), mode=request.mode.value,
        documents_accessed=docs_accessed, latency_ms=response.latency_ms)
```

### 6.4 Docker Configuration

**`docker/Dockerfile`:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev
COPY openrag/ ./openrag/
EXPOSE 8000 9090
ENTRYPOINT ["uv", "run", "openrag"]
CMD ["serve", "--config", "/config/pipeline.yaml"]
```

**`docker/docker-compose.yaml`** — defines services: `openrag-api`, `openrag-worker`, `qdrant`, `neo4j`, `postgres`, `redis`, `otel-collector`, `prometheus`, `grafana`.

### 6.5 Kubernetes Helm Chart

`helm/openrag/` with:
- `templates/deployment-api.yaml` — `openrag-api` Deployment with HPA (scale on CPU ≥ 70%)
- `templates/deployment-worker.yaml` — `openrag-worker` Deployment (scale on Redis queue depth)
- `templates/service.yaml` — ClusterIP + optional LoadBalancer
- `templates/ingress.yaml` — NGINX Ingress with TLS
- `templates/configmap.yaml` — mounts `pipeline.yaml`
- `templates/secret.yaml` — API keys, DB credentials
- `templates/servicemonitor.yaml` — Prometheus scraping
- `values.yaml` — all defaults; `values.prod.yaml` overrides

---

## Phase 7: CLI & Developer Experience (Week 24a)

### Goals
Ship a polished CLI and comprehensive quickstart.

### 7.1 CLI (`openrag/cli/main.py`)

Using `typer`:

```
openrag init [--config pipeline.yaml]
    → Generates pipeline.yaml scaffold + .env.example + README

openrag check [--config pipeline.yaml]
    → Validates config, checks all adapter health endpoints

openrag ingest <path> [--namespace ns] [--workers 4]
    → Calls IngestionOrchestrator; streams progress to stdout

openrag query "<text>" [--mode hybrid] [--namespace ns] [--stream]
    → Calls QueryOrchestrator; prints answer + citations

openrag serve [--config pipeline.yaml] [--port 8000] [--workers 4] [--dev]
    → Starts FastAPI + Uvicorn; --dev uses in-memory adapters

openrag namespace create <name>
openrag namespace list
openrag namespace delete <name>

openrag job status <job_id>
openrag job cancel <job_id>
```

### 7.2 Developer Quickstart (target: 5 minutes)

```bash
pip install openrag
openrag init            # creates pipeline.yaml
export OPENAI_API_KEY=sk-...
openrag serve --dev     # in-memory mode, no external DBs needed
# In another terminal:
openrag ingest ./my_docs/ --namespace demo
openrag query "What are the main topics?" --namespace demo
```

---

## Phase 8: Testing, Documentation & v0.1.0 Release (Week 24b)

### 8.1 Test Strategy

| Layer | Framework | Coverage Target |
|---|---|---|
| Unit tests | `pytest` + `pytest-asyncio` | ≥ 80% on all modules |
| Integration tests | `pytest` + Docker fixtures | All adapters, full ingestion flow |
| Contract tests | `AdapterContractTestSuite` | All storage adapters pass identical suite |
| End-to-end tests | `pytest` + live API calls | Ingest PDF → query → verify citations |
| Security tests | `bandit`, manual ACL cross-tenant test | All ACL boundaries verified |
| Performance tests | `locust` | P95 query latency ≤ 2s at 50 concurrent users |

### 8.2 Documentation Plan

| Doc | Tool | Location |
|---|---|---|
| API Reference | Swagger auto-gen | `/docs` (live) |
| GraphQL Reference | Strawberry introspection | `/graphql` (live) |
| Quickstart Guide | MkDocs Material | `docs/quickstart.md` |
| YAML Pipeline Reference | MkDocs Material | `docs/pipeline_config.md` |
| Provider Guides | MkDocs Material | `docs/providers/openai.md`, etc. |
| Adapter Development Guide | MkDocs Material | `docs/extending/adapters.md` |
| Deployment Guide | MkDocs Material | `docs/deployment/docker.md`, `kubernetes.md` |

### 8.3 Release Checklist (v0.1.0)

- [ ] All Phase 0–7 acceptance criteria met
- [ ] `mypy --strict` passes with zero errors
- [ ] `ruff check` passes with zero warnings
- [ ] Test coverage ≥ 80% verified in CI
- [ ] `pip install openrag` → `openrag init` → `openrag serve --dev` → ingest + query works
- [ ] Docker image built and pushed to Docker Hub (`openrag/openrag:0.1.0`)
- [ ] Helm chart lints clean (`helm lint`)
- [ ] PyPI release via GitHub Actions on `release/0.1.0` tag
- [ ] GitHub release with changelog
- [ ] OpenAPI spec exported to `docs/openapi.json`

---

## Appendix A: Dependency Table

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | ≥ 0.110 | REST API framework |
| `uvicorn[standard]` | ≥ 0.29 | ASGI server |
| `strawberry-graphql` | ≥ 0.223 | GraphQL server |
| `pydantic` | ≥ 2.6 | Data validation |
| `pydantic-settings` | ≥ 2.2 | Config + env var |
| `anyio` | ≥ 4.3 | Async task groups |
| `structlog` | ≥ 24.0 | Structured logging |
| `opentelemetry-sdk` | ≥ 1.24 | Distributed tracing |
| `prometheus-client` | ≥ 0.20 | Metrics |
| `python-jose` | ≥ 3.3 | JWT |
| `passlib` | ≥ 1.7 | Password hashing |
| `tenacity` | ≥ 8.2 | Retry logic |
| `typer` | ≥ 0.12 | CLI |
| `pymupdf` | ≥ 1.24 | PDF parsing |
| `docling` | latest | Office doc parsing |
| `tree-sitter` | ≥ 0.22 | Code AST parsing |
| `openai-whisper` | latest | Audio transcription |
| `sentence-transformers` | ≥ 2.7 | HF embeddings + reranker |
| `sympy` | ≥ 1.12 | Equation parsing |
| `rank-bm25` | ≥ 0.2 | BM25 indexing |
| `qdrant-client` | ≥ 1.9 | Qdrant vector DB |
| `neo4j` | ≥ 5.19 | Neo4j graph DB |
| `asyncpg` | ≥ 0.29 | PostgreSQL async |
| `rapidfuzz` | ≥ 3.8 | Entity deduplication |
| `jsonschema` | ≥ 4.22 | Structured output validation |

---

## Appendix B: Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| LLM API rate limits slow ingestion | High | Medium | Exponential backoff (`tenacity`); batch size tuning; local model fallback |
| VLM response parsing failures | Medium | Medium | 4-strategy JSON fallback parser; regex extraction as last resort |
| Graph DB performance at scale | Medium | High | Introduce graph materialized views; cap traversal depth; add result caching |
| Multi-tenant ACL bypass | Low | Critical | Property-based security tests; penetration testing; independent security audit |
| Audio/video processing latency | High | Low | Process asynchronously as a background job; return partial results |
| Embedding model deprecation | Low | Medium | Abstracted adapter — swap model in config YAML, no code change |
