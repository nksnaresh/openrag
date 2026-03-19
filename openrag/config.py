"""
OpenRAG Configuration System.

All settings are defined in OpenRAGConfig and its nested sub-models.
Settings can be loaded from:
  1. A YAML pipeline file  →  OpenRAGConfig.from_yaml("pipeline.yaml")
  2. Environment variables →  OPENRAG_NAMESPACE=my-ns  (with __ for nesting)
  3. Programmatic API     →  OpenRAGConfig(namespace="my-ns", ...)

Precedence (highest → lowest): env vars > YAML > defaults.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Sub-configurations ─────────────────────────────────────────────────────────

class ParserConfig(BaseSettings):
    """Parser selection per file type."""

    pdf: str = "pymupdf"
    docx: str = "docling"
    pptx: str = "docling"
    xlsx: str = "docling"
    audio: str = "whisper"
    video: str = "ffmpeg"
    web: str = "beautifulsoup"
    code: str = "treesitter"

    model_config = SettingsConfigDict(env_prefix="OPENRAG_PARSER_")


class TextProcessorConfig(BaseSettings):
    enabled: bool = True
    chunking_strategy: Literal["sentence", "token", "recursive", "semantic"] = "sentence"
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    model_config = SettingsConfigDict(env_prefix="OPENRAG_PROC_TEXT_")


class ImageProcessorConfig(BaseSettings):
    enabled: bool = True
    extract_ocr: bool = True
    bounding_box_entities: bool = True

    model_config = SettingsConfigDict(env_prefix="OPENRAG_PROC_IMAGE_")


class TableProcessorConfig(BaseSettings):
    enabled: bool = True
    formats: list[str] = Field(default_factory=lambda: ["markdown", "html", "csv"])
    extract_numeric_trends: bool = True

    model_config = SettingsConfigDict(env_prefix="OPENRAG_PROC_TABLE_")


class EquationProcessorConfig(BaseSettings):
    enabled: bool = True
    input_format: Literal["latex", "mathml", "auto"] = "auto"
    symbolic_simplify: bool = True

    model_config = SettingsConfigDict(env_prefix="OPENRAG_PROC_EQUATION_")


class CodeProcessorConfig(BaseSettings):
    enabled: bool = True
    languages: list[str] = Field(
        default_factory=lambda: ["python", "javascript", "typescript", "java", "go", "rust", "sql"]
    )
    extract_functions: bool = True
    extract_classes: bool = True

    model_config = SettingsConfigDict(env_prefix="OPENRAG_PROC_CODE_")


class AudioVideoProcessorConfig(BaseSettings):
    enabled: bool = False
    asr_model: str = "base"
    keyframe_interval_seconds: int = 30

    model_config = SettingsConfigDict(env_prefix="OPENRAG_PROC_AV_")


class ProcessorsConfig(BaseSettings):
    text: TextProcessorConfig = Field(default_factory=TextProcessorConfig)
    image: ImageProcessorConfig = Field(default_factory=ImageProcessorConfig)
    table: TableProcessorConfig = Field(default_factory=TableProcessorConfig)
    equation: EquationProcessorConfig = Field(default_factory=EquationProcessorConfig)
    code: CodeProcessorConfig = Field(default_factory=CodeProcessorConfig)
    audio_video: AudioVideoProcessorConfig = Field(default_factory=AudioVideoProcessorConfig)

    model_config = SettingsConfigDict(env_prefix="OPENRAG_PROCESSORS_")


class ContextConfig(BaseSettings):
    strategy: Literal[
        "page_window", "block_window", "section_ancestor", "semantic_neighbour"
    ] = "page_window"
    window_size: int = 2
    max_tokens: int = 1500
    include_captions: bool = True
    filter_types: list[str] = Field(default_factory=lambda: ["text"])

    model_config = SettingsConfigDict(env_prefix="OPENRAG_CONTEXT_")


class EmbeddingConfig(BaseSettings):
    provider: str = "openai"
    model: str = "text-embedding-3-large"
    dimensions: int = 3072
    batch_size: int = 100
    cache_backend: Literal["in_memory", "redis", "sqlite"] = "in_memory"
    cache_url: str | None = None

    model_config = SettingsConfigDict(env_prefix="OPENRAG_EMBEDDING_")


class VectorDBConfig(BaseSettings):
    adapter: str = "in_memory"
    url: str | None = None
    collection: str = "openrag_chunks"
    distance_metric: Literal["cosine", "dot", "euclidean"] = "cosine"
    api_key: str | None = None

    model_config = SettingsConfigDict(env_prefix="OPENRAG_VECTOR_DB_")


class GraphDBConfig(BaseSettings):
    enabled: bool = True
    adapter: str = "networkx"
    url: str | None = None
    username: str | None = None
    password: str | None = None

    model_config = SettingsConfigDict(env_prefix="OPENRAG_GRAPH_DB_")


class DocumentStoreConfig(BaseSettings):
    adapter: str = "sqlite"
    url: str = "sqlite+aiosqlite:///./openrag_store.db"

    model_config = SettingsConfigDict(env_prefix="OPENRAG_DOC_STORE_")


class LLMConfig(BaseSettings):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_output_tokens: int = 2048
    api_key: str | None = Field(default=None)
    base_url: str | None = None

    model_config = SettingsConfigDict(env_prefix="OPENRAG_LLM_")


class RerankConfig(BaseSettings):
    enabled: bool = False
    adapter: Literal["cohere", "huggingface"] = "huggingface"
    model: str = "BAAI/bge-reranker-base"
    top_n: int = 10

    model_config = SettingsConfigDict(env_prefix="OPENRAG_RERANK_")


class RetrievalConfig(BaseSettings):
    default_mode: str = "hybrid"
    top_k: int = 20
    rrf_k: int = 60
    hyde_enabled: bool = False

    model_config = SettingsConfigDict(env_prefix="OPENRAG_RETRIEVAL_")


class APIServerConfig(BaseSettings):
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000
    workers: int = 1
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 60
    auth_mode: Literal["api_key", "jwt", "none"] = "none"
    jwt_secret: str = "change-me-in-production"  # noqa: S105

    model_config = SettingsConfigDict(env_prefix="OPENRAG_API_")


class ObservabilityConfig(BaseSettings):
    tracing_enabled: bool = False
    tracing_exporter: Literal["otlp", "console", "none"] = "none"
    tracing_endpoint: str = "http://localhost:4317"
    metrics_enabled: bool = True
    metrics_port: int = 9090
    audit_log_enabled: bool = False

    model_config = SettingsConfigDict(env_prefix="OPENRAG_OBS_")


class IngestionConfig(BaseSettings):
    max_concurrent_files: int = 4
    retry_attempts: int = 3
    retry_backoff_seconds: float = 2.0
    dedup_enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="OPENRAG_INGEST_")


# ── Root configuration ─────────────────────────────────────────────────────────

class OpenRAGConfig(BaseSettings):
    """
    Master configuration for an OpenRAG instance.

    Load from YAML:
        config = OpenRAGConfig.from_yaml("pipeline.yaml")

    Or from env vars alone:
        config = OpenRAGConfig()   # reads OPENRAG_* env vars
    """

    namespace: str = "default"
    tenant_id: str = "default"
    working_dir: str = "./openrag_storage"

    parsers: ParserConfig = Field(default_factory=ParserConfig)
    processors: ProcessorsConfig = Field(default_factory=ProcessorsConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    graph_db: GraphDBConfig = Field(default_factory=GraphDBConfig)
    document_store: DocumentStoreConfig = Field(default_factory=DocumentStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    api: APIServerConfig = Field(default_factory=APIServerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)

    model_config = SettingsConfigDict(
        env_prefix="OPENRAG_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @field_validator("working_dir")
    @classmethod
    def expand_working_dir(cls, v: str) -> str:
        return str(Path(os.path.expandvars(v)).expanduser().resolve())

    @classmethod
    def from_yaml(cls, path: str | Path) -> OpenRAGConfig:
        """
        Load config from a YAML file with ${ENV_VAR} interpolation.

        Values in the YAML may reference environment variables using
        the ${VAR_NAME} syntax, which are expanded at load time.
        """
        raw = Path(path).read_text(encoding="utf-8")
        # Expand ${ENV_VAR} placeholders
        interpolated = re.sub(
            r"\$\{([^}]+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            raw,
        )
        data: dict[str, Any] = yaml.safe_load(interpolated) or {}
        return cls.model_validate(data)

    def ensure_working_dir(self) -> Path:
        """Create working_dir if it doesn't exist and return as Path."""
        p = Path(self.working_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
