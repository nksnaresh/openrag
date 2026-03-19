# Component Architecture: REST API Layer

The REST API Layer is the gateway for all external interactions with the OpenRAG framework. It is built using **FastAPI** for high performance and standard compliance (OpenAPI/Swagger).

## 1. Core Structure
The API is located in `openrag/api/` and is divided into three main files:
- `app.py`: Application initialization, global configuration, and lifespan management.
- `routes.py`: Endpoint definitions and request/response orchestration.
- `models.py`: Pydantic schemas for data validation and API documentation.

## 2. Key Endpoints

### 📥 Ingestion
- `POST /v1/ingest`: Accepts a single file for processing into a namespace.
- `POST /v1/ingest/batch`: (Future) For high-throughput background ingestion.

### 🔍 Query
- `POST /v1/query`: The primary endpoint for RAG. It accepts a query string, namespace, and optional search parameters.

### 📂 Management
- `GET /v1/namespaces`: Lists all active storage namespaces.
- `POST /v1/namespaces`: Creates a new logical isolation boundary (namespace).

## 3. Configuration & State Management
The API server utilizes the `OpenRAG` core facade via a lifespan-managed instance. This ensures that:
1. Resources (storage, LLM clients) are initialized once at startup.
2. Shutdown events cleanly close database connections and file handles.

## 4. Error Handling
Global exception handlers catch `OpenRAGError` and its subclasses, translating internal framework errors into appropriate HTTP status codes (e.g., 404 for missing namespaces, 500 for ingestion failures).

---
> **File Reference**: [openrag/api/app.py](file:///Users/nareshsingh/MEGA-P/dev/openrag/openrag/api/app.py)
