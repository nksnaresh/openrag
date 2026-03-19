# Component Architecture: Unified Storage Layer

OpenRAG uses a pluggable, unified storage architecture that supports both in-memory development and persistent production environments.

## 1. Storage Adapters (`openrag/storage/`)
All storage interaction happens through abstract base classes, allowing seamless backend swaps:
- **Vector DB**: Manages high-dimensional embeddings.
- **Graph DB**: Manages entities and their semantic relationships.
- **Document Store**: Manages raw doc content, metadata, and citation logs.

## 2. NPZ Persistence (`openrag/storage/vector/npz.py`)
A custom-built, lightweight persistence adapter:
- Uses standard **NumPy (`.npz`)** files for storage.
- **Zero-Dependency**: No need to run external services like Qdrant or Pinecone for local setups.
- **Atomic Saves**: Uses temporary file replacement to prevent data corruption during crashes.

## 3. Metadata Persistence (`openrag/storage/document/sqlite.py`)
Relational data and document logs are stored in a local **SQLite** database (`openrag_store.db`). This ensures that even after a server restart, all document names, hashes, and namespaces are preserved.

## 4. Namespace Isolation
The storage layer enforces strict logical isolation. Each `namespace` corresponds to a unique directory or database prefix, ensuring multi-tenant data security at the storage level.

---
> **File Reference**: [openrag/storage/vector/npz.py](file:///Users/nareshsingh/MEGA-P/dev/openrag/openrag/storage/vector/npz.py)
