# Component Architecture: Frontend (Streamlit)

The Frontend provides a premium, interactive user interface for managing and querying the OpenRAG knowledge base.

## 1. Technology Stack
- **Framework**: Streamlit
- **Communication**: HTTP REST calls to the FastAPI backend.
- **Visuals**: Custom CSS for premium styling, real-time charts, and markdown rendering.

## 2. Key Modules (`frontend/main.py`)
- **Knowledge Management**: Upload files via drag-and-drop, view ingested documents, and monitor ingestion logs.
- **Pro Query Interface**: Multi-column layout featuring the AI chat assistant, document citations, and real-time processing metrics.
- **Configuration Sidebar**: Allows users to toggle between search modes (Dense, Hybrid, Graph) and adjust Top-K parameters.

## 3. State Handling
The UI maintains session-level state for the conversation history and selected namespace. It performs automatic health checks on the backend API before allowing user operations.

---
> **File Reference**: [frontend/main.py](file:///Users/nareshsingh/MEGA-P/dev/openrag/frontend/main.py)
