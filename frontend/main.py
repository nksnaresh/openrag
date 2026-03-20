import streamlit as st
import requests
import os
from pathlib import Path
import json

# --- Page Config ---
st.set_page_config(
    page_title="OpenRAG Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stSecondaryBlock {
        background-color: #1e2130;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .citation-box {
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 10px;
        font-size: 0.85em;
    }
    .source-tag {
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 5px;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .terminal-window {
        background-color: #1a1b26;
        color: #7aa2f7;
        font-family: 'Courier New', Courier, monospace;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #414868;
        height: 400px;
        overflow-y: auto;
        font-size: 0.9em;
        line-height: 1.4;
    }
    .terminal-line { margin-bottom: 2px; }
    .terminal-info { color: #bb9af7; }
    .terminal-success { color: #9ece6a; }
    .terminal-error { color: #f7768e; }
    .terminal-debug { color: #565f89; }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000/api/v1"
if "show_telemetry" not in st.session_state:
    st.session_state.show_telemetry = True
if "last_telemetry" not in st.session_state:
    st.session_state.last_telemetry = None
if "terminal_logs" not in st.session_state:
    st.session_state.terminal_logs = []

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ OpenRAG Settings")
    st.markdown("---")
    
    st.subheader("📁 Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload files (.pdf, .txt, .py, .md)", 
        accept_multiple_files=True,
        type=["pdf", "txt", "py", "md"]
    )
    
    if st.button("🚀 Ingest Files") and uploaded_files:
        st.session_state.terminal_logs = ["> STARTING BATCH INGESTION..."]
        for uploaded_file in uploaded_files:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                res = requests.post(f"{st.session_state.api_url}/ingest/stream", files=files, stream=True)
                for line in res.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("RESULT:"):
                            data = json.loads(decoded[7:])
                            st.session_state.last_telemetry = {
                                "type": "Ingestion",
                                "file": uploaded_file.name,
                                "status": data.get("status"),
                                "block_count": data.get("block_count"),
                                "raw": data
                            }
                        else:
                            st.session_state.terminal_logs.append(decoded)
                            # Force update if visible
                            if st.session_state.show_telemetry:
                                st.rerun()
                st.success(f"Ingested {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
    
    st.markdown("---")
    st.subheader("🖥️ UI Settings")
    st.session_state.show_telemetry = st.checkbox("Show Telemetry Panel", value=st.session_state.show_telemetry)
    
    st.markdown("---")
    st.subheader("💡 Tips")
    st.info("""
    - Use .py files to test code retrieval.
    - Large PDFs may take a few seconds to process.
    - OpenRAG uses Hybrid Search (Vector + Graph + BM25).
    """)

# --- Helpers ---
def get_metrics():
    try:
        res = requests.get(f"{st.session_state.api_url}/metrics")
        if res.status_code == 200:
            lines = res.text.split("\n")
            metrics = {}
            for line in lines:
                if line.startswith("#") or not line:
                    continue
                parts = line.split(" ")
                if len(parts) >= 2:
                    metrics[parts[0]] = float(parts[1])
            return metrics
    except:
        return {}
    return {}

# --- Main UI ---
st.title("🤖 OpenRAG Explorer")

# Layout with optional telemetry panel
if st.session_state.show_telemetry:
    main_col, tel_col = st.columns([0.7, 0.3])
else:
    main_col = st.container()
    tel_col = None

with main_col:
    tab_chat, tab_obs = st.tabs(["💬 Pro Query", "📊 Observability"])

with tab_chat:
    st.markdown("Query your private knowledge base with multimodal RAG.")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message and message["citations"]:
                with st.expander("References"):
                    for cit in message["citations"]:
                        st.markdown(f"""
                        <div class="citation-box">
                            <div class="source-tag">📄 {cit['source_path']} (Score: {cit['score']:.4f})</div>
                            <div>{cit['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input
    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.terminal_logs = [f"> QUERY: {prompt}"]
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call API for response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            citations = []
            
            try:
                res = requests.post(
                    f"{st.session_state.api_url}/query/stream", 
                    json={"text": prompt, "namespace": "default"},
                    stream=True
                )
                
                for line in res.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("RESULT:"):
                            data = json.loads(decoded[7:])
                            full_response = data["answer"]
                            citations = data["citations"]
                            
                            st.session_state.last_telemetry = {
                                "type": "Query",
                                "text": prompt,
                                "latency_ms": data.get("latency_ms"),
                                "citations_count": len(citations),
                                "raw": data
                            }
                        else:
                            st.session_state.terminal_logs.append(decoded)
                            # We can't easily rerun here without breaking the loop 
                            # but we can try to update the terminal_placeholder if we had one
                
                if full_response:
                    message_placeholder.markdown(full_response)
                    if citations:
                        with st.expander("References"):
                            for cit in citations:
                                st.markdown(f"""
                                <div class="citation-box">
                                    <div class="source-tag">📄 {cit['source_path']} (Score: {cit['score']:.4f})</div>
                                    <div>{cit['content']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "citations": citations
                    })
                # Final rerun to update everything
                st.rerun()
            except Exception as e:
                st.error(f"Error connecting to API: {e}")

with tab_obs:
    st.header("📊 System Performance & Observability")
    metrics = get_metrics()
    
    if not metrics:
        st.warning("No metrics available. Make sure the API server is running and has processed some data.")
    else:
        # High Level Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Ingestion Docs (Metric suffix depends on tags, we'll look for substrings)
        ingest_total = sum(v for k, v in metrics.items() if "openrag_ingest_documents_total" in k)
        query_total = sum(v for k, v in metrics.items() if "openrag_query_requests_total" in k and 'status="completed"' in k)
        
        col1.metric("Documents Ingested", int(ingest_total))
        col2.metric("Total Queries", int(query_total))
        
        # Latency Histograms (Simplified: show avg if available via _sum/_count)
        ingest_sum = next((v for k, v in metrics.items() if "openrag_ingest_duration_seconds_sum" in k), 0)
        ingest_count = next((v for k, v in metrics.items() if "openrag_ingest_duration_seconds_count" in k), 1)
        avg_ingest = ingest_sum / max(ingest_count, 1)
        col3.metric("Avg Ingest Latency", f"{avg_ingest:.2f}s")
        
        query_sum = next((v for k, v in metrics.items() if "openrag_query_duration_seconds_sum" in k), 0)
        query_count = next((v for k, v in metrics.items() if "openrag_query_duration_seconds_count" in k), 1)
        avg_query = query_sum / max(query_count, 1)
        col4.metric("Avg Query Latency", f"{avg_query:.2f}s")
        
        st.markdown("---")
        
        # Charts
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.subheader("🚀 Ingestion Volume")
            blocks = {k.split('"')[3]: v for k, v in metrics.items() if "openrag_ingest_blocks_total" in k and 'type=' in k}
            if blocks:
                st.bar_chart(blocks)
            else:
                st.info("Ingest some documents to see block distribution.")
                
        with c_right:
            st.subheader("💰 Token Consumption")
            tokens = {k.split('"')[3]: v for k, v in metrics.items() if "openrag_llm_tokens_total" in k and 'model=' in k}
            if tokens:
                st.bar_chart(tokens)
            else:
                st.info("Perform some queries to see token usage.")

        st.markdown("---")
        st.subheader("🕵️ Advanced Tracing")
        st.markdown("""
        Automatic tracing is enabled via **OpenTelemetry**.  
        Traces are exported as OTLP spans to your configured collector. 
        """)
        if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            st.success(f"Connected to OTLP Collector at: `{os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT')}`")
        else:
            st.info("No OTLP collector configured. Spans are being generated but not exported externally.")

# --- Telemetry Panel (Terminal Side) ---
if tel_col:
    with tel_col:
        st.subheader("📟 System Console")
        # Terminal-style log box
        log_html = "".join([f'<div class="terminal-line">{line}</div>' for line in st.session_state.terminal_logs])
        st.markdown(f"""
        <div class="terminal-window">
            {log_html}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Console"):
            st.session_state.terminal_logs = []
            st.session_state.last_telemetry = None
            st.rerun()

        st.markdown("---")
        st.subheader("📡 Extraction Details")
        if not st.session_state.last_telemetry:
            st.info("No active extraction data.")
        else:
            tel = st.session_state.last_telemetry
            if tel["type"] == "Query":
                st.write(f"⏱️ **Latency**: `{tel['latency_ms']:.2f}ms`")
                if tel.get("raw") and "citations" in tel["raw"]:
                    for i, cit in enumerate(tel["raw"]["citations"][:5]):
                        st.caption(f"[{i+1}] {cit['source_path']} (Score: {cit['score']:.4f})")
            elif tel["type"] == "Ingestion":
                st.write(f"🧩 **Blocks**: `{tel['block_count']}`")
