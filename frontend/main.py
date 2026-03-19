import streamlit as st
import requests
import os
from pathlib import Path

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
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000/api/v1"

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
        with st.spinner("Ingesting..."):
            for uploaded_file in uploaded_files:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                try:
                    res = requests.post(f"{st.session_state.api_url}/ingest", files=files)
                    if res.status_code == 200:
                        st.success(f"Ingested {uploaded_file.name}")
                    else:
                        st.error(f"Failed to ingest {uploaded_file.name}: {res.text}")
                except Exception as e:
                    st.error(f"Error connecting to API: {e}")
    
    st.markdown("---")
    st.subheader("💡 Tips")
    st.info("""
    - Use .py files to test code retrieval.
    - Large PDFs may take a few seconds to process.
    - OpenRAG uses Hybrid Search (Vector + Graph + BM25).
    """)

# --- Main UI ---
st.title("🤖 OpenRAG Explorer")
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
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API for response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        citations = []
        
        try:
            # We use the blocking /query endpoint for simplicity in this version
            # Stream response support can be added later
            res = requests.post(
                f"{st.session_state.api_url}/query", 
                json={"text": prompt, "namespace": "default"}
            )
            
            if res.status_code == 200:
                data = res.json()
                full_response = data["answer"]
                citations = data["citations"]
                
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
                
                # Add assistant message to chat state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "citations": citations
                })
            else:
                st.error(f"API Error: {res.status_code} - {res.text}")
                
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
