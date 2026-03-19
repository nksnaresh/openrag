# Kill existing if they are hanging
pkill -f "openrag/api/app.py"
pkill -f "streamlit run frontend/main.py"

# Start Backend
nohup .venv/bin/python openrag/api/app.py > api_server.log 2>&1 &

# Start Frontend
nohup .venv/bin/streamlit run frontend/main.py --server.port 8501 --server.headless true > frontend.log 2>&1 &

