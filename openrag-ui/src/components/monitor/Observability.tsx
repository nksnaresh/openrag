import { Activity, Database, Clock, Layers, ShieldCheck, TerminalSquare, Trash2 } from 'lucide-react';
import { useState, useEffect } from 'react';
import './Observability.css';

export function Observability() {
  const [metrics, setMetrics] = useState({
    totalQueries: 0,
    avgLatency: 0,
    vectorsIndexed: 0,
    tokensUsed: 0
  });
  const [logs, setLogs] = useState<string[]>([]);
  const [traces, setTraces] = useState<any[]>([]);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await fetch('/api/v1/metrics');
        const text = await res.text();
        
        let queries = 0;
        let latencySum = 0;
        let blocks = 0;
        let tokens = 0;

        text.split('\n').forEach(line => {
          if (line.includes('openrag_query_requests_total') && line.includes('status="completed"')) {
            queries += parseFloat(line.split(' ').pop() || '0');
          }
          if (line.startsWith('openrag_query_duration_seconds_sum')) {
            latencySum += parseFloat(line.split(' ').pop() || '0');
          }
          if (line.startsWith('openrag_ingest_blocks_total')) {
            blocks += parseFloat(line.split(' ').pop() || '0');
          }
          if (line.startsWith('openrag_llm_tokens_total')) {
            tokens += parseFloat(line.split(' ').pop() || '0');
          }
        });

        const avg = queries > 0 ? Math.round((latencySum / queries) * 1000) : 0;
        
        setMetrics({
          totalQueries: queries,
          avgLatency: avg,
          vectorsIndexed: blocks,
          tokensUsed: tokens
        });
      } catch (err) {}

      try {
        const logRes = await fetch('/api/v1/observability/logs');
        const logData = await logRes.json();
        if (logData && logData.logs) setLogs(logData.logs);

        const traceRes = await fetch('/api/v1/observability/traces');
        const traceData = await traceRes.json();
        if (traceData && traceData.trace) setTraces(traceData.trace);
      } catch (err) {}
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleClearLogs = async () => {
    try {
      await fetch('/api/v1/observability/logs', { method: 'DELETE' });
      setLogs([]);
    } catch(e) {}
  };

  return (
    <div className="observability-container flex-col gap-4">
      <div className="card pipeline-header">
        <h2>System Observability</h2>
        <p className="text-secondary">Monitor performance metrics, trace executions, and check system health.</p>
      </div>

      <div className="kpi-grid">
        <div className="card kpi-card">
          <div className="kpi-icon text-primary"><Activity size={20}/></div>
          <div className="kpi-data">
            <span className="kpi-label">Total Queries</span>
            <span className="kpi-val">{metrics.totalQueries}</span>
            <span className="kpi-trend positive">Live updates</span>
          </div>
        </div>
        <div className="card kpi-card">
          <div className="kpi-icon text-success"><Clock size={20}/></div>
          <div className="kpi-data">
            <span className="kpi-label">Avg Latency</span>
            <span className="kpi-val">{metrics.avgLatency}ms</span>
            <span className="kpi-trend positive">Real-time</span>
          </div>
        </div>
        <div className="card kpi-card">
          <div className="kpi-icon text-warning"><Database size={20}/></div>
          <div className="kpi-data">
            <span className="kpi-label">Vectors Indexed</span>
            <span className="kpi-val">{metrics.vectorsIndexed.toLocaleString()}</span>
            <span className="kpi-trend">Stable volume</span>
          </div>
        </div>
        <div className="card kpi-card">
          <div className="kpi-icon text-danger"><Layers size={20}/></div>
          <div className="kpi-data">
            <span className="kpi-label">Tokens Used</span>
            <span className="kpi-val">{metrics.tokensUsed >= 1000 ? (metrics.tokensUsed / 1000).toFixed(1) + 'k' : metrics.tokensUsed}</span>
            <span className="kpi-trend negative">Gemini limits apply</span>
          </div>
        </div>
      </div>

      <div className="grid-2col gap-4">
        <div className="card h-full flex-col">
          <div className="flex justify-between items-center mb-4 border-b pb-4">
            <h3 className="flex items-center gap-2"><ShieldCheck size={18} className="text-primary"/> Recent Query Trace</h3>
            <span className="badge badge-outline">ID: req_9x42b</span>
          </div>
          
          <div className="trace-list flex-1">
            {traces.length === 0 ? (
              <div className="p-4 text-center text-secondary text-sm">Waiting for query trace execution...</div>
            ) : traces.map((t, i) => (
              <div className="trace-item" key={i}>
                <div className="trace-time">{t.time}</div>
                <div className="trace-content">
                  <span dangerouslySetInnerHTML={{__html: t.content}}></span>
                  {t.meta && <div className="trace-meta text-secondary text-xs mt-1">{t.meta}</div>}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card h-full flex-col logs-container">
          <div className="flex justify-between items-center mb-4 border-b pb-4">
            <h3 className="flex items-center gap-2"><TerminalSquare size={18} className="text-secondary"/> System Logs</h3>
            <div className="flex gap-2 items-center">
              <span className="badge badge-processing">Live</span>
              <button className="btn btn-ghost p-1" onClick={handleClearLogs} title="Clear Logs">
                <Trash2 size={16} className="text-secondary hover:text-danger"/>
              </button>
            </div>
          </div>
          <div className="terminal-logs flex-1" style={{ maxHeight: '430px', overflowY: 'auto' }}>
            {logs.length === 0 ? (
              <div className="log-line"><span className="text-secondary">--:--:--</span> <span className="text-secondary">[INFO]</span> No live events...</div>
            ) : logs.map((log, i) => {
              const cleaned = log.replace(/^(\[.*?\])/, '');
              const tagMatch = log.match(/^(\[.*?\])/);
              const tag = tagMatch ? tagMatch[1] : '';
              let tagClass = 'text-secondary';
              if (tag.includes('INFO')) tagClass = 'text-success';
              if (tag.includes('ERROR')) tagClass = 'text-danger';
              if (tag.includes('WARN')) tagClass = 'text-warning';
              if (tag.includes('DEBUG')) tagClass = 'text-primary';

              return (
                <div className="log-line" key={i}>
                  <span className="timestamp font-mono text-xs opacity-50 mr-2">{new Date().toLocaleTimeString()}</span>
                  {tag && <span className={`${tagClass} mr-1 font-bold`}>{tag}</span>}
                  <span>{cleaned}</span>
                </div>
              );
            })}
            <div className="log-line streaming-log mt-2"><span className="text-success blink">_</span></div>
          </div>
        </div>
      </div>
    </div>
  );
}
