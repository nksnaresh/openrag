import { useState, useEffect, useRef } from 'react';
import { Activity, Database, BarChart3, Clock, RefreshCw, Zap, ZapOff, Trash2 } from 'lucide-react';
import './Observability.css';

export interface TraceNode {
  time: string;
  content: string;
  meta: string;
}

export interface PrometheusMetric {
  name: string;
  help: string;
  type: string;
  value: string;
  labels?: Record<string, string>;
}

interface ObservabilityProps {
  headers?: Record<string, string>;
  onAuthError?: () => void;
}

export function Observability({ headers, onAuthError }: ObservabilityProps) {
  const [activeTab, setActiveTab] = useState('metrics');
  const [logs, setLogs] = useState<string[]>([]);
  const [trace, setTrace] = useState<TraceNode[]>([]);
  const [metrics, setMetrics] = useState<PrometheusMetric[]>([]);
  const [isLive, setIsLive] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  
  const logEndRef = useRef<HTMLDivElement>(null);

  const fetchLogs = async () => {
    try {
      const res = await fetch('/api/v1/observability/logs', { headers });
      if (res.status === 401 && onAuthError) onAuthError();
      if (!res.ok) return;
      const data = await res.json();
      setLogs(data.logs || []);
    } catch (err) {}
  };

  const fetchTraces = async () => {
    try {
      const res = await fetch('/api/v1/observability/traces', { headers });
      if (res.status === 401 && onAuthError) onAuthError();
      if (!res.ok) return;
      const data = await res.json();
      setTrace(data.trace || []);
    } catch (err) {}
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch('/api/v1/metrics', { headers });
      if (res.status === 401 && onAuthError) onAuthError();
      if (!res.ok) return;
      const text = await res.text();
      setMetrics(parseMetrics(text));
    } catch (err) {}
  };

  const refreshAll = async () => {
    setIsRefreshing(true);
    await Promise.all([fetchLogs(), fetchTraces(), fetchMetrics()]);
    setIsRefreshing(false);
  };

  const clearLogs = async () => {
    if (!window.confirm("Clear all system logs?")) return;
    try {
      const res = await fetch('/api/v1/observability/logs', { method: 'DELETE', headers });
      if (res.status === 401 && onAuthError) onAuthError();
      if (res.ok) setLogs([]);
    } catch (err) {}
  };

  useEffect(() => {
    refreshAll();
  }, [headers]);

  useEffect(() => {
    if (!isLive) return;
    
    const interval = setInterval(() => {
      if (activeTab === 'metrics') fetchMetrics();
      if (activeTab === 'logs') fetchLogs();
      if (activeTab === 'traces') fetchTraces();
    }, 3000);

    return () => clearInterval(interval);
  }, [isLive, activeTab, headers]);

  useEffect(() => {
    if (isLive && activeTab === 'logs') {
      logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, isLive]);

  const parseMetrics = (text: string): PrometheusMetric[] => {
    const lines = text.split('\n');
    const result: PrometheusMetric[] = [];
    const familyHelp: Record<string, string> = {};
    const familyType: Record<string, string> = {};

    lines.forEach(line => {
      const trimmed = line.trim();
      if (!trimmed) return;

      if (trimmed.startsWith('# HELP')) {
        const parts = trimmed.split(' ');
        const name = parts[2];
        const help = parts.slice(3).join(' ');
        familyHelp[name] = help;
      } else if (trimmed.startsWith('# TYPE')) {
        const parts = trimmed.split(' ');
        const name = parts[2];
        const type = parts[3];
        familyType[name] = type;
      } else if (!trimmed.startsWith('#')) {
        const parts = trimmed.split(/\s+/);
        const fullName = parts[0];
        const value = parts[parts.length - 1];

        // Parse labels: name{k="v",...}
        let baseName = fullName;
        let labels: Record<string, string> = {};
        if (fullName.includes('{')) {
          baseName = fullName.split('{')[0];
          const labelStr = fullName.split('{')[1].split('}')[0];
          labelStr.split(',').forEach(pair => {
            const [k, v] = pair.split('=');
            if (k && v) labels[k.trim()] = v.replace(/"/g, '').trim();
          });
        }

        // Filter out _CREATED and _BUCKET as they are noisy metadata
        if (baseName.endsWith('_created') || baseName.includes('_bucket')) return;

        result.push({
          name: baseName,
          help: familyHelp[baseName] || 'System telemetry metric',
          type: familyType[baseName] || 'gauge',
          value,
          labels
        });
      }
    });
    return result;
  };

  const getLogLevelClass = (log: string) => {
    const lower = log.toLowerCase();
    // Reduce alarm for BertModel load reports
    if (lower.includes('bertmodel load report') || lower.includes('unexpected') || lower.includes('notes:')) return 'log-debug';
    
    if (lower.includes('error') || lower.includes('403') || lower.includes('500') || lower.includes('failed')) return 'log-error';
    if (lower.includes('warn')) return 'log-warn';
    if (lower.includes('success') || lower.includes('200 ok') || lower.includes('completed')) return 'log-success';
    if (lower.includes('debug')) return 'log-debug';
    return 'log-info';
  };

  return (
    <div className="obs-container">
      <div className="obs-header">
        <div className="obs-tabs">
          <button className={`tab ${activeTab === 'metrics' ? 'active' : ''}`} onClick={() => setActiveTab('metrics')}>
            <BarChart3 size={18} /> System Metrics
          </button>
          <button className={`tab ${activeTab === 'traces' ? 'active' : ''}`} onClick={() => setActiveTab('traces')}>
            <Activity size={18} /> Query Traces
          </button>
          <button className={`tab ${activeTab === 'logs' ? 'active' : ''}`} onClick={() => setActiveTab('logs')}>
            <Database size={18} /> System Logs
          </button>
        </div>
        
        <div className="obs-controls">
          <button 
            className={`control-btn ${isLive ? 'live-active' : ''}`} 
            onClick={() => setIsLive(!isLive)}
            title={isLive ? "Disable Live Mode" : "Enable Live Mode"}
          >
            {isLive ? <Zap size={16} /> : <ZapOff size={16} />}
            <span>{isLive ? 'LIVE' : 'AUTO-REFRESH OFF'}</span>
          </button>
          
          <button 
            className="control-btn refresh-btn" 
            onClick={refreshAll}
            disabled={isRefreshing}
            title="Refresh Data"
          >
            <RefreshCw size={16} className={isRefreshing ? 'spinner' : ''} />
            <span>Refresh</span>
          </button>

          {activeTab === 'logs' && (
             <button className="control-btn delete-btn" onClick={clearLogs} title="Clear Logs">
                <Trash2 size={16} />
             </button>
          )}
        </div>
      </div>

      <div className="obs-content">
        {activeTab === 'metrics' && (
          <div className="metrics-grid">
            {metrics.map((m, i) => {
              const labelText = m.labels && Object.keys(m.labels).length > 0
                ? Object.entries(m.labels).map(([k, v]) => `${k}=${v}`).join(', ')
                : '';
              
              return (
                <div key={i} className="metric-card">
                  <div className="metric-header">
                    <div className="metric-name-group">
                      <span className="metric-name">{(m.name || 'unnamed').replace('openrag_', '').toUpperCase()}</span>
                      {labelText && <span className="metric-labels">{labelText}</span>}
                    </div>
                    <BarChart3 size={16} className="text-secondary" />
                  </div>
                  <div className="metric-value">
                    {m.name.includes('duration') ? `${parseFloat(m.value).toFixed(3)}s` : parseFloat(m.value || '0').toLocaleString()}
                  </div>
                  <div className="metric-footer">{m.help || 'No description available'}</div>
                </div>
              );
            })}
            {metrics.length === 0 && <div className="no-data">No metrics available. Click Refresh.</div>}
          </div>
        )}

        {activeTab === 'traces' && (
          <div className="trace-list">
            {trace.map((node, i) => (
              <div key={i} className="trace-node shadow-sm">
                <div className="trace-time"><Clock size={14} /> {node.time}</div>
                <div className="trace-main">
                   <div dangerouslySetInnerHTML={{ __html: node.content }} />
                   {node.meta && <span className="trace-meta">{node.meta}</span>}
                </div>
              </div>
            ))}
            {trace.length === 0 && <div className="no-data">No traces found. Run some queries first.</div>}
          </div>
        )}

        {activeTab === 'logs' && (
          <div className="log-viewer terminal-style">
            {logs.map((log, i) => (
              <div key={i} className={`log-entry ${getLogLevelClass(log)}`}>
                <span className="log-num">{i + 1}</span>
                <span className="log-text">{log}</span>
              </div>
            ))}
            <div ref={logEndRef} />
            {logs.length === 0 && <div className="no-data">Log file is empty or missing.</div>}
          </div>
        )}
      </div>
    </div>
  );
}
