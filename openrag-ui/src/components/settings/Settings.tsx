import { Save, AlertTriangle } from 'lucide-react';

export function Settings() {
  return (
    <div className="h-full overflow-auto p-8" style={{ padding: '2rem' }}>
      <h2 style={{ marginBottom: '1.5rem', fontSize: '1.5rem', fontWeight: 600 }}>Platform Configuration</h2>
      
      <div className="card" style={{ maxWidth: '800px', marginBottom: '2rem' }}>
        <h3 className="mb-4" style={{ marginBottom: '1rem', fontWeight: 500 }}>API Configuration</h3>
        <div style={{ marginBottom: '1.2rem' }}>
          <label style={{ display: 'block', fontSize: '0.875rem', color: 'var(--secondary-color)', marginBottom: '0.5rem' }}>Core Engine Endpoint</label>
          <input type="text" className="input" style={{ width: '100%' }} defaultValue="http://localhost:8000/api/v1" />
        </div>
        <div style={{ marginBottom: '1.5rem' }}>
          <label style={{ display: 'block', fontSize: '0.875rem', color: 'var(--secondary-color)', marginBottom: '0.5rem' }}>Tracing Backend</label>
          <select className="input" style={{ width: '100%' }}>
            <option>Local Console Output</option>
            <option>Prometheus / Grafana Endpoint</option>
            <option>Datadog Agent (StatsD)</option>
          </select>
        </div>
        <button className="btn btn-primary flex items-center">
          <Save size={16} className="mr-2" style={{ marginRight: '0.5rem' }}/> Save Configuration
        </button>
      </div>

      <div className="card" style={{ maxWidth: '800px', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
        <h3 style={{ marginBottom: '0.5rem', fontWeight: 500, color: '#ef4444', display: 'flex', alignItems: 'center' }}>
          <AlertTriangle size={18} style={{ marginRight: '0.5rem' }} /> Danger Zone
        </h3>
        <p style={{ color: 'var(--secondary-color)', fontSize: '0.875rem', marginBottom: '1rem' }}>
          Irreversible actions that affect the entire workspace context.
        </p>
        <button className="btn btn-outline" style={{ borderColor: '#ef4444', color: '#ef4444' }}>
          Hard Reset Vector DB (Not Implemented)
        </button>
      </div>
    </div>
  );
}
