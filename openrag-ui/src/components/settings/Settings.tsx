import { LogOut, User as UserIcon, Shield, Laptop, Globe, Info } from 'lucide-react';
import './Settings.css';

interface SettingsProps {
  onLogout?: () => void;
  user?: { username: string, role: string } | null;
}

export function Settings({ onLogout, user }: SettingsProps) {
  return (
    <div className="settings-container p-8">
      <div className="settings-header mb-8">
        <h2>Settings</h2>
        <p className="text-secondary">Manage your account and platform configuration.</p>
      </div>

      <div className="settings-grid">
        <section className="settings-section card">
          <div className="section-header">
            <UserIcon size={20} />
            <h3>User Profile</h3>
          </div>
          <div className="profile-card">
            <div className="profile-avatar">
              {user?.username?.charAt(0).toUpperCase() || 'U'}
            </div>
            <div className="profile-info">
              <div className="profile-name">{user?.username || 'Unknown User'}</div>
              <div className="profile-role">
                <Shield size={12} /> {user?.role?.toUpperCase() || 'VIEWER'}
              </div>
            </div>
            <button className="btn btn-danger btn-sm ml-auto" onClick={onLogout}>
              <LogOut size={16} /> Logout
            </button>
          </div>
        </section>

        <section className="settings-section card">
          <div className="section-header">
            <Laptop size={20} />
            <h3>System Status</h3>
          </div>
          <div className="status-list">
            <div className="status-item">
              <span className="label">API Version</span>
              <span className="value">v1.2.4 (Enterprise)</span>
            </div>
            <div className="status-item">
              <span className="label">Embedding Engine</span>
              <span className="value text-primary">Local (BGE-base)</span>
            </div>
            <div className="status-item">
              <span className="label">Synthesis Model</span>
              <span className="value">Gemini-Flash</span>
            </div>
            <div className="status-item">
              <span className="label">Vector DB</span>
              <span className="value">NPZ Persistent</span>
            </div>
          </div>
        </section>

        <section className="settings-section card">
          <div className="section-header">
            <Globe size={20} />
            <h3>Localization</h3>
          </div>
          <p className="text-secondary text-sm mb-4">This instance is configured for 100% on-premise execution of embedding models.</p>
          <div className="status-item">
            <span className="label">Region</span>
            <span className="value">Localhost</span>
          </div>
          <div className="status-item">
            <span className="label">Privacy Mode</span>
            <span className="value text-primary">Air-Gapped Ready</span>
          </div>
        </section>

        <section className="settings-section card">
          <div className="section-header">
            <Info size={20} />
            <h3>Documentation</h3>
          </div>
          <p className="text-secondary text-sm">Access the OpenRAG architectural guides and API documentation.</p>
          <a href="#" className="btn btn-outline btn-sm mt-4 w-fit">Open Documentation</a>
        </section>
      </div>
    </div>
  );
}
