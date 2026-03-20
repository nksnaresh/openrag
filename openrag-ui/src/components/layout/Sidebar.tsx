import { MessageSquare, LayoutDashboard, Database, Activity, Settings, HelpCircle } from 'lucide-react';

interface SidebarProps {
  currentView: string;
  setCurrentView: (view: string) => void;
}

export function Sidebar({ currentView, setCurrentView }: SidebarProps) {
  const navItems = [
    { id: 'query', label: 'Query Studio', icon: MessageSquare },
    { id: 'knowledge', label: 'Knowledge Base', icon: Database },
    { id: 'ingestion', label: 'Ingestion Pipeline', icon: LayoutDashboard },
    { id: 'observability', label: 'Observability', icon: Activity },
  ];

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div style={{ width: 24, height: 24, borderRadius: 6, background: 'var(--primary-color)' }}></div>
        OpenRAG
      </div>
      
      <div className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <div 
              key={item.id}
              className={`nav-item ${currentView === item.id ? 'active' : ''}`}
              onClick={() => setCurrentView(item.id)}
            >
              <Icon size={18} />
              {item.label}
            </div>
          );
        })}
      </div>
      
      <div className="sidebar-footer">
        <div 
          className="nav-item cursor-pointer" 
          onClick={() => window.open('https://github.com/nareshsingh/openrag', '_blank')}
        >
          <HelpCircle size={18} />
          Documentation
        </div>
        <div 
          className={`nav-item cursor-pointer ${currentView === 'settings' ? 'active' : ''}`}
          onClick={() => setCurrentView('settings')}
        >
          <Settings size={18} />
          Settings
        </div>
      </div>
    </div>
  );
}
