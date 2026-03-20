import { useState, useEffect } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { TopNav } from './components/layout/TopNav';
import { QueryStudio } from './components/chat/QueryStudio';
import { KnowledgeBase } from './components/kb/KnowledgeBase';
import { IngestionPipeline } from './components/pipeline/IngestionPipeline';
import { Observability } from './components/monitor/Observability';
import { Settings } from './components/settings/Settings';
import { Login } from './components/auth/Login';
import './App.css';
import './components/auth/Login.css';

export interface DocumentRecord {
  id: string;
  name: string;
  type: string;
  size: string;
  status: string;
  reason?: string;
  file?: File;
}

function App() {
  const [currentView, setCurrentView] = useState('query');
  const [uploadQueue, setUploadQueue] = useState<File[]>([]);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  
  // Auth State
  const [isAuthenticated, setIsAuthenticated] = useState(() => !!localStorage.getItem('openrag_token'));
  const [user, setUser] = useState<{username: string, role: string} | null>(() => {
    const saved = localStorage.getItem('openrag_user');
    return saved ? JSON.parse(saved) : null;
  });

  const getAuthHeaders = (): Record<string, string> => {
    const token = localStorage.getItem('openrag_token');
    return token ? { 'Authorization': `Bearer ${token}` } : {};
  };

  const logout = () => {
    localStorage.removeItem('openrag_token');
    localStorage.removeItem('openrag_user');
    setIsAuthenticated(false);
    setUser(null);
  };

  const handleLogin = (token: string, username: string, role: string) => {
    localStorage.setItem('openrag_token', token);
    localStorage.setItem('openrag_user', JSON.stringify({ username, role }));
    setIsAuthenticated(true);
    setUser({ username, role });
    setCurrentView('query');
  };

  useEffect(() => {
    if (!isAuthenticated) return;
    const fetchDocs = async () => {
      try {
        const res = await fetch('/api/v1/documents', {
          headers: getAuthHeaders() as HeadersInit
        });
        if (res.status === 401) {
          logout();
          return;
        }
        if (!res.ok) return;
        const data = await res.json();
        
        if (data && data.documents) {
          const loadedDocs: DocumentRecord[] = data.documents.map((d: any) => ({
            id: d.document_id,
            name: d.source_path.split('/').pop(),
            type: d.source_path.split('.').pop()?.toUpperCase() || 'FILE',
            size: d.metadata?.file_size_bytes ? (d.metadata.file_size_bytes / 1024 / 1024).toFixed(2) + ' MB' : 'Unknown',
            status: 'completed',
            reason: ''
          }));
          setDocuments(loadedDocs);
        }
      } catch (err) {
        console.error('Failed to hydrate documents from backend', err);
      }
    };
    fetchDocs();
  }, [isAuthenticated]);

  const handleUpload = (files: File[]) => {
    setUploadQueue(files);
    setCurrentView('ingestion');
    
    setDocuments(prev => {
      // Remove any previously queued or processed versions of these exact files
      const newNames = files.map(f => f.name);
      const filtered = prev.filter(d => !newNames.includes(d.name));
      
      const newDocs = files.map(file => ({
        id: crypto.randomUUID(),
        name: file.name,
        type: file.name.split('.').pop()?.toUpperCase() || 'FILE',
        size: (file.size / 1024 / 1024).toFixed(2) + ' MB',
        status: 'processing',
        file: file
      }));
      
      return [...newDocs, ...filtered];
    });
  };

  const handleIngestionComplete = (fileName: string, success: boolean, reason?: string) => {
    setDocuments(prev => prev.map(doc => {
      if (doc.name === fileName) {
        return { ...doc, status: success ? 'completed' : 'failed', reason };
      }
      return doc;
    }));
  };

  const handleDelete = async (id: string) => {
    try { 
      const res = await fetch(`/api/v1/documents/${id}`, { 
        method: 'DELETE',
        headers: getAuthHeaders()
      }); 
      if (res.status === 401) logout();
    } catch(e) {}
    setDocuments(prev => prev.filter(doc => doc.id !== id));
  };

  const handleDeleteMultiple = async (ids: string[]) => {
    for (const id of ids) {
      try { 
        const res = await fetch(`/api/v1/documents/${id}`, { 
          method: 'DELETE',
          headers: getAuthHeaders()
        }); 
        if (res.status === 401) logout();
      } catch(e) {}
    }
    setDocuments(prev => prev.filter(doc => !ids.includes(doc.id)));
  };

  const handleClearAll = async () => {
    if (window.confirm("Are you sure you want to completely empty the knowledge base? This action cannot be undone.")) {
      for (const doc of documents) {
        try { 
          const res = await fetch(`/api/v1/documents/${doc.id}`, { 
            method: 'DELETE',
            headers: getAuthHeaders()
          }); 
          if (res.status === 401) logout();
        } catch(e) {}
      }
      setDocuments([]);
    }
  };

  const getTitle = () => {
    switch (currentView) {
      case 'query': return 'Query Studio';
      case 'knowledge': return 'Knowledge Base';
      case 'ingestion': return 'Ingestion Pipeline';
      case 'observability': return 'Observability';
      default: return 'OpenRAG';
    }
  };

  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <div className="app-layout">
      <Sidebar currentView={currentView} setCurrentView={setCurrentView} role={user?.role || 'viewer'} />
      
      <div className="main-content">
        <TopNav title={getTitle()} />
        <div className="content-area relative h-full">
          <div style={{ display: currentView === 'query' ? 'block' : 'none', height: '100%' }}>
            <QueryStudio headers={getAuthHeaders()} onAuthError={logout} />
          </div>
          {user?.role === 'admin' && (
            <>
              <div style={{ display: currentView === 'knowledge' ? 'block' : 'none', height: '100%' }}>
                <KnowledgeBase 
                  documents={documents} 
                  onUpload={handleUpload} 
                  onDelete={handleDelete}
                  onDeleteMultiple={handleDeleteMultiple}
                  onClearAll={handleClearAll}
                  role={user?.role || 'viewer'}
                />
              </div>
              <div style={{ display: currentView === 'ingestion' ? 'block' : 'none', height: '100%' }}>
                <IngestionPipeline uploadQueue={uploadQueue} onComplete={handleIngestionComplete} headers={getAuthHeaders()} />
              </div>
              <div style={{ display: currentView === 'observability' ? 'block' : 'none', height: '100%' }}>
                <Observability headers={getAuthHeaders()} onAuthError={logout} />
              </div>
            </>
          )}
          <div style={{ display: currentView === 'settings' ? 'block' : 'none', height: '100%' }}>
            <Settings onLogout={logout} user={user} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
