import { useState, useEffect } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { TopNav } from './components/layout/TopNav';
import { QueryStudio } from './components/chat/QueryStudio';
import { KnowledgeBase } from './components/kb/KnowledgeBase';
import { IngestionPipeline } from './components/pipeline/IngestionPipeline';
import { Observability } from './components/monitor/Observability';
import { Settings } from './components/settings/Settings';
import './App.css';

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

  useEffect(() => {
    const fetchDocs = async () => {
      try {
        const res = await fetch('/api/v1/documents');
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
  }, []);

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
    try { await fetch(`/api/v1/documents/${id}`, { method: 'DELETE' }); } catch(e) {}
    setDocuments(prev => prev.filter(doc => doc.id !== id));
  };

  const handleDeleteMultiple = async (ids: string[]) => {
    for (const id of ids) {
      try { await fetch(`/api/v1/documents/${id}`, { method: 'DELETE' }); } catch(e) {}
    }
    setDocuments(prev => prev.filter(doc => !ids.includes(doc.id)));
  };

  const handleClearAll = async () => {
    if (window.confirm("Are you sure you want to completely empty the knowledge base? This action cannot be undone.")) {
      for (const doc of documents) {
        try { await fetch(`/api/v1/documents/${doc.id}`, { method: 'DELETE' }); } catch(e) {}
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

  return (
    <div className="app-layout">
      <Sidebar currentView={currentView} setCurrentView={setCurrentView} />
      
      <div className="main-content">
        <TopNav title={getTitle()} />
        <div className="content-area relative h-full">
          <div style={{ display: currentView === 'query' ? 'block' : 'none', height: '100%' }}>
            <QueryStudio />
          </div>
          <div style={{ display: currentView === 'knowledge' ? 'block' : 'none', height: '100%' }}>
            <KnowledgeBase 
              documents={documents} 
              onUpload={handleUpload} 
              onDelete={handleDelete}
              onDeleteMultiple={handleDeleteMultiple}
              onClearAll={handleClearAll}
            />
          </div>
          <div style={{ display: currentView === 'ingestion' ? 'block' : 'none', height: '100%' }}>
            <IngestionPipeline uploadQueue={uploadQueue} onComplete={handleIngestionComplete} />
          </div>
          <div style={{ display: currentView === 'observability' ? 'block' : 'none', height: '100%' }}>
            <Observability />
          </div>
          <div style={{ display: currentView === 'settings' ? 'block' : 'none', height: '100%' }}>
            <Settings />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
