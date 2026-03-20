import { useState, useRef } from 'react';
import { UploadCloud, FileText, Trash2, RefreshCw } from 'lucide-react';
import './KnowledgeBase.css';

interface Props {
  documents: {
    id: string;
    name: string;
    type: string;
    size: string;
    status: string;
    reason?: string;
    file?: File;
  }[];
  onUpload: (files: File[]) => void;
  onDelete: (id: string) => void;
  onDeleteMultiple: (ids: string[]) => void;
  onClearAll: () => void;
}

export function KnowledgeBase({ documents, onUpload, onDelete, onDeleteMultiple, onClearAll }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const fileInputRef = useRef<HTMLInputElement>(null);

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) newSet.delete(id);
      else newSet.add(id);
      return newSet;
    });
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === documents.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(documents.map(d => d.id)));
    }
  };

  const handleDeleteSelected = () => {
    if (selectedIds.size > 0 && window.confirm(`Are you sure you want to delete ${selectedIds.size} documents?`)) {
      onDeleteMultiple(Array.from(selectedIds));
      setSelectedIds(new Set());
    }
  };

  const handleRefreshClick = () => {
    setIsRefreshing(true);
    setTimeout(() => setIsRefreshing(false), 800);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onUpload(Array.from(e.target.files));
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onUpload(Array.from(e.dataTransfer.files));
    }
  };

  return (
    <div className="kb-container flex-col gap-4">
      <div className="kb-header">
        <h2>Knowledge Base Management</h2>
        <p className="text-secondary">Upload, review, and manage your vectors.</p>
      </div>

      <div 
        className={`upload-zone card ${isDragging ? 'dragging' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <UploadCloud size={48} className="upload-icon" />
        <h3>Drag & drop documents here</h3>
        <p className="text-secondary">Supports PDF, TXT, MD, PY (Max 200MB)</p>
        <button className="btn btn-primary mt-4" onClick={() => fileInputRef.current?.click()}>Browse Files</button>
        <input type="file" ref={fileInputRef} hidden multiple onChange={handleFileChange} />
      </div>

      <div className="docs-table-container card">
        <div className="flex justify-between items-center mb-4">
          <h3>Indexed Documents</h3>
          <div className="flex gap-2 items-center"> {/* Added flex and items-center for alignment */}
            <div className="search-box">
              <input type="text" placeholder="Search documents..." className="input" />
            </div>
            {selectedIds.size > 0 && (
              <button className="btn btn-outline" style={{ borderColor: '#ef4444', color: '#ef4444' }} onClick={handleDeleteSelected}>
                Delete Selected ({selectedIds.size})
              </button>
            )}
            {documents.length > 0 && (
              <button className="btn btn-outline" style={{ borderColor: '#ef4444', color: '#ef4444' }} onClick={onClearAll}>
                Clear All
              </button>
            )}
            <button className="btn btn-ghost" onClick={handleRefreshClick}>
              <RefreshCw size={16} className={isRefreshing ? 'spinner' : ''} />
            </button>
          </div>
        </div>
        
        <table className="docs-table">
          <thead>
            <tr>
              <th style={{ width: 40 }}>
                <input 
                  type="checkbox" 
                  checked={documents.length > 0 && selectedIds.size === documents.length}
                  onChange={toggleSelectAll}
                />
              </th>
              <th>Document Name</th>
              <th>Type</th>
              <th>Size</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {documents.length === 0 && (
              <tr>
                <td colSpan={6} className="text-center py-8 text-secondary"> {/* Changed colSpan to 6 */}
                  No documents have been indexed yet.
                </td>
              </tr>
            )}
            {documents.map(doc => (
              <tr key={doc.id}>
                <td>
                  <input 
                    type="checkbox" 
                    checked={selectedIds.has(doc.id)}
                    onChange={() => toggleSelect(doc.id)}
                  />
                </td>
                <td>
                  <div className="flex items-center gap-3 font-medium"> {/* Corrected gap-3">2 to gap-3 */}
                    <FileText size={16} className="text-secondary" />
                    {doc.name}
                  </div>
                </td>
                <td><span className="badge badge-outline">{doc.type}</span></td>
                <td><span className="text-secondary">{doc.size}</span></td>
                <td>
                  <div className="flex-col">
                    <span className={`badge badge-${doc.status}`}>
                      {doc.status.charAt(0).toUpperCase() + doc.status.slice(1)}
                    </span>
                    {doc.reason && <span className="text-xs text-danger mt-1">{doc.reason}</span>}
                  </div>
                </td>
                <td>
                  <div className="flex actions gap-2">
                    <button 
                      className={`btn btn-ghost action-btn ${!doc.file ? 'opacity-50 cursor-not-allowed' : ''}`}
                      title="Reprocess" 
                      onClick={() => doc.file && onUpload([doc.file])}
                      disabled={!doc.file || doc.status === 'processing' || doc.status === 'queued'}
                    ><RefreshCw size={14}/></button>
                    <button 
                      className="btn btn-ghost action-btn danger" 
                      title="Delete"
                      onClick={() => onDelete(doc.id)}
                    ><Trash2 size={14}/></button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
