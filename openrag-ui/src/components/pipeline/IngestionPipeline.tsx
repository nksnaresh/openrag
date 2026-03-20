import React, { useState, useEffect } from 'react';
import { Circle, CheckCircle2, Loader2, PlayCircle } from 'lucide-react';
import './IngestionPipeline.css';

interface Props {
  uploadQueue?: File[];
  onComplete?: (fileName: string, success: boolean, reason?: string) => void;
  headers?: Record<string, string>;
  onAuthError?: () => void;
}

export function IngestionPipeline({ uploadQueue, onComplete, headers, onAuthError }: Props) {
  const steps = ['Upload', 'Parse', 'Chunk', 'Embed', 'Index'];
  const [currentStep, setCurrentStep] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);

  const currentFile = uploadQueue && uploadQueue.length > currentIndex ? uploadQueue[currentIndex] : null;

  useEffect(() => {
    if (!currentFile) {
      if (uploadQueue && uploadQueue.length > 0 && currentIndex >= uploadQueue.length) {
        setLogs(prev => [...prev, '[INFO] Batch upload complete.']);
        setIsProcessing(false);
      } else {
        setLogs(['[INFO] Waiting for files...']);
      }
      return;
    }

    setLogs(prev => [...prev, `---`, `[INFO] Starting upload for ${currentFile.name}...`]);
    setCurrentStep(0);
    setIsProcessing(true);

    const formData = new FormData();
    formData.append('file', currentFile);

    const abortController = new AbortController();

    fetch('/api/v1/ingest/stream', {
      method: 'POST',
      headers: headers,
      body: formData,
      signal: abortController.signal
    }).then(async (response) => {
      if (response.status === 401 && onAuthError) {
        onAuthError();
        return;
      }
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      setLogs(prev => [...prev, '[INFO] Server received file. Pipeline execution started.']);
      setCurrentStep(1); // Parse
      
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) return;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        lines.forEach(line => {
          if (!line.trim()) return;
          
          if (line.startsWith('RESULT:')) {
            setCurrentStep(5); // Completed all steps
            try {
              const data = JSON.parse(line.substring(7));
              if (data.status === 'success' || data.status === 'completed') {
                setLogs(prev => [...prev, `[SUCCESS] ${currentFile.name} successfully indexed.`]);
                if (onComplete) onComplete(currentFile.name, true);
              } else if (data.status === 'skipped') {
                setLogs(prev => [...prev, `[INFO] Skipped: ${currentFile.name} (${data.reason || 'Already processed'})`]);
                if (onComplete) onComplete(currentFile.name, true, data.reason || 'Duplicate skipped');
              } else {
                setLogs(prev => [...prev, `[ERROR] Ingestion failed: ${data.reason || 'Unknown error'}`]);
                if (onComplete) onComplete(currentFile.name, false, data.reason || 'Unknown error');
              }
            } catch (e) {
              setLogs(prev => [...prev, `[SUCCESS] ${currentFile.name} successfully indexed.`]);
              if (onComplete) onComplete(currentFile.name, true);
            }
            
            setTimeout(() => setCurrentIndex(prev => prev + 1), 1000);
            return;
          }
          
          setLogs(prev => [...prev, line]);
          
          const lower = line.toLowerCase();
          if (lower.includes('extracting') || lower.includes('parse')) setCurrentStep(1);
          if (lower.includes('chunk') || lower.includes('generated')) setCurrentStep(2);
          if (lower.includes('embed')) setCurrentStep(3);
          if (lower.includes('store') || lower.includes('index')) setCurrentStep(4);
        });
      }
    }).catch(err => {
      if (err.name !== 'AbortError') {
        setLogs(prev => [...prev, `[ERROR] Ingestion failed for ${currentFile.name}: ${err.message}`]);
        if (onComplete) onComplete(currentFile.name, false, err.message);
        setTimeout(() => setCurrentIndex(prev => prev + 1), 1000);
      }
    });

    return () => abortController.abort();
  }, [currentFile]);

  return (
    <div className="pipeline-container flex-col gap-4">
      <div className="card pipeline-header flex justify-between items-center">
        <div>
          <h2>Ingestion Pipeline</h2>
          <p className="text-secondary">Monitor the real-time processing of documents through the RAG framework.</p>
        </div>
        <button className="btn btn-outline text-primary border-primary">
          <PlayCircle size={16} /> View History
        </button>
      </div>

      <div className="card py-8">
        <h3 className="mb-4 text-center">
          Current Job {uploadQueue && uploadQueue.length > 0 && `(${Math.min(currentIndex + 1, uploadQueue.length)}/${uploadQueue.length})`}: <span className="text-primary font-medium">{currentFile ? currentFile.name : 'Idle'}</span>
        </h3>
        <div className="pipeline-flow-wrapper">
          <div className="pipeline-flow">
            {steps.map((step, idx) => {
              const isCompleted = currentStep > idx;
              const isActive = currentStep === idx && isProcessing;
              return (
                <React.Fragment key={step}>
                  <div className={`step-node ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}>
                    <div className="step-icon-wrapper">
                      {isCompleted ? <CheckCircle2 size={24} /> : isActive ? <Loader2 size={24} className="spinner" /> : <Circle size={24} />}
                    </div>
                    <span className="step-label">{step}</span>
                  </div>
                  {idx < steps.length - 1 && (
                    <div className={`step-connector flex-1 ${currentStep > idx ? 'completed' : ''}`}></div>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>
      </div>

      <div className="card pipeline-logs-card">
        <h3>Live Activity Stream</h3>
        <div className="terminal-logs">
          {logs.map((log, i) => (
            <div key={i} className={`log-line ${isProcessing && i === logs.length - 1 ? 'streaming-log text-primary font-medium' : 'text-secondary'}`}>
              <span className="timestamp font-mono text-xs opacity-50 mr-2">{new Date().toLocaleTimeString()}</span>
              {log}
            </div>
          ))}
          {isProcessing && <div className="cursor-blink font-mono text-primary mt-2">_</div>}
        </div>
      </div>
    </div>
  );
}
