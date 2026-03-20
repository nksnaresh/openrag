import { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Loader2, BookOpen, FileText, Copy, Check, ChevronDown, ChevronUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './QueryStudio.css';

interface Citation {
  source_path: string;
  score: number;
  content: string;
  block_type: string;
  page_number?: number;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  streaming?: boolean;
}

function MessageContent({ content, role }: { content: string, role: string }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const isLong = content.length > 800;
  const displayContent = isLong && !isExpanded ? content.slice(0, 800) + '...' : content;

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (role === 'user') {
    return <div className="message-text">{content}</div>;
  }

  return (
    <div className="markdown-container">
      <div className="markdown-body">
        <ReactMarkdown 
          remarkPlugins={[remarkGfm]}
          components={{
            code({ node, inline, className, children, ...props }: any) {
              const match = /language-(\w+)/.exec(className || '');
              return !inline && match ? (
                <div className="code-block-wrapper">
                  <div className="code-header">
                    <span>{match[1]}</span>
                    <button className="copy-code-btn" onClick={() => navigator.clipboard.writeText(String(children).replace(/\n$/, ''))}>
                      <Copy size={12} />
                    </button>
                  </div>
                  <SyntaxHighlighter
                    style={oneDark}
                    language={match[1]}
                    PreTag="div"
                    {...props}
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                </div>
              ) : (
                <code className={className} {...props}>
                  {children}
                </code>
              );
            }
          }}
        >
          {displayContent}
        </ReactMarkdown>
      </div>
      
      <div className="message-footer-actions">
        {isLong && (
          <button className="btn-toggle-expand" onClick={() => setIsExpanded(!isExpanded)}>
            {isExpanded ? (
              <><ChevronUp size={14} /> Show Less</>
            ) : (
              <><ChevronDown size={14} /> Show More</>
            )}
          </button>
        )}
        <button className={`btn-copy-msg ${copied ? 'copied' : ''}`} onClick={handleCopy}>
          {copied ? <><Check size={14} /> Copied</> : <><Copy size={14} /> Copy</>}
        </button>
      </div>
    </div>
  );
}

interface QueryStudioProps {
  headers?: Record<string, string>;
  onAuthError?: () => void;
}

export function QueryStudio({ headers, onAuthError }: QueryStudioProps) {
  const [messages, setMessages] = useState<Message[]>(() => {
    const saved = localStorage.getItem('openrag_chat_history');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {}
    }
    return [];
  });
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeCitations, setActiveCitations] = useState<Citation[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    localStorage.setItem('openrag_chat_history', JSON.stringify(messages));
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);
    setActiveCitations([]);

    const botMsgId = (Date.now() + 1).toString();
    setMessages(prev => [...prev, { id: botMsgId, role: 'assistant', content: '', streaming: true }]);

    try {
      const response = await fetch('/api/v1/query/stream', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          ...headers
        },
        body: JSON.stringify({ text: input, namespace: 'default', top_k: 5 }),
      });

      if (response.status === 401 && onAuthError) {
        onAuthError();
        return;
      }

      if (!response.body) throw new Error('No body in response');
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      
      let done = false;
      let statusLogs = "";

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (!line.trim()) continue;
            
            if (line.startsWith('RESULT:')) {
              const data = JSON.parse(line.substring(7));
              setMessages(prev => prev.map(m => 
                m.id === botMsgId ? { ...m, content: data.answer, citations: data.citations, streaming: false } : m
              ));
              setActiveCitations(data.citations || []);
            } else {
              statusLogs = line; // update with latest log
              setMessages(prev => prev.map(m => 
                m.id === botMsgId ? { ...m, content: `⚙️ ${statusLogs}` } : m
              ));
            }
          }
        }
      }
    } catch (error) {
      console.error(error);
      setMessages(prev => prev.map(m => 
        m.id === botMsgId ? { ...m, content: 'Error connecting to the API.', streaming: false } : m
      ));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="query-studio-container">
      <div className="chat-section">
        <div className="chat-history">
          {messages.length === 0 ? (
            <div className="empty-state">
              <Bot size={48} className="empty-icon" />
              <h2>How can I help you today?</h2>
              <p>Ask a question to search your enterprise knowledge base.</p>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className={`message-wrapper ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? <User size={18} /> : <Bot size={18} />}
                </div>
                <div className="message-content">
                  <div className="message-bubble">
                    {msg.streaming && msg.role === 'assistant' && !msg.content.includes('###') ? (
                      <div className="streaming-indicator">
                        <Loader2 className="spinner" size={16} />
                        {msg.content}
                      </div>
                    ) : (
                      <MessageContent content={msg.content} role={msg.role} />
                    )}
                  </div>
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="message-actions">
                      <button className="btn btn-ghost btn-sm" onClick={() => setActiveCitations(msg.citations!)}>
                        <BookOpen size={14} /> View {msg.citations.length} sources
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <div className="chat-input-container">
          <form className="chat-form" onSubmit={handleSubmit}>
            <input 
              type="text" 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Message OpenRAG..."
              className="chat-input"
              disabled={isLoading}
            />
            <button type="submit" className="chat-submit-btn" disabled={!input.trim() || isLoading}>
              <Send size={18} />
            </button>
          </form>
          <div className="chat-disclaimer">AI can make mistakes. Verify important information.</div>
        </div>
      </div>

      <div className="context-panel">
        <div className="panel-header">
          <h3>Retrieved Context</h3>
        </div>
        <div className="panel-content">
          {activeCitations.length === 0 ? (
            <div className="empty-context">
              <FileText size={32} />
              <p>No active citations.</p>
              <span>Submit a query or click on a message's sources to view them here.</span>
            </div>
          ) : (
            <div className="citations-list">
              {activeCitations.map((cit, idx) => (
                <div key={idx} className="citation-card">
                  <div className="citation-header">
                    <span className="citation-badge">#{idx + 1}</span>
                    <span className="citation-title" title={cit.source_path}>{cit.source_path ? cit.source_path.split('/').pop() : 'Unknown Document'}</span>
                    <span className="citation-score">{(cit.score * 100).toFixed(1)}% match</span>
                  </div>
                  <div className="citation-body">
                    {cit.content ? cit.content.substring(0, 150) : ''}...
                  </div>
                  <div className="citation-footer">
                    Type: {cit.block_type} {cit.page_number && `• Page: ${cit.page_number}`}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
