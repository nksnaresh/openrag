import React, { useState } from 'react';
import { Lock, User, LogIn, Loader2, AlertCircle } from 'lucide-react';
import './Login.css';

interface LoginProps {
  onLogin: (token: string, username: string, role: string) => void;
}

export function Login({ onLogin }: LoginProps) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);

      const response = await fetch('/api/v1/auth/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Login failed');
      }

      const data = await response.json();
      
      // For the demo, we'll infer the role from the username if the API doesn't return it
      // In production, the token payload or a separate /me endpoint should provide this.
      const role = username.includes('admin') ? 'admin' : 'viewer';
      
      onLogin(data.access_token, username, role);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-header">
          <div className="login-logo">
            <Lock size={32} />
          </div>
          <h1>OpenRAG Enterprise</h1>
          <p>Sign in to access your knowledge base</p>
        </div>

        <form className="login-form" onSubmit={handleSubmit}>
          {error && (
            <div className="login-error">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          <div className="form-group">
            <label>Username</label>
            <div className="input-wrapper">
              <User size={18} />
              <input 
                type="text" 
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
                required 
              />
            </div>
          </div>

          <div className="form-group">
            <label>Password</label>
            <div className="input-wrapper">
              <Lock size={18} />
              <input 
                type="password" 
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password"
                required 
              />
            </div>
          </div>

          <button type="submit" className="btn-login" disabled={isLoading}>
            {isLoading ? <Loader2 className="spinner" size={20} /> : <><LogIn size={20} /> Sign In</>}
          </button>
        </form>

        <div className="login-footer">
          <p>Demo accounts: <strong>admin / admin123</strong> or <strong>viewer / viewer123</strong></p>
        </div>
      </div>
    </div>
  );
}
