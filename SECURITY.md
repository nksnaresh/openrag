# Security Policy

## Supported Versions

OpenRAG is currently in active development. We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## Reporting a Vulnerability

We take the security of OpenRAG seriously. If you believe you have found a security vulnerability, please report it to us responsibly.

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, please send an email to **[security@openrag.io](mailto:security@openrag.io)** (Note: This is a placeholder, please update with your actual security contact).

### What to Include
- A detailed description of the vulnerability.
- Steps to reproduce the issue.
- Potential impact of the vulnerability.
- Any suggested fixes or mitigations.

## Security Governance

OpenRAG implements several layers of security to ensure enterprise readiness:

### 1. Authentication & Authorization
- **JWT (JSON Web Tokens)**: All API access is secured via stateless JWT tokens.
- **RBAC (Role-Based Access Control)**: Enforced at both the API and UI layers. 
  - `admin`: Full system access.
  - `viewer`: Read-only/Query-only access.

### 2. Data Protection
- **Password Hashing**: We use `PBKDF2-SHA256` (via Passlib) for secure credential storage.
- **Secure Communication**: It is highly recommended to deploy OpenRAG behind a reverse proxy (like Nginx or Caddy) with TLS/SSL enabled.

### 3. Secret Management
- **Environment Variables**: Sensitive information such as `GEMINI_API_KEY` and the JWT `SECRET_KEY` should ALWAYS be managed via environment variables or a secure secret manager (e.g., AWS Secrets Manager, HashiCorp Vault).
- **Proactive Warnings**: Do not hardcode internal secrets in `auth.py` for production deployments.

## Vulnerability Disclosure Process

1. **Acknowledgment**: We will acknowledge receipt of your report within 48 hours.
2. **Investigation**: We will investigate the issue and determine its severity.
3. **Fix**: If a vulnerability is confirmed, we will develop a fix and test it thoroughly.
4. **Notification**: We will notify the reporter once the fix is ready for release.
5. **Release**: A new version containing the security fix will be released.
6. **Public Announcement**: After the fix is deployed and verified, a public announcement may be made (with credit to the reporter, if desired).

---

*Thank you for helping us keep OpenRAG secure!*
