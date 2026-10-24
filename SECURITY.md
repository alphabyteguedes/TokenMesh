# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to **security@tokenmesh.dev**.

You should receive a response within 48 hours. If the vulnerability is confirmed, we will:

1. Work on a fix privately
2. Release a patch as soon as possible
3. Credit you in the release notes (unless you prefer to remain anonymous)

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Security Best Practices for Users

- **Never commit `.env` files** containing API keys to version control
- Use the built-in DLP module to prevent accidental key leakage through prompts
- Run TokenMesh behind a reverse proxy (nginx/Caddy) in production
- Keep your TokenMesh installation up to date
- Restrict CORS origins in production (`TOKENMESH_CORS_ORIGINS`)
