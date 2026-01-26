# Security Policy

## 🔒 Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |

## 🚨 Reporting a Vulnerability

If you discover a security vulnerability in DeepGuard, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Email the maintainers directly with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## 🛡️ Security Best Practices

When using DeepGuard:

- **API Keys**: Never commit API keys (e.g., Gemini API key) to version control
- **Input Validation**: The system validates inputs, but always sanitize external data
- **Model Weights**: Only use model weights from trusted sources
- **Dependencies**: Keep dependencies updated to patch known vulnerabilities

## ⚠️ Disclaimer

DeepGuard is a research prototype for deepfake detection. It should be used as a decision-support tool, not as an absolute authority. The system may produce false positives or false negatives depending on input quality, lighting, compression, and other factors.
