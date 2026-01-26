# Contributing to DeepGuard

Thank you for your interest in contributing to DeepGuard! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/omtripathi52/deepguard/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)

### Suggesting Features

1. Open an issue with the `enhancement` label
2. Describe the feature and its use case
3. Explain how it aligns with DeepGuard's goals

### Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📋 Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/deepguard.git
cd deepguard

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 🧪 Testing

Before submitting a PR, ensure:

```bash
# Test image pipeline
python -m core.image_pipeline --path face_0.jpg

# Test detector module
python core/test_detector.py

# Test face detection
python core/test_face_detector.py
```

## 📝 Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and modular

## 🔒 Security

- Never commit API keys or credentials
- Report security vulnerabilities privately to maintainers

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for helping make DeepGuard better! 🛡️
