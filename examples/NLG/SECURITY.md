# Security Policy

## Reporting Security Issues

If you discover a security vulnerability in GP-LoRA, please report it responsibly.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please email the maintainers directly or use GitHub's private vulnerability reporting feature if available.

### What to Include

When reporting a security issue, please include:

- Type of issue (e.g., code injection, unsafe deserialization, etc.)
- Full paths of source file(s) related to the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue

## Security Best Practices

When using GP-LoRA for NLG tasks:

1. **Model Checkpoints**: Only load model checkpoints from trusted sources
2. **Dependencies**: Keep PyTorch and other dependencies updated
3. **Data**: Validate and sanitize training data from external sources
4. **Generated Text**: Be aware that language models may generate harmful content
