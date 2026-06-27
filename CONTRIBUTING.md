# Contributing to DigitVision AI

Thank you for considering contributing to **DigitVision AI**! Every contribution — from bug fixes to new features and documentation improvements — is warmly welcomed.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Coding Standards](#coding-standards)

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please be kind and constructive in all communications.

---

## How to Contribute

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/CodeAlpha_Handwritten_Character_Recognition.git
   cd CodeAlpha_Handwritten_Character_Recognition
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes**, following the [Coding Standards](#coding-standards)
5. **Test** your changes thoroughly
6. **Commit** with a clear message:
   ```bash
   git commit -m "feat: add XYZ feature"
   ```
7. **Push** to your fork and open a Pull Request

---

## Development Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows

# Install dependencies
pip install -r requirements.txt

# Train the model (required for app and evaluation)
python train.py

# Run the web app
streamlit run app.py
```

---

## Pull Request Guidelines

- Keep PRs focused on a single concern
- Include a clear description of what the PR does and why
- Reference any related issues using `Closes #<issue-number>`
- Ensure your code passes existing tests and add new ones if applicable
- Update the `README.md` if your changes affect usage or project structure

---

## Reporting Bugs

Please open a [GitHub Issue](https://github.com/muhammadabdullah-devpk/CodeAlpha_Handwritten_Character_Recognition/issues) with:

- A clear, descriptive title
- Steps to reproduce the bug
- Expected vs. actual behavior
- Your environment (OS, Python version, package versions)
- Any relevant screenshots or error messages

---

## Suggesting Features

Feature requests are welcome! Open an issue with:

- A clear title and description of the feature
- Why this feature would be useful
- Any implementation ideas you have in mind

---

## Coding Standards

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints where appropriate
- Write clear docstrings for all functions and classes
- Keep functions small and focused (single responsibility)
- Prefer descriptive variable names over abbreviations

---

## Commit Message Convention

We follow a simplified [Conventional Commits](https://www.conventionalcommits.org/) format:

| Prefix | Usage |
|---|---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `style:` | Formatting, no logic change |
| `refactor:` | Code restructuring |
| `test:` | Adding or updating tests |
| `chore:` | Build, CI, or tooling updates |

---

Thank you for making DigitVision AI better! 🚀
