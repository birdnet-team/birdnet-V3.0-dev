# Contributing to BirdNET+ V3.0

Thank you for your interest in contributing! This repository is the entry point for the BirdNET+ V3.0 models: reference code, examples, and a demo app. Keep contributions small and focused on that purpose. Full applications belong in [BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer) or the [birdnet](https://github.com/birdnet-team/birdnet) Python package, and model conversion belongs in [model-converter](https://github.com/birdnet-team/model-converter).

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to <ccb-birdnet@cornell.edu>.

## How to Contribute

### Reporting Bugs

- Use a clear and descriptive title.
- Describe the exact steps to reproduce the problem.
- Include the Python version, OS, and the model file (format and precision) you used.
- Attach error tracebacks and, if possible, a short audio file that shows the problem.

### Reporting Model Errors

Wrong or missed detections help us improve future models. Please include the audio (or a link), the time range, the expected species, and the model file and settings you used.

### Pull Requests

1. Fork the repo and create a branch from `main`.
2. Keep examples short and self-explanatory; prefer one clear script over a configurable framework.
3. Run `ruff check .` and `ruff format .`, and make sure `python analyze.py examples/audio/soundscape.wav` still works.
4. Keep commits small and focused; write clear commit messages.
5. Open a pull request with a description of what changed and why.

### AI-Assisted Contributions

We welcome contributions that use AI coding assistants, but please keep PRs **focused on a single, isolated change**. In particular:

- **Do not** submit PRs that bundle AI-generated reformatting, renaming, or stylistic changes alongside functional work.
- **Do** verify that AI-generated code is correct and that you understand what it does.
- **Do** mention in the PR description if AI tools were used.

## Development Setup

```bash
git clone https://github.com/birdnet-team/birdnet-V3.0-dev.git
cd birdnet-V3.0-dev
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt ruff
```

## Style Guide

- Use American English in code, comments, docs, and commit messages.
- Follow [PEP 8](https://peps.python.org/pep-0008/); `ruff` enforces the rules in `ruff.toml` (120-character lines).
- Use type hints for function signatures and docstrings for public functions.

## Commit Messages

- Use the present tense and imperative mood ("Add feature", not "Added feature").
- Keep the first line under 72 characters.
- Reference issues with `#123` where applicable.
