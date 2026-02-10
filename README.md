# CommitGen 🚀

AI-powered git commit message generator that works with local models (Ollama) and cloud APIs (Claude, OpenRouter).

## Why CommitGen?

- ✅ **Works offline** with Ollama (privacy-first)
- ✅ **Supports tiny models** (1B/2B params) through smart chunking
- ✅ **Zero Node.js required** - pure Python
- ✅ **Free tier available** via OpenRouter
- ✅ **Conventional commits** format by default
- ✅ **One command** to generate and commit

## Installation

```bash
pip install commitgen
```

Or install from source:
```bash
git clone https://github.com/souravtripathy/commitgen.git
cd commitgen
pip install -e .
```

## Quick Start

### 1. With Ollama (Recommended for local/free usage)

Install Ollama and pull a model:
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2:1b  # Fast, small model
```

Stage your changes and generate commit:
```bash
git add .
commitgen commit
```

### 2. With Claude API

Set up API key:
```bash
commitgen init
# Follow prompts to enter Claude API key
```

Generate commit:
```bash
git add .
commitgen commit
```

### 3. With OpenRouter (Free models)

Get free API key from https://openrouter.ai

```bash
commitgen config --set openrouter.api_key=sk-or-...
commitgen config --set general.provider=openrouter
commitgen commit
```

## Usage

### Basic Commands

```bash
# Generate commit message (interactive)
commitgen commit

# Auto-commit with generated message
commitgen commit --auto

# Amend last commit with better message
commitgen commit --amend

# Use specific provider
commitgen commit --provider claude
commitgen commit --provider ollama --model llama3.2:1b
```

### Configuration

```bash
# Interactive setup
commitgen init

# View current config
commitgen config

# Set specific values
commitgen config --set general.provider=ollama
commitgen config --set ollama.model=qwen2.5:1b
commitgen config --set general.max_length=50

# Get specific value
commitgen config --get general.provider
```

## License

MIT License - see LICENSE file

## Author

Sourav Tripathy
# Test change
