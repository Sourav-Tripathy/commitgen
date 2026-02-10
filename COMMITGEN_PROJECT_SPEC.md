# CommitGen - AI Commit Message Generator (Python)

## Project Overview

A Python CLI tool that generates meaningful git commit messages using AI. Supports local models via Ollama (including small 1B/2B models), Claude API, and OpenRouter for free model access.

---

## Core Features

### 1. Multi-Provider Support
- **Ollama (Local)**: Works with small models (1B/2B) via chunking for large diffs
- **Claude API**: High-quality commit messages
- **OpenRouter**: Free model access with API key
- **Auto-fallback**: Tries local first, then cloud if configured

### 2. Smart Diff Handling
- Handles large diffs by chunking for small models
- Filters out generated files, lock files, minified code
- Summarizes file changes before generating commit message
- Supports conventional commit format (feat:, fix:, chore:, etc.)

### 3. Developer Experience
- Simple installation: `pip install commitgen`
- Works immediately with Ollama if installed
- Interactive config setup on first run
- Supports git hooks for automatic commits

---

## Project Structure

```
commitgen/
├── commitgen/
│   ├── __init__.py
│   ├── __main__.py                 # Entry point for python -m commitgen
│   ├── cli.py                      # Click-based CLI interface
│   ├── core/
│   │   ├── __init__.py
│   │   ├── git_handler.py         # Git operations and diff extraction
│   │   ├── diff_processor.py      # Diff filtering and chunking
│   │   ├── message_generator.py   # Core message generation logic
│   │   └── config_manager.py      # Configuration handling
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract base provider
│   │   ├── ollama_provider.py     # Ollama integration with chunking
│   │   ├── claude_provider.py     # Claude API integration
│   │   └── openrouter_provider.py # OpenRouter integration
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── system_prompts.py      # System prompts for different scenarios
│   │   └── templates.py           # Commit message templates
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # Logging configuration
│       └── validators.py          # Input validation
├── tests/
│   ├── __init__.py
│   ├── test_git_handler.py
│   ├── test_diff_processor.py
│   ├── test_providers.py
│   └── fixtures/
│       └── sample_diffs.py
├── .gitignore
├── pyproject.toml                  # Modern Python packaging
├── setup.py                        # Fallback for older pip versions
├── README.md
├── LICENSE
└── CONTRIBUTING.md
```

---

## Detailed File Specifications

### 1. `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "commitgen"
version = "0.1.0"
description = "AI-powered git commit message generator with local and cloud support"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["git", "commit", "ai", "ollama", "claude", "cli"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "click>=8.0.0",
    "gitpython>=3.1.0",
    "ollama>=0.1.0",
    "anthropic>=0.18.0",
    "openai>=1.0.0",  # For OpenRouter compatibility
    "pydantic>=2.0.0",
    "rich>=13.0.0",  # Beautiful terminal output
    "toml>=0.10.0",
    "tiktoken>=0.5.0",  # Token counting
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.scripts]
commitgen = "commitgen.cli:main"
cgen = "commitgen.cli:main"  # Short alias

[project.urls]
Homepage = "https://github.com/yourusername/commitgen"
Documentation = "https://github.com/yourusername/commitgen#readme"
Repository = "https://github.com/yourusername/commitgen"
Issues = "https://github.com/yourusername/commitgen/issues"

[tool.setuptools.packages.find]
where = ["."]
include = ["commitgen*"]

[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311']

[tool.ruff]
line-length = 100
target-version = "py38"

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

---

### 2. `commitgen/__init__.py`

```python
"""
CommitGen - AI-powered git commit message generator
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

from commitgen.core.message_generator import generate_commit_message
from commitgen.core.config_manager import ConfigManager

__all__ = ["generate_commit_message", "ConfigManager"]
```

---

### 3. `commitgen/__main__.py`

```python
"""
Entry point for running commitgen as a module: python -m commitgen
"""

from commitgen.cli import main

if __name__ == "__main__":
    main()
```

---

### 4. `commitgen/cli.py`

**Purpose**: Main CLI interface using Click

**Key Functions**:
- `main()`: Entry point
- `commit()`: Generate commit message
- `config()`: Manage configuration
- `init()`: Initialize config interactively
- `install-hook()`: Install git hook

**Commands**:
```bash
commitgen commit              # Generate and show commit message
commitgen commit --auto       # Generate and commit automatically
commitgen commit --amend      # Improve last commit message
commitgen config              # Show current config
commitgen config --set provider=ollama
commitgen config --set model=llama3.2:1b
commitgen init                # Interactive setup wizard
commitgen install-hook        # Install pre-commit hook
```

**Code Structure**:
```python
import click
from rich.console import Console
from rich.prompt import Prompt, Confirm
from commitgen.core.message_generator import generate_commit_message
from commitgen.core.config_manager import ConfigManager
from commitgen.core.git_handler import GitHandler

console = Console()

@click.group()
@click.version_option()
def main():
    """AI-powered git commit message generator"""
    pass

@main.command()
@click.option('--auto', is_flag=True, help='Automatically commit with generated message')
@click.option('--amend', is_flag=True, help='Amend the last commit')
@click.option('--provider', type=str, help='Override configured provider')
@click.option('--model', type=str, help='Override configured model')
def commit(auto, amend, provider, model):
    """Generate a commit message for staged changes"""
    # Implementation here
    pass

@main.command()
@click.option('--set', 'set_value', type=str, help='Set config value (key=value)')
@click.option('--get', 'get_key', type=str, help='Get config value')
def config(set_value, get_key):
    """Manage commitgen configuration"""
    # Implementation here
    pass

@main.command()
def init():
    """Interactive configuration wizard"""
    # Implementation here
    pass

@main.command()
def install_hook():
    """Install git pre-commit hook"""
    # Implementation here
    pass

if __name__ == "__main__":
    main()
```

---

### 5. `commitgen/core/config_manager.py`

**Purpose**: Handle configuration loading, saving, and validation

**Configuration File Location**: `~/.config/commitgen/config.toml`

**Default Configuration**:
```toml
[general]
provider = "ollama"  # ollama, claude, openrouter
max_length = 72  # First line max length
conventional_commits = true
language = "en"

[ollama]
model = "llama3.2:1b"  # Default to small model
host = "http://localhost:11434"
timeout = 30
max_tokens_per_chunk = 1000  # For small models

[claude]
api_key = ""  # User must set this
model = "claude-sonnet-4-5-20250929"
max_tokens = 1024

[openrouter]
api_key = ""  # User must set this
model = "google/gemma-2-9b-it:free"  # Free model
site_url = ""  # Optional for rankings
site_name = "commitgen"

[advanced]
filter_generated_files = true
filter_lock_files = true
filter_minified = true
chunk_large_diffs = true  # Required for small models
```

**Code Structure**:
```python
from pathlib import Path
from typing import Dict, Any, Optional
import toml
from pydantic import BaseModel, Field

class OllamaConfig(BaseModel):
    model: str = "llama3.2:1b"
    host: str = "http://localhost:11434"
    timeout: int = 30
    max_tokens_per_chunk: int = 1000

class ClaudeConfig(BaseModel):
    api_key: str = ""
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 1024

class OpenRouterConfig(BaseModel):
    api_key: str = ""
    model: str = "google/gemma-2-9b-it:free"
    site_url: str = ""
    site_name: str = "commitgen"

class GeneralConfig(BaseModel):
    provider: str = "ollama"
    max_length: int = 72
    conventional_commits: bool = True
    language: str = "en"

class AdvancedConfig(BaseModel):
    filter_generated_files: bool = True
    filter_lock_files: bool = True
    filter_minified: bool = True
    chunk_large_diffs: bool = True

class Config(BaseModel):
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)

class ConfigManager:
    def __init__(self):
        self.config_dir = Path.home() / ".config" / "commitgen"
        self.config_file = self.config_dir / "config.toml"
        
    def load(self) -> Config:
        """Load configuration from file or create default"""
        pass
    
    def save(self, config: Config) -> None:
        """Save configuration to file"""
        pass
    
    def get(self, key: str) -> Any:
        """Get a specific config value"""
        pass
    
    def set(self, key: str, value: Any) -> None:
        """Set a specific config value"""
        pass
```

---

### 6. `commitgen/core/git_handler.py`

**Purpose**: Git operations - extract diff, staged files, commit history

**Key Functions**:
- `get_staged_diff()`: Get diff of staged changes
- `get_staged_files()`: List staged files with change type
- `get_last_commit_message()`: Get last commit message
- `commit_with_message(message)`: Perform git commit
- `amend_commit(message)`: Amend last commit

**Code Structure**:
```python
from typing import List, Tuple, Optional
from pathlib import Path
import git
from git import Repo

class GitHandler:
    def __init__(self, repo_path: Optional[str] = None):
        """Initialize with repository path (defaults to current directory)"""
        self.repo_path = repo_path or "."
        self.repo = Repo(self.repo_path, search_parent_directories=True)
    
    def get_staged_diff(self) -> str:
        """
        Get diff of staged changes
        Returns: Full diff string
        """
        pass
    
    def get_staged_files(self) -> List[Tuple[str, str]]:
        """
        Get list of staged files with their change type
        Returns: List of (filename, change_type) tuples
        change_type: A=Added, M=Modified, D=Deleted, R=Renamed
        """
        pass
    
    def get_branch_name(self) -> str:
        """Get current branch name"""
        pass
    
    def get_last_commit_message(self) -> str:
        """Get the last commit message"""
        pass
    
    def commit_with_message(self, message: str) -> bool:
        """Execute git commit with message"""
        pass
    
    def amend_commit(self, message: str) -> bool:
        """Amend the last commit with new message"""
        pass
    
    def has_staged_changes(self) -> bool:
        """Check if there are staged changes"""
        pass
```

---

### 7. `commitgen/core/diff_processor.py`

**Purpose**: Process and filter diffs, handle chunking for small models

**Key Functions**:
- `filter_diff()`: Remove generated files, lock files, minified code
- `chunk_diff()`: Split large diffs for small context models
- `summarize_changes()`: Create file-level summary
- `estimate_tokens()`: Estimate token count

**Code Structure**:
```python
from typing import List, Dict, Tuple
import re
import tiktoken

class DiffProcessor:
    def __init__(self, max_tokens_per_chunk: int = 1000):
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT tokenizer
        
        # Patterns for filtering
        self.generated_patterns = [
            r'.*\.min\.js$',
            r'.*\.min\.css$',
            r'package-lock\.json$',
            r'yarn\.lock$',
            r'poetry\.lock$',
            r'Pipfile\.lock$',
            r'.*-lock\.json$',
            r'.*\.bundle\.js$',
            r'dist/.*',
            r'build/.*',
            r'.*\.generated\..*',
        ]
    
    def filter_diff(self, diff: str, files: List[Tuple[str, str]]) -> str:
        """
        Filter out generated files, lock files from diff
        Args:
            diff: Full git diff
            files: List of (filename, change_type)
        Returns:
            Filtered diff
        """
        pass
    
    def chunk_diff(self, diff: str) -> List[str]:
        """
        Split diff into chunks that fit in small model context
        Intelligently splits by file, then by hunk if needed
        
        Args:
            diff: Filtered git diff
        Returns:
            List of diff chunks
        """
        pass
    
    def summarize_changes(self, files: List[Tuple[str, str]]) -> str:
        """
        Create high-level summary of changes
        
        Example output:
        - Added: 3 files (authentication.py, user_model.py, test_auth.py)
        - Modified: 2 files (config.py, README.md)
        - Deleted: 1 file (legacy_auth.py)
        """
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return len(self.encoding.encode(text))
    
    def extract_file_diffs(self, diff: str) -> Dict[str, str]:
        """
        Extract individual file diffs from combined diff
        Returns: Dict mapping filename -> file diff
        """
        pass
```

---

### 8. `commitgen/providers/base.py`

**Purpose**: Abstract base class for all AI providers

```python
from abc import ABC, abstractmethod
from typing import List, Optional

class BaseProvider(ABC):
    """Base class for AI providers"""
    
    def __init__(self, config: dict):
        self.config = config
    
    @abstractmethod
    def generate_commit_message(
        self, 
        diff: str, 
        file_summary: str,
        context: Optional[str] = None
    ) -> str:
        """
        Generate commit message from diff
        
        Args:
            diff: Git diff content
            file_summary: High-level summary of changed files
            context: Optional additional context
            
        Returns:
            Generated commit message
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model being used"""
        pass
```

---

### 9. `commitgen/providers/ollama_provider.py`

**Purpose**: Ollama integration with smart chunking for small models (1B/2B)

**Key Features**:
- Detects model size and adjusts strategy
- For small models (< 3B params): chunks diff, generates per-file messages, combines
- For large models (>= 3B params): sends full diff
- Auto-fallback if Ollama not running

**Code Structure**:
```python
from typing import List, Optional
import ollama
from commitgen.providers.base import BaseProvider
from commitgen.core.diff_processor import DiffProcessor

class OllamaProvider(BaseProvider):
    def __init__(self, config: dict):
        super().__init__(config)
        self.client = ollama.Client(host=config.get('host', 'http://localhost:11434'))
        self.model = config.get('model', 'llama3.2:1b')
        self.max_tokens_per_chunk = config.get('max_tokens_per_chunk', 1000)
        self.diff_processor = DiffProcessor(self.max_tokens_per_chunk)
        
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            self.client.list()
            return True
        except Exception:
            return False
    
    def get_model_name(self) -> str:
        return self.model
    
    def _is_small_model(self) -> bool:
        """
        Detect if model is small (<3B parameters) based on name
        Small models: llama3.2:1b, qwen2.5:0.5b, phi3:mini
        """
        small_indicators = ['1b', '2b', '0.5b', 'mini', 'small']
        return any(indicator in self.model.lower() for indicator in small_indicators)
    
    def _generate_with_chunking(
        self, 
        diff: str, 
        file_summary: str
    ) -> str:
        """
        For small models: chunk diff, generate messages per chunk, combine
        
        Strategy:
        1. Split diff by file
        2. Generate commit message for each file
        3. Combine into single coherent message
        """
        pass
    
    def _generate_direct(
        self, 
        diff: str, 
        file_summary: str
    ) -> str:
        """For large models: send full diff"""
        pass
    
    def generate_commit_message(
        self, 
        diff: str, 
        file_summary: str,
        context: Optional[str] = None
    ) -> str:
        """
        Main entry point - routes to chunking or direct based on model size
        """
        if self._is_small_model():
            return self._generate_with_chunking(diff, file_summary)
        else:
            return self._generate_direct(diff, file_summary)
    
    def _call_ollama(self, prompt: str, system_prompt: str) -> str:
        """Make API call to Ollama"""
        response = self.client.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            options={
                'temperature': 0.7,
                'num_predict': 200,  # Max tokens for commit message
            }
        )
        return response['message']['content']
```

**Chunking Strategy for Small Models**:
```
Large diff (5000 tokens) -> Split by file
  File 1 (auth.py): 1500 tokens
  File 2 (config.py): 800 tokens
  File 3 (tests.py): 2700 tokens -> Split further

Generate per-file:
  - auth.py: "Add JWT authentication"
  - config.py: "Update database config"
  - tests.py: "Add integration tests for auth"

Combine with final prompt:
  "Given these file-level changes:
   - auth.py: Add JWT authentication
   - config.py: Update database config
   - tests.py: Add integration tests for auth
   
   Generate a single conventional commit message"

Output: "feat(auth): implement JWT authentication with tests"
```

---

### 10. `commitgen/providers/claude_provider.py`

**Purpose**: Claude API integration

**Code Structure**:
```python
from typing import Optional
from anthropic import Anthropic
from commitgen.providers.base import BaseProvider

class ClaudeProvider(BaseProvider):
    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', 'claude-sonnet-4-5-20250929')
        self.max_tokens = config.get('max_tokens', 1024)
        
        if self.api_key:
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None
    
    def is_available(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key and self.client)
    
    def get_model_name(self) -> str:
        return self.model
    
    def generate_commit_message(
        self, 
        diff: str, 
        file_summary: str,
        context: Optional[str] = None
    ) -> str:
        """Generate commit message using Claude API"""
        if not self.is_available():
            raise ValueError("Claude API key not configured")
        
        system_prompt = """You are an expert at writing git commit messages.
Generate a concise, informative commit message following conventional commits format.

Format: <type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, test, chore
Keep first line under 72 characters.
Be specific but concise."""

        user_prompt = f"""File changes summary:
{file_summary}

Git diff:
{diff}

Generate a conventional commit message for these changes."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            system=system_prompt
        )
        
        return response.content[0].text.strip()
```

---

### 11. `commitgen/providers/openrouter_provider.py`

**Purpose**: OpenRouter integration for free model access

**Code Structure**:
```python
from typing import Optional
from openai import OpenAI
from commitgen.providers.base import BaseProvider

class OpenRouterProvider(BaseProvider):
    """
    OpenRouter provider - access to free models like:
    - google/gemma-2-9b-it:free
    - meta-llama/llama-3.1-8b-instruct:free
    - microsoft/phi-3-mini-128k-instruct:free
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', 'google/gemma-2-9b-it:free')
        self.site_url = config.get('site_url', '')
        self.site_name = config.get('site_name', 'commitgen')
        
        if self.api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )
        else:
            self.client = None
    
    def is_available(self) -> bool:
        return bool(self.api_key and self.client)
    
    def get_model_name(self) -> str:
        return self.model
    
    def generate_commit_message(
        self, 
        diff: str, 
        file_summary: str,
        context: Optional[str] = None
    ) -> str:
        """Generate commit message using OpenRouter API"""
        if not self.is_available():
            raise ValueError("OpenRouter API key not configured")
        
        system_prompt = """You are an expert at writing git commit messages.
Generate a concise, informative commit message following conventional commits format.

Format: <type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, test, chore
Keep first line under 72 characters."""

        user_prompt = f"""File changes:
{file_summary}

Diff:
{diff[:3000]}

Generate a conventional commit message."""

        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            extra_headers=extra_headers,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
```

---

### 12. `commitgen/core/message_generator.py`

**Purpose**: Core orchestration - combines git handler, diff processor, and providers

**Code Structure**:
```python
from typing import Optional
from commitgen.core.git_handler import GitHandler
from commitgen.core.diff_processor import DiffProcessor
from commitgen.core.config_manager import ConfigManager
from commitgen.providers.ollama_provider import OllamaProvider
from commitgen.providers.claude_provider import ClaudeProvider
from commitgen.providers.openrouter_provider import OpenRouterProvider

class MessageGenerator:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.load()
        self.git_handler = GitHandler()
        self.diff_processor = DiffProcessor(
            max_tokens_per_chunk=self.config.ollama.max_tokens_per_chunk
        )
        
    def _get_provider(self, provider_override: Optional[str] = None):
        """Get appropriate provider based on config or override"""
        provider_name = provider_override or self.config.general.provider
        
        if provider_name == "ollama":
            return OllamaProvider(self.config.ollama.dict())
        elif provider_name == "claude":
            return ClaudeProvider(self.config.claude.dict())
        elif provider_name == "openrouter":
            return OpenRouterProvider(self.config.openrouter.dict())
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    def generate(
        self, 
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> str:
        """
        Main generation flow:
        1. Get staged diff
        2. Filter and process diff
        3. Generate file summary
        4. Call AI provider
        5. Validate and format result
        """
        # Check for staged changes
        if not self.git_handler.has_staged_changes():
            raise ValueError("No staged changes found. Use 'git add' first.")
        
        # Get diff and files
        raw_diff = self.git_handler.get_staged_diff()
        staged_files = self.git_handler.get_staged_files()
        
        # Filter if enabled
        if self.config.advanced.filter_generated_files:
            diff = self.diff_processor.filter_diff(raw_diff, staged_files)
        else:
            diff = raw_diff
        
        # Generate summary
        file_summary = self.diff_processor.summarize_changes(staged_files)
        
        # Get provider
        provider = self._get_provider(provider_override)
        
        # Override model if specified
        if model_override:
            provider.model = model_override
        
        # Check availability
        if not provider.is_available():
            raise ValueError(
                f"Provider '{provider_override or self.config.general.provider}' "
                f"is not available. Check configuration."
            )
        
        # Generate message
        commit_message = provider.generate_commit_message(diff, file_summary)
        
        # Validate and format
        commit_message = self._validate_and_format(commit_message)
        
        return commit_message
    
    def _validate_and_format(self, message: str) -> str:
        """Ensure message follows conventions and length limits"""
        lines = message.strip().split('\n')
        first_line = lines[0]
        
        # Enforce max length on first line
        if len(first_line) > self.config.general.max_length:
            first_line = first_line[:self.config.general.max_length]
            lines[0] = first_line
        
        # Ensure conventional commits format if enabled
        if self.config.general.conventional_commits:
            if not self._is_conventional_format(first_line):
                # Try to fix by adding "chore: " prefix
                first_line = f"chore: {first_line}"
                lines[0] = first_line
        
        return '\n'.join(lines)
    
    def _is_conventional_format(self, line: str) -> bool:
        """Check if line follows conventional commits format"""
        import re
        pattern = r'^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+\))?: .+'
        return bool(re.match(pattern, line))


def generate_commit_message(
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """
    Convenience function for generating commit messages
    
    Args:
        provider: Override configured provider
        model: Override configured model
        
    Returns:
        Generated commit message
    """
    config_manager = ConfigManager()
    generator = MessageGenerator(config_manager)
    return generator.generate(provider_override=provider, model_override=model)
```

---

### 13. `commitgen/prompts/system_prompts.py`

**Purpose**: Store all AI prompts in one place

```python
"""System prompts for different AI providers and scenarios"""

# Base prompt for all providers
BASE_SYSTEM_PROMPT = """You are an expert at writing git commit messages.
Generate concise, informative commit messages following the conventional commits format.

Format: <type>(<scope>): <description>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes (formatting, etc.)
- refactor: Code refactoring
- test: Adding or updating tests
- chore: Maintenance tasks
- perf: Performance improvements
- ci: CI/CD changes
- build: Build system changes

Rules:
1. Keep the first line under 72 characters
2. Use imperative mood ("add" not "added")
3. Don't capitalize the first letter after the colon
4. No period at the end of the subject line
5. Be specific but concise
6. Focus on WHAT changed and WHY, not HOW"""

# For small models with limited context
SMALL_MODEL_FILE_PROMPT = """Given this git diff for a single file, generate a brief description of what changed.
Focus on the main purpose of the change. Keep it to one sentence, under 10 words.

Example outputs:
- "Add user authentication"
- "Fix null pointer in login"
- "Update README with examples"
- "Refactor database queries"

Diff:
{diff}

Brief description:"""

SMALL_MODEL_COMBINE_PROMPT = """Given these file-level change descriptions, generate a single conventional commit message.

Changes:
{file_descriptions}

Generate a commit message in this format: <type>(<scope>): <description>

Commit message:"""

# For combining multiple chunks
CHUNK_COMBINE_PROMPT = """You are combining information from multiple code changes.
Generate a single, coherent conventional commit message that summarizes all changes.

Individual change descriptions:
{chunk_summaries}

Generate one conventional commit message that captures the overall change:"""

# For amending commits
AMEND_PROMPT = """Improve this commit message to be more clear and follow conventional commits format.

Current message:
{current_message}

Recent changes:
{file_summary}

Generate an improved commit message:"""
```

---

### 14. `commitgen/prompts/templates.py`

**Purpose**: Prompt templates with variable substitution

```python
"""Prompt templates for different scenarios"""

from string import Template

class PromptTemplates:
    """Manages prompt templates with variable substitution"""
    
    @staticmethod
    def full_diff_prompt(file_summary: str, diff: str) -> str:
        """Prompt for full diff (large models)"""
        template = Template("""File changes summary:
$file_summary

Git diff:
$diff

Generate a conventional commit message for these changes. 
Keep the first line under 72 characters.

Commit message:""")
        return template.substitute(file_summary=file_summary, diff=diff)
    
    @staticmethod
    def file_diff_prompt(filename: str, diff: str) -> str:
        """Prompt for single file diff (small models)"""
        template = Template("""File: $filename

Changes:
$diff

Briefly describe what changed in this file (one sentence, under 10 words):""")
        return template.substitute(filename=filename, diff=diff)
    
    @staticmethod
    def combine_files_prompt(file_descriptions: str) -> str:
        """Prompt to combine file-level descriptions"""
        template = Template("""These files were changed:
$file_descriptions

Generate a single conventional commit message that summarizes all these changes.
Format: <type>(<scope>): <description>

Commit message:""")
        return template.substitute(file_descriptions=file_descriptions)
    
    @staticmethod
    def amend_commit_prompt(current_message: str, file_summary: str) -> str:
        """Prompt to improve existing commit message"""
        template = Template("""Current commit message:
$current_message

Changed files:
$file_summary

Improve this commit message to be clearer and follow conventional commits format.

Improved commit message:""")
        return template.substitute(
            current_message=current_message,
            file_summary=file_summary
        )
```

---

### 15. `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db

# Config (don't commit user configs)
config.toml
```

---

### 16. `README.md` (Comprehensive)

```markdown
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
git clone https://github.com/yourusername/commitgen.git
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

### Git Hook (Auto-generate on commit)

```bash
# Install pre-commit hook
commitgen install-hook

# Now regular git commits will show AI suggestions
git commit  # Opens editor with AI-generated message
```

## Configuration File

Location: `~/.config/commitgen/config.toml`

Example configuration:
```toml
[general]
provider = "ollama"
max_length = 72
conventional_commits = true

[ollama]
model = "llama3.2:1b"
host = "http://localhost:11434"
max_tokens_per_chunk = 1000

[claude]
api_key = "sk-ant-..."
model = "claude-sonnet-4-5-20250929"

[openrouter]
api_key = "sk-or-..."
model = "google/gemma-2-9b-it:free"
```

## Supported Models

### Ollama (Local)
- **Small (< 2GB)**: `llama3.2:1b`, `qwen2.5:0.5b`, `phi3:mini`
- **Medium (2-8GB)**: `llama3.2:3b`, `mistral:7b`, `qwen2.5:7b`
- **Large (> 8GB)**: `llama3.1:70b`, `qwen2.5:32b`

### Claude API
- `claude-sonnet-4-5-20250929` (recommended)
- `claude-opus-4-5-20251101`
- `claude-haiku-4-5-20251001`

### OpenRouter (Free Tier)
- `google/gemma-2-9b-it:free`
- `meta-llama/llama-3.1-8b-instruct:free`
- `microsoft/phi-3-mini-128k-instruct:free`

## How It Works

### For Small Models (1B/2B)

CommitGen uses **intelligent chunking** to work with small models:

1. **Extract diff** for staged changes
2. **Split by file** - process each file separately
3. **Generate per-file descriptions** - "Add authentication", "Fix bug in parser"
4. **Combine** - Final prompt combines file descriptions into one commit message

Example:
```
Files changed:
- auth.py: "Add JWT token validation"
- config.py: "Update database URL"
- tests.py: "Add auth tests"

→ Final message: "feat(auth): implement JWT authentication with tests"
```

### For Large Models (7B+)

Sends the full diff directly - no chunking needed.

## Examples

### Example 1: Feature Addition
```bash
$ git add src/auth.py tests/test_auth.py
$ commitgen commit

Generated message:
feat(auth): implement JWT authentication

- Add token validation middleware
- Add user authentication endpoints
- Add integration tests for auth flow

[y/N/e(dit)]: y
```

### Example 2: Bug Fix
```bash
$ git add src/parser.py
$ commitgen commit --auto

✓ Committed: fix(parser): handle null values in JSON parsing
```

### Example 3: Multiple Files
```bash
$ git add .
$ commitgen commit --provider claude

Generated message:
refactor(core): improve error handling across modules

- Add custom exception classes
- Update error messages for clarity
- Add error logging in critical paths

[y/N/e(dit)]: y
```

## Comparison with Other Tools

| Feature | CommitGen | ai-commit (npm) | Other tools |
|---------|-----------|-----------------|-------------|
| Installation | `pip install` | `npm install` | Varies |
| Local AI | ✅ Ollama | ❌ | Some |
| Small models (1B-2B) | ✅ Smart chunking | ❌ | ❌ |
| Free tier | ✅ OpenRouter | ❌ | Some |
| No API key required | ✅ With Ollama | ❌ | ❌ |
| Language | Python | JavaScript | Varies |

## Troubleshooting

### "No staged changes found"
```bash
# Stage your changes first
git add <files>
```

### "Ollama not available"
```bash
# Check if Ollama is running
ollama list

# Start Ollama
ollama serve

# Pull a model if needed
ollama pull llama3.2:1b
```

### "Provider not configured"
```bash
# Run interactive setup
commitgen init

# Or manually set API key
commitgen config --set claude.api_key=sk-ant-...
```

### Generated message too long
```bash
# Reduce max length
commitgen config --set general.max_length=50
```

## Development

```bash
# Clone repo
git clone https://github.com/yourusername/commitgen.git
cd commitgen

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black commitgen/
ruff check commitgen/

# Type checking
mypy commitgen/
```

## Roadmap

- [ ] Support for more providers (Gemini, Cohere)
- [ ] Custom prompt templates
- [ ] Multi-language commit messages
- [ ] Integration with GitHub CLI
- [ ] VS Code extension
- [ ] Commit message history/suggestions

## Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - see LICENSE file

## Author

Your Name - [@yourhandle](https://twitter.com/yourhandle)

## Acknowledgments

- Inspired by [ai-commit](https://github.com/insulineru/ai-commit)
- Built with [Ollama](https://ollama.ai), [Claude](https://anthropic.com), and [OpenRouter](https://openrouter.ai)
```

---

### 17. `tests/test_diff_processor.py` (Example test file)

```python
import pytest
from commitgen.core.diff_processor import DiffProcessor

class TestDiffProcessor:
    @pytest.fixture
    def processor(self):
        return DiffProcessor(max_tokens_per_chunk=1000)
    
    def test_filter_lock_files(self, processor):
        """Test that lock files are filtered out"""
        files = [
            ("src/main.py", "M"),
            ("package-lock.json", "M"),
            ("poetry.lock", "M"),
            ("README.md", "M")
        ]
        
        # Mock diff with these files
        diff = """diff --git a/src/main.py b/src/main.py
+++ changed
diff --git a/package-lock.json b/package-lock.json
+++ lock file changes"""
        
        filtered = processor.filter_diff(diff, files)
        
        assert "package-lock.json" not in filtered
        assert "src/main.py" in filtered
    
    def test_summarize_changes(self, processor):
        """Test change summary generation"""
        files = [
            ("src/auth.py", "A"),
            ("src/config.py", "M"),
            ("old_file.py", "D")
        ]
        
        summary = processor.summarize_changes(files)
        
        assert "Added: 1 file" in summary or "auth.py" in summary
        assert "Modified: 1 file" in summary or "config.py" in summary
        assert "Deleted: 1 file" in summary or "old_file.py" in summary
    
    def test_chunk_large_diff(self, processor):
        """Test that large diffs are chunked appropriately"""
        # Create a large diff
        large_diff = "diff --git a/file.py b/file.py\n" + ("+line\n" * 2000)
        
        chunks = processor.chunk_diff(large_diff)
        
        assert len(chunks) > 1  # Should be split
        for chunk in chunks:
            # Each chunk should be under token limit
            tokens = processor.estimate_tokens(chunk)
            assert tokens <= processor.max_tokens_per_chunk * 1.2  # 20% tolerance
```

---

## Additional Implementation Notes

### Smart Chunking Algorithm (For Small Models)

```python
# Pseudo-code for chunking strategy
def chunk_diff_for_small_model(diff, max_tokens=1000):
    """
    Strategy:
    1. Split diff by file (each file is a natural boundary)
    2. If single file > max_tokens, split by hunks (@@ markers)
    3. If single hunk > max_tokens, split by lines (less ideal)
    """
    
    files = extract_file_diffs(diff)
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for filename, file_diff in files.items():
        file_tokens = estimate_tokens(file_diff)
        
        if file_tokens > max_tokens:
            # File too large - split by hunks
            hunks = split_by_hunks(file_diff)
            for hunk in hunks:
                chunks.append(hunk)
        else:
            # File fits - add to current chunk
            if current_tokens + file_tokens > max_tokens:
                # Current chunk full - start new one
                chunks.append('\n'.join(current_chunk))
                current_chunk = [file_diff]
                current_tokens = file_tokens
            else:
                current_chunk.append(file_diff)
                current_tokens += file_tokens
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks
```

### Error Handling Strategy

```python
# In message_generator.py
def generate_with_fallback(self):
    """Try multiple providers with graceful fallback"""
    
    providers = [
        ('ollama', OllamaProvider),
        ('openrouter', OpenRouterProvider),
        ('claude', ClaudeProvider)
    ]
    
    errors = []
    
    for provider_name, ProviderClass in providers:
        try:
            provider = ProviderClass(self.config)
            if provider.is_available():
                return provider.generate_commit_message(diff, summary)
        except Exception as e:
            errors.append(f"{provider_name}: {str(e)}")
    
    # All providers failed
    raise RuntimeError(
        f"All providers failed:\n" + "\n".join(errors)
    )
```

---

## Installation & Publishing Workflow

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in editable mode
pip install -e ".[dev]"

# Run tests
pytest

# Test the CLI
commitgen --help
```

### Publishing to PyPI

```bash
# Build package
python -m build

# Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ commitgen

# If all good, upload to PyPI
python -m twine upload dist/*
```

---

## Launch Strategy

### Day 1: Product Hunt
- Post with title: "CommitGen - AI commit messages that work offline with 1B models"
- Highlight: Privacy-first, works without API keys, Python alternative to ai-commit
- Demo: GIF showing `commitgen commit` in action

### Day 1-3: Twitter/X
- Thread explaining the chunking algorithm for small models
- Comparison table vs ai-commit
- Code snippets showing usage

### Week 1: Reddit
- r/python - "Built a Python alternative to ai-commit"
- r/programming - "How I made AI commit messages work with 1B parameter models"
- r/ollama - "New CLI tool for commit messages using Ollama"

### Week 2: Dev.to / Hashnode
- Article: "Building an AI commit message generator that respects your privacy"
- Technical deep-dive on the chunking algorithm

---

## Success Metrics

- **Week 1**: 100 pip installs
- **Month 1**: 500 pip installs, 50 GitHub stars
- **Month 3**: Featured on Product Hunt homepage, 2000 pip installs

---

## Next Steps to Build This

1. **Use Claude Code or Cursor** to generate all the files based on this spec
2. **Test locally** with Ollama and a small model
3. **Iterate** based on real diff testing
4. **Package** and publish to PyPI
5. **Launch** on Product Hunt

**Ready to generate the actual code?** Upload this spec to Claude Code or I can start writing the files one by one right here!
